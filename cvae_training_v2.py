#!/usr/bin/env python3
"""
cvae_training_v2.py

Module 3 — CVAE Training (v2, with fixes for posterior collapse)

Changes from v1:
    1. INPUT NORMALIZATION: z-score activations before training, store stats
       for denormalization at generation time. This puts MSE and KL on
       comparable scales, preventing the KL term from dominating.

    2. WIDER ARCHITECTURE: (d+1) → 1024 → 512 → z=32 instead of 512 → 256.
       The original bottleneck was too aggressive for d=2304.

    3. MUCH LOWER β_max: Start with 0.01 instead of 1.0.
       For high-dimensional activations, β=1.0 is way too strong.
       The encoder collapses rather than fight against it.

    4. CLASS BALANCING: Oversample the minority class so the CVAE sees
       equal numbers of benign and harmful activations per epoch.

    5. KL FREE BITS: Each latent dimension gets a minimum "free" KL
       allowance (λ=0.1). Below that threshold, no penalty is applied.
       This prevents the encoder from being penalized for using a dimension
       at all, encouraging more active units.

Usage:
    python cvae_training_v2.py --layers 10 15 20 25
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import tqdm
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
HARMFUL_DATASET = "TrustAIRLab/in-the-wild-jailbreak-prompts"
HARMFUL_CONFIG = "jailbreak_2023_05_07"


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  1. NORMALIZATION                                                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class ActivationNormalizer:
    """
    Z-score normalization for activations.

    WHY THIS MATTERS:
        Raw activations have norms ~261. MSE loss on these is ~11.0.
        KL divergence for a collapsed posterior is 0.0.
        So even β=0.01 makes the KL term significant relative to
        any marginal reconstruction improvement → encoder gives up → collapse.

        After z-scoring, activations have mean=0, std=1 per dimension.
        MSE on normalized data is ~1.0, making it comparable to KL.
        The encoder can now balance reconstruction vs. regularization.

    The normalizer stores (mean, std) so we can:
        - normalize before encoding
        - denormalize after decoding (for Module 4)
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self._fitted = False

    def fit(self, acts: torch.Tensor):
        """Compute mean and std from training activations."""
        self.mean = acts.mean(dim=0)                    # [d]
        self.std = acts.std(dim=0).clamp(min=1e-8)      # [d], avoid div by 0
        self._fitted = True
        return self

    def normalize(self, acts: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "Call fit() first"
        return (acts - self.mean) / self.std

    def denormalize(self, acts: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "Call fit() first"
        return acts * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d: dict):
        self.mean = d["mean"]
        self.std = d["std"]
        self._fitted = True


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  2. CVAE ARCHITECTURE (wider)                                          ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class CVAEEncoder(nn.Module):
    """
    Wider encoder: (d+1) → 1024 → 512 → (μ, log σ²)

    v1 used 512 → 256 which was too narrow for d=2304.
    Added LayerNorm after each hidden layer for training stability.
    """

    def __init__(self, activation_dim: int, z_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(activation_dim + 1, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        xc = torch.cat([x, c.unsqueeze(-1)], dim=-1)
        h = self.net(xc)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    """
    Wider decoder: (z_dim+1) → 512 → 1024 → d

    Mirrors the encoder. Linear output (activations are unbounded).
    """

    def __init__(self, activation_dim: int, z_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, activation_dim),  # linear output
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor):
        zc = torch.cat([z, c.unsqueeze(-1)], dim=-1)
        return self.net(zc)


class CVAE(nn.Module):
    """
    Full CVAE with normalization support.

    Training: normalized activations in → normalized reconstruction out
    Generation: sample z → decode → denormalize → raw δf
    """

    def __init__(self, activation_dim: int, z_dim: int = 32):
        super().__init__()
        self.encoder = CVAEEncoder(activation_dim, z_dim)
        self.decoder = CVAEDecoder(activation_dim, z_dim)
        self.z_dim = z_dim
        self.activation_dim = activation_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar

    def sample(self, c: float, n_samples: int = 1,
               device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Sample from prior and decode. Returns NORMALIZED space vectors.
        Caller must denormalize if raw activation-space vectors are needed.
        """
        z = torch.randn(n_samples, self.z_dim, device=device)
        c_tensor = torch.full((n_samples,), c, dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.decoder(z, c_tensor)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  3. LOSS WITH FREE BITS                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def compute_beta(epoch: int, warmup_epochs: int = 20, beta_max: float = 0.01) -> float:
    """Linear annealing from 0 to β_max."""
    if epoch < warmup_epochs:
        return beta_max * (epoch / warmup_epochs)
    return beta_max


def cvae_loss(x: torch.Tensor, x_hat: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float, free_bits: float = 0.1):
    """
    β-VAE loss with free bits.

    FREE BITS (Kingma et al., 2016):
        For each latent dimension j, compute KL_j.
        Only penalize KL_j if it exceeds a threshold λ (free_bits).
        This gives each dimension a "free" information budget,
        preventing the encoder from being punished for using z at all.

        Without free bits: encoder kills all dimensions to minimize KL.
        With free bits (λ=0.1): each dimension can encode up to 0.1 nats
        of information for free → more active units.

    Args:
        free_bits: minimum KL per dimension before penalty applies (λ)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")

    # KL per dimension: [batch, z_dim]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Free bits: clamp each dimension's KL to at least λ
    # This means dimensions with KL < λ contribute λ (not their actual KL),
    # removing the incentive to push KL to exactly 0.
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    kl_loss = kl_per_dim.sum(dim=-1).mean()  # sum over dims, mean over batch

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  4. DATASET WITH CLASS BALANCING                                       ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class ActivationDataset(Dataset):
    """Same as v1 but stores class indices for weighted sampling."""

    def __init__(self, benign_acts: torch.Tensor, harmful_acts: torch.Tensor):
        self.activations = torch.cat([benign_acts, harmful_acts], dim=0)
        self.labels = torch.cat([
            torch.zeros(len(benign_acts)),
            torch.ones(len(harmful_acts)),
        ])

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create a sampler that oversamples the minority class.

        With 5000 benign + 500 harmful:
            - Without balancing: ~91% benign batches → CVAE barely learns harmful
            - With balancing: ~50/50 per batch → CVAE learns both classes equally

        Each harmful sample gets weight = N_benign / N_harmful = 10x,
        so it's sampled 10x more often per epoch.
        """
        class_counts = torch.bincount(self.labels.long())
        # Weight = 1 / class_count for each sample's class
        weights_per_class = 1.0 / class_counts.float()
        sample_weights = weights_per_class[self.labels.long()]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self),        # epoch length stays the same
            replacement=True,             # must be True for oversampling
        )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  5. HARMFUL ACTIVATION EXTRACTION                                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def extract_harmful_activations(
    model, tokenizer, layers: list[int],
    n_samples: int = 500, max_length: int = 256, k: int = 5,
) -> dict[int, torch.Tensor]:
    """Extract activations from jailbreak prompts. Same as v1."""
    print(f"[-] Loading harmful dataset: {HARMFUL_DATASET}...")
    ds = load_dataset(HARMFUL_DATASET, name=HARMFUL_CONFIG,
                      split=f"train[:{n_samples}]")

    storage = {l: [] for l in layers}

    for item in tqdm.tqdm(ds, desc="Extracting harmful activations"):
        text = item.get("prompt") or item.get("text") or item.get("question")
        if not text:
            continue

        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        all_states = outputs.hidden_states
        for layer_idx in layers:
            if layer_idx >= len(all_states):
                continue
            seq_len = all_states[layer_idx].shape[1]
            actual_k = min(k, seq_len)
            act = all_states[layer_idx][0, -actual_k:, :].mean(dim=0)
            storage[layer_idx].append(act.cpu().to(torch.float32))

    return {l: torch.stack(acts) for l, acts in storage.items() if acts}


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  6. TRAINING LOOP                                                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def train_cvae_for_layer(
    benign_acts: torch.Tensor,
    harmful_acts: torch.Tensor,
    layer_idx: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CVAE, ActivationNormalizer]:
    """
    Train one CVAE per layer with all v2 fixes applied.

    Returns:
        cvae:       trained model
        normalizer: fitted normalizer (needed for Module 4 denormalization)
    """
    activation_dim = benign_acts.shape[1]

    # ── Step 1: Normalize ──────────────────────────────────────────────
    normalizer = ActivationNormalizer()
    all_raw = torch.cat([benign_acts, harmful_acts], dim=0)
    normalizer.fit(all_raw)

    benign_normed = normalizer.normalize(benign_acts)
    harmful_normed = normalizer.normalize(harmful_acts)

    # Verify normalization worked
    combined = torch.cat([benign_normed, harmful_normed])
    print(f"    Post-normalization: mean={combined.mean():.4f}, "
          f"std={combined.std():.4f} (should be ~0 and ~1)")

    # ── Step 2: Build dataset with balanced sampling ───────────────────
    dataset = ActivationDataset(benign_normed, harmful_normed)
    sampler = dataset.get_balanced_sampler()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,       # balanced sampling instead of shuffle
        drop_last=False,
    )

    # ── Step 3: Initialize model ───────────────────────────────────────
    cvae = CVAE(activation_dim=activation_dim, z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.lr)

    # Optional: learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    print(f"\n{'=' * 60}")
    print(f"  Training CVAE v2 for Layer {layer_idx}")
    print(f"  Activation dim: {activation_dim}")
    print(f"  Latent dim (z): {args.z_dim}")
    print(f"  Dataset: {len(benign_acts)} benign + {len(harmful_acts)} harmful")
    print(f"  Balanced sampling: ON")
    print(f"  Normalization: z-score")
    print(f"  Free bits: {args.free_bits}")
    print(f"  β annealing: 0 → {args.beta_max} over {args.warmup_epochs} epochs")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"{'=' * 60}")

    # ── Step 4: Training ───────────────────────────────────────────────
    cvae.train()
    for epoch in range(args.epochs):
        beta = compute_beta(epoch, args.warmup_epochs, args.beta_max)

        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for acts_batch, labels_batch in dataloader:
            acts_batch = acts_batch.to(device)
            labels_batch = labels_batch.to(device)

            x_hat, mu, logvar = cvae(acts_batch, labels_batch)

            total_loss, recon_loss, kl_loss = cvae_loss(
                acts_batch, x_hat, mu, logvar, beta,
                free_bits=args.free_bits,
            )

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            avg_total = epoch_total / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            print(f"  [L{layer_idx}] Epoch {epoch:3d}/{args.epochs} | "
                  f"β={beta:.4f} | "
                  f"Loss={avg_total:.4f} "
                  f"(Recon={avg_recon:.4f}, KL={avg_kl:.4f})")

    cvae.eval()
    return cvae, normalizer


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  7. MAIN                                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args):
    device = get_device()
    print(f"[-] Device: {device}")

    # Load model for harmful extraction
    print(f"[-] Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        output_hidden_states=True,
    )
    model.eval()

    # Load benign activations from Module 2
    print("[-] Loading benign activations...")
    base_dir = Path("artifacts")
    benign_data = {}
    for layer_idx in args.layers:
        path = base_dir / f"layer_{layer_idx}" / "activations.pt"
        if not path.exists():
            print(f"[!] Missing: {path}")
            continue
        ckpt = torch.load(path, weights_only=True)
        benign_data[layer_idx] = ckpt["activations"].to(torch.float32)
        print(f"    Layer {layer_idx}: {benign_data[layer_idx].shape}")

    if not benign_data:
        print("[!] No benign activations. Run Extraction.py first.")
        return

    # Extract harmful activations
    harmful_data = extract_harmful_activations(
        model, tokenizer,
        layers=list(benign_data.keys()),
        n_samples=args.n_harmful,
        max_length=256,
    )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Train per layer
    for layer_idx in benign_data:
        if layer_idx not in harmful_data:
            print(f"[!] No harmful data for layer {layer_idx}")
            continue

        cvae, normalizer = train_cvae_for_layer(
            benign_acts=benign_data[layer_idx],
            harmful_acts=harmful_data[layer_idx],
            layer_idx=layer_idx,
            args=args,
            device=device,
        )

        # Save CVAE + normalizer together
        save_path = base_dir / f"layer_{layer_idx}" / "cvae_v2.pt"
        torch.save({
            "model_state_dict": cvae.state_dict(),
            "normalizer": normalizer.state_dict(),
            "activation_dim": cvae.activation_dim,
            "z_dim": cvae.z_dim,
            "layer": layer_idx,
            "epochs": args.epochs,
            "beta_max": args.beta_max,
            "free_bits": args.free_bits,
            "version": 2,
        }, save_path)
        print(f"    -> Saved to {save_path}")

    print("\n[-] Module 3 v2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 3: CVAE Training v2")
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 15, 20, 25])
    # Architecture
    parser.add_argument("--z-dim", type=int, default=32)
    # Training — NOTE THE CHANGED DEFAULTS
    parser.add_argument("--epochs", type=int, default=100,
                        help="More epochs (was 50)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Lower LR for stability (was 1e-3)")
    parser.add_argument("--beta-max", type=float, default=0.01,
                        help="MUCH lower β (was 1.0)")
    parser.add_argument("--warmup-epochs", type=int, default=30,
                        help="Longer warmup (was 20)")
    parser.add_argument("--free-bits", type=float, default=0.1,
                        help="Free bits per latent dimension (NEW)")
    # Data
    parser.add_argument("--n-harmful", type=int, default=500)
    args = parser.parse_args()
    main(args)
