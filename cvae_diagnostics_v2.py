"""
cvae_diagnostics_v2.py

Validation suite for CVAE v2 (with normalization, free bits, wider arch).
Same 6 tests as v1 but handles:
    - Loading cvae_v2.pt checkpoint format
    - ActivationNormalizer (normalize inputs, denormalize outputs)
    - Imports from cvae_training_v2

Usage:
    python cvae_diagnostics_v2.py --layer 20
    python cvae_diagnostics_v2.py --layer 20 --save-plots
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import from v2 module
from cvae_training_v2 import (
    CVAE, ActivationNormalizer, ActivationDataset, cvae_loss, compute_beta,
    extract_harmful_activations, get_device, MODEL_NAME,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  LOADERS                                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def load_cvae_v2(layer_idx: int, device: torch.device):
    """Load CVAE v2 checkpoint + normalizer."""
    path = Path("artifacts") / f"layer_{layer_idx}" / "cvae_v2.pt"
    if not path.exists():
        raise FileNotFoundError(f"No v2 CVAE at {path}. Run cvae_training_v2.py first.")

    ckpt = torch.load(path, weights_only=False, map_location=device)

    cvae = CVAE(
        activation_dim=ckpt["activation_dim"],
        z_dim=ckpt["z_dim"],
    ).to(device)
    cvae.load_state_dict(ckpt["model_state_dict"])
    cvae.eval()

    normalizer = ActivationNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])

    return cvae, normalizer, ckpt


def load_benign_activations(layer_idx: int) -> torch.Tensor:
    path = Path("artifacts") / f"layer_{layer_idx}" / "activations.pt"
    ckpt = torch.load(path, weights_only=True)
    return ckpt["activations"].to(torch.float32)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 1: Reconstruction Quality                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def test_reconstruction(cvae, normalizer, benign_acts, harmful_acts, device):
    """
    Encode → decode in NORMALIZED space, then denormalize and compare
    against the RAW originals. This tests end-to-end fidelity.

    WHAT TO LOOK FOR:
        - Cosine similarity > 0.90 (good) or > 0.95 (excellent)
        - Both classes should reconstruct well
    """
    print("\n" + "=" * 60)
    print("  TEST 1: Reconstruction Quality")
    print("=" * 60)

    results = {}
    for name, acts, label in [("Benign", benign_acts, 0.0),
                               ("Harmful", harmful_acts, 1.0)]:
        x_raw = acts[:200]
        x_norm = normalizer.normalize(x_raw).to(device)
        c = torch.full((len(x_norm),), label, device=device)

        with torch.no_grad():
            x_hat_norm, mu, logvar = cvae(x_norm, c)

        # Denormalize reconstruction back to raw space for comparison
        x_hat_raw = normalizer.denormalize(x_hat_norm.cpu())
        x_raw_np = x_raw.numpy()
        x_hat_np = x_hat_raw.numpy()

        # Cosine similarity in RAW space (what Module 4 cares about)
        cos_sim = F.cosine_similarity(x_raw, x_hat_raw, dim=-1).mean().item()

        # Relative MSE
        mse = F.mse_loss(x_hat_raw, x_raw).item()
        x_norm_sq = (x_raw ** 2).sum(dim=-1).mean().item()
        rel_mse = mse / x_norm_sq if x_norm_sq > 0 else float("inf")

        # Pearson correlation
        corr = np.corrcoef(x_raw_np.flatten(), x_hat_np.flatten())[0, 1]

        results[name] = {"cos_sim": cos_sim, "rel_mse": rel_mse, "corr": corr}

        status = "✓" if cos_sim > 0.90 else "✗"
        print(f"  {status} {name:8s} | Cosine Sim: {cos_sim:.4f} | "
              f"Relative MSE: {rel_mse:.6f} | Pearson r: {corr:.4f}")

    return results


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 2: Posterior Collapse Check                                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def test_posterior_collapse(cvae, normalizer, benign_acts, harmful_acts, device):
    """
    Count active latent dimensions.
    v2 should show MANY more active units than v1's 1/32.
    """
    print("\n" + "=" * 60)
    print("  TEST 2: Posterior Collapse Check")
    print("=" * 60)

    all_mu = []
    all_logvar = []

    for acts, label in [(benign_acts, 0.0), (harmful_acts, 1.0)]:
        x_norm = normalizer.normalize(acts[:500]).to(device)
        c = torch.full((len(x_norm),), label, device=device)
        with torch.no_grad():
            mu, logvar = cvae.encoder(x_norm, c)
        all_mu.append(mu.cpu())
        all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    # Active units
    mu_var = all_mu.var(dim=0)
    active = (mu_var > 0.01).sum().item()
    total = all_mu.shape[1]

    # KL per dimension
    kl_per_dim = -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())
    mean_kl = kl_per_dim.mean(dim=0)

    # Mean σ²
    mean_sigma_sq = all_logvar.exp().mean(dim=0)

    status = "✓" if active > total * 0.5 else "✗"
    print(f"  {status} Active latent units: {active}/{total}")
    print(f"    Var(μ) per dim: min={mu_var.min():.4f}, "
          f"max={mu_var.max():.4f}, median={mu_var.median():.4f}")
    print(f"    Mean KL per dim: min={mean_kl.min():.4f}, "
          f"max={mean_kl.max():.4f}, mean={mean_kl.mean():.4f}")
    print(f"    Mean σ²: min={mean_sigma_sq.min():.4f}, "
          f"max={mean_sigma_sq.max():.4f}")

    if active < total * 0.25:
        print("  ⚠ WARNING: Still significant collapse. Try β_max=0.001")

    return {"active_units": active, "total_dims": total,
            "mu_var_per_dim": mu_var, "mean_kl_per_dim": mean_kl}


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 3: Latent Space Separation                                       ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def test_latent_separation(cvae, normalizer, benign_acts, harmful_acts,
                           device, save_plots=False, layer_idx=0):
    """
    Are benign vs harmful separable in latent space?
    This is the most important test for Module 4 to work.
    """
    print("\n" + "=" * 60)
    print("  TEST 3: Latent Space Separation (benign vs. harmful)")
    print("=" * 60)

    n = min(500, len(benign_acts), len(harmful_acts))
    mus = []
    labels = []

    for acts, label in [(benign_acts[:n], 0.0), (harmful_acts[:n], 1.0)]:
        x_norm = normalizer.normalize(acts).to(device)
        c = torch.full((len(x_norm),), label, device=device)
        with torch.no_grad():
            mu, _ = cvae.encoder(x_norm, c)
        mus.append(mu.cpu())
        labels.extend([label] * len(acts))

    mu_all = torch.cat(mus).numpy()
    labels = np.array(labels)

    # Centroid distance
    centroid_b = mu_all[labels == 0].mean(axis=0)
    centroid_h = mu_all[labels == 1].mean(axis=0)
    cos_dist = 1 - np.dot(centroid_b, centroid_h) / (
        np.linalg.norm(centroid_b) * np.linalg.norm(centroid_h) + 1e-8
    )

    # Linear probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, mu_all, labels, cv=5, scoring="accuracy")
    acc = scores.mean()

    status_d = "✓" if cos_dist > 0.05 else "✗"
    status_a = "✓" if acc > 0.70 else "✗"
    print(f"  {status_d} Centroid cosine distance: {cos_dist:.4f}")
    print(f"  {status_a} Linear probe accuracy: {acc:.3f} ± {scores.std():.3f}")

    # Plot
    if HAS_PLOTTING and save_plots:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(mu_all)

        fig, ax = plt.subplots(figsize=(8, 6))
        mask_b = labels == 0
        mask_h = labels == 1
        ax.scatter(X_2d[mask_b, 0], X_2d[mask_b, 1],
                   alpha=0.4, s=10, label="Benign", c="steelblue")
        ax.scatter(X_2d[mask_h, 0], X_2d[mask_h, 1],
                   alpha=0.4, s=10, label="Harmful", c="tomato")
        ax.set_title(f"Layer {layer_idx} — Latent Space (PCA of μ)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend()
        path = Path("artifacts") / f"layer_{layer_idx}" / "latent_separation_v2.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    → Plot saved to {path}")

    return {"cos_dist": cos_dist, "linear_probe_acc": acc}


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 4: Generation Drift (c=0 vs c=1)                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def test_generation_drift(cvae, normalizer, benign_acts, device):
    """
    Sample with c=0 and c=1, denormalize both, compare in raw space.
    """
    print("\n" + "=" * 60)
    print("  TEST 4: Generation Drift (c=0 vs c=1)")
    print("=" * 60)

    n = 500
    # Sample in normalized space
    samples_0_norm = cvae.sample(c=0.0, n_samples=n, device=device)
    samples_1_norm = cvae.sample(c=1.0, n_samples=n, device=device)

    # Denormalize to raw activation space
    samples_0 = normalizer.denormalize(samples_0_norm.cpu())
    samples_1 = normalizer.denormalize(samples_1_norm.cpu())

    # Centroid comparison in raw space
    centroid_0 = samples_0.mean(dim=0)
    centroid_1 = samples_1.mean(dim=0)
    cos_sim = F.cosine_similarity(
        centroid_0.unsqueeze(0), centroid_1.unsqueeze(0)
    ).item()

    # Drift relative to benign activation scale
    l2_dist = (centroid_0 - centroid_1).norm().item()
    benign_scale = benign_acts[:200].norm(dim=-1).mean().item()
    relative_drift = l2_dist / benign_scale

    # How far are c=1 samples from the actual benign centroid?
    benign_centroid = benign_acts.mean(dim=0)
    dist_0 = (samples_0 - benign_centroid).norm(dim=-1).mean().item()
    dist_1 = (samples_1 - benign_centroid).norm(dim=-1).mean().item()

    status = "✓" if cos_sim < 0.95 else "✗"
    print(f"  {status} Centroid cosine sim (c=0 vs c=1): {cos_sim:.4f}")
    print(f"    L2 drift / benign scale: {relative_drift:.4f}")
    print(f"    Distance to benign centroid: c=0 → {dist_0:.2f}, c=1 → {dist_1:.2f}")

    if cos_sim > 0.99:
        print("  ⚠ Decoder ignoring concept label.")

    return {"cos_sim_01": cos_sim, "relative_drift": relative_drift}


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 5: Perturbation Norm Distribution                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def test_perturbation_norms(cvae, normalizer, benign_acts, device):
    """
    Check diversity and scale of generated perturbations in RAW space.
    """
    print("\n" + "=" * 60)
    print("  TEST 5: Perturbation Norm Distribution")
    print("=" * 60)

    n = 1000
    # Generate in normalized space, denormalize
    delta_norm = cvae.sample(c=1.0, n_samples=n, device=device)
    delta_raw = normalizer.denormalize(delta_norm.cpu())

    norms = delta_raw.norm(dim=-1)
    benign_norms = benign_acts[:200].norm(dim=-1)

    cv = (norms.std() / norms.mean()).item() if norms.mean() > 0 else 0

    print(f"  Raw δf norms:     mean={norms.mean():.2f}, "
          f"std={norms.std():.2f}, min={norms.min():.2f}, max={norms.max():.2f}")
    print(f"  Benign act norms: mean={benign_norms.mean():.2f}, "
          f"std={benign_norms.std():.2f}")

    status = "✓" if cv > 0.1 else "✗"
    print(f"  {status} Coefficient of variation: {cv:.4f}")

    # Also check diversity via pairwise cosine similarity
    # (if all perturbations point the same direction → mode collapse)
    subset = delta_raw[:100]
    pairwise_cos = F.cosine_similarity(
        subset.unsqueeze(0), subset.unsqueeze(1), dim=-1
    )
    # Exclude diagonal (self-similarity = 1)
    mask = ~torch.eye(len(subset), dtype=torch.bool)
    mean_pairwise = pairwise_cos[mask].mean().item()

    status2 = "✓" if mean_pairwise < 0.95 else "✗"
    print(f"  {status2} Mean pairwise cosine sim: {mean_pairwise:.4f} "
          f"(<0.95 = diverse directions)")

    if not torch.isfinite(norms).all():
        print("  ⚠ CRITICAL: Non-finite norms detected.")

    return {"norm_mean": norms.mean().item(), "norm_cv": cv,
            "pairwise_cos": mean_pairwise}


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 6: Final Loss vs. Trivial Baseline                              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def test_final_loss(cvae, normalizer, benign_acts, harmful_acts, device,
                    beta=0.01, free_bits=0.1):
    """
    Compare CVAE reconstruction loss against the trivial baseline
    (predicting the dataset mean). Works in NORMALIZED space.
    """
    print("\n" + "=" * 60)
    print("  TEST 6: Final Loss Decomposition")
    print("=" * 60)

    total_recon = 0
    total_kl = 0
    n_total = 0

    for acts, label in [(benign_acts, 0.0), (harmful_acts, 1.0)]:
        x_norm = normalizer.normalize(acts).to(device)
        c = torch.full((len(x_norm),), label, device=device)
        with torch.no_grad():
            x_hat, mu, logvar = cvae(x_norm, c)
            _, recon, kl = cvae_loss(x_norm, x_hat, mu, logvar, beta, free_bits)
        total_recon += recon.item() * len(acts)
        total_kl += kl.item() * len(acts)
        n_total += len(acts)

    avg_recon = total_recon / n_total
    avg_kl = total_kl / n_total

    # Trivial baseline in normalized space (predict 0 = the mean)
    # After z-scoring, MSE of predicting zero = mean of sum of x² ≈ 1.0 per dim
    all_norm = normalizer.normalize(torch.cat([benign_acts, harmful_acts]))
    trivial_mse = (all_norm ** 2).mean().item()

    improvement = 1 - (avg_recon / trivial_mse) if trivial_mse > 0 else 0

    status = "✓" if improvement > 0.5 else "✗"
    print(f"  {status} Final loss: {avg_recon + beta * avg_kl:.4f} "
          f"(Recon={avg_recon:.4f} + β·KL={beta * avg_kl:.4f})")
    print(f"    Trivial baseline (predict mean): {trivial_mse:.4f}")
    print(f"    Improvement over baseline: {improvement:.1%}")
    print(f"    KL divergence: {avg_kl:.4f}")

    if improvement < 0.2:
        print("  ⚠ Still underperforming. Check training logs.")

    return {"recon": avg_recon, "kl": avg_kl, "improvement": improvement}


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SUMMARY                                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def print_summary(results, layer_idx):
    print("\n" + "=" * 60)
    print(f"  SUMMARY — Layer {layer_idx} (CVAE v2)")
    print("=" * 60)

    checks = [
        ("Reconstruction (cos_sim > 0.90)",
         results.get("recon", {}).get("Benign", {}).get("cos_sim", 0) > 0.90),
        ("No posterior collapse (active > 50%)",
         results.get("collapse", {}).get("active_units", 0) >
         results.get("collapse", {}).get("total_dims", 32) * 0.5),
        ("Latent separation (probe > 0.70)",
         results.get("separation", {}).get("linear_probe_acc", 0) > 0.70),
        ("Conditioning works (cos_sim < 0.95)",
         results.get("drift", {}).get("cos_sim_01", 1.0) < 0.95),
        ("Diverse perturbations (CV > 0.1)",
         results.get("norms", {}).get("norm_cv", 0) > 0.1),
        ("Diverse directions (pairwise < 0.95)",
         results.get("norms", {}).get("pairwise_cos", 1.0) < 0.95),
        ("Beats trivial baseline (>50%)",
         results.get("loss", {}).get("improvement", 0) > 0.5),
    ]

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")

    print(f"\n  Result: {passed}/{len(checks)} checks passed")
    if passed >= 6:
        print("  → CVAE v2 looks healthy. Proceed to Module 4.")
    elif passed >= 4:
        print("  → Mostly OK. Review failing checks.")
    else:
        print("  → Still problems. See suggestions below.")
        print("    • If collapse persists: --beta-max 0.001")
        print("    • If no separation: more harmful data (--n-harmful 1000)")
        print("    • If low diversity: --z-dim 64")

    # v1 vs v2 comparison hint
    print(f"\n  v1 → v2 key metrics to compare:")
    print(f"    Active units:   1/32 → {results.get('collapse', {}).get('active_units', '?')}/32")
    print(f"    Linear probe:   0.537 → {results.get('separation', {}).get('linear_probe_acc', '?'):.3f}")
    print(f"    Baseline impr:  2.0% → {results.get('loss', {}).get('improvement', 0):.1%}")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def main(args):
    device = get_device()
    print(f"[-] Device: {device}")

    # Load CVAE v2 + normalizer
    cvae, normalizer, ckpt = load_cvae_v2(args.layer, device)
    print(f"[-] Loaded CVAE v2 for layer {args.layer} "
          f"(d={ckpt['activation_dim']}, z={ckpt['z_dim']}, "
          f"epochs={ckpt['epochs']}, β_max={ckpt['beta_max']}, "
          f"free_bits={ckpt.get('free_bits', 'N/A')})")

    # Load benign activations
    benign_acts = load_benign_activations(args.layer)
    print(f"[-] Benign activations: {benign_acts.shape}")

    # Extract harmful activations
    print("[-] Loading model for harmful extraction...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        output_hidden_states=True,
    )
    model.eval()

    harmful_data = extract_harmful_activations(
        model, tokenizer, layers=[args.layer],
        n_samples=args.n_harmful, max_length=256,
    )
    harmful_acts = harmful_data[args.layer]
    print(f"[-] Harmful activations: {harmful_acts.shape}")

    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Run all tests
    results = {}
    results["recon"] = test_reconstruction(
        cvae, normalizer, benign_acts, harmful_acts, device)
    results["collapse"] = test_posterior_collapse(
        cvae, normalizer, benign_acts, harmful_acts, device)
    results["separation"] = test_latent_separation(
        cvae, normalizer, benign_acts, harmful_acts, device,
        save_plots=args.save_plots, layer_idx=args.layer)
    results["drift"] = test_generation_drift(
        cvae, normalizer, benign_acts, device)
    results["norms"] = test_perturbation_norms(
        cvae, normalizer, benign_acts, device)
    results["loss"] = test_final_loss(
        cvae, normalizer, benign_acts, harmful_acts, device,
        beta=ckpt["beta_max"],
        free_bits=ckpt.get("free_bits", 0.1))

    print_summary(results, args.layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVAE v2 Diagnostics")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--n-harmful", type=int, default=500)
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()
    main(args)