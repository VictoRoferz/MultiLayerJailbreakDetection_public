
#!/usr/bin/env python3
"""
Extraction.py

Module 1 — Data Preparation:
    - Loads WikiText-103 as the sole benign corpus.
    - Filters passages to [64, 256] tokens.
    - Splits into N main extraction passages and N_cal calibration hold-out passages.

Module 2 — Activation Extraction:
    - Extracts residual stream activations at layers {10, 15, 20, 25}.
    - Uses the mean of the last 5 token positions as the sequence representation.
    - Saves per-layer activation tensors as .pt artifacts.

Usage:
    python Extraction.py --layers 10 15 20 25 --n-samples 5000 --n-calibration 1000
"""

import os
import argparse
import torch
import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_NAME = "google/gemma-2-2b-it"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Module 1: Data Preparation
# ---------------------------------------------------------------------------
def prepare_dataset(tokenizer, n_main: int, n_calibration: int,
                    min_tokens: int = 64, max_tokens: int = 256) -> tuple:
    """
    Load WikiText-103 and filter passages to [min_tokens, max_tokens].

    Returns:
        main_passages        - list[str] of length <= n_main
        calibration_passages - list[str] of length <= n_calibration
    """
    # load dataset
    print("[-] Loading WikiText-103...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    total_needed = n_main + n_calibration
    filtered: list[str] = []

    # filter dataset
    print(f"[-] Filtering passages to [{min_tokens}, {max_tokens}] tokens...")
    for item in tqdm.tqdm(ds, desc="Filtering passages"):
        text = (item.get("text") or "").strip()
        if not text: continue

        token_len = len(tokenizer.encode(text, add_special_tokens=False))
        if min_tokens <= token_len <= max_tokens:
            filtered.append(text)

        if len(filtered) >= total_needed: break

    main_passages = filtered[:n_main]
    calibration_passages = filtered[n_main:n_main + n_calibration]

    print(f"[-] Main extraction set : {len(main_passages)} passages")
    print(f"[-] Calibration hold-out: {len(calibration_passages)} passages")

    return main_passages, calibration_passages

# ---------------------------------------------------------------------------
# Module 2: Activation Extraction
# ---------------------------------------------------------------------------
def extract_activations(model, tokenizer, passages: list[str],
                        layers: list[int], max_length: int,
                        k: int = 5) -> dict:
    """
    Run the frozen LLM on each passage and extract the mean of the last
    k token positions at each requested layer.

    Returns:
        storage - { layer_idx: { 'acts': list[Tensor] } }
    """
    storage = {l: {"acts": []} for l in layers}

    for text in tqdm.tqdm(passages, desc="Extracting activations"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # hidden_states: tuple of length (num_layers + 1)
        #   index 0 = embedding output
        #   index i = output of transformer layer i
        all_states = outputs.hidden_states

        for layer_idx in layers:
            if layer_idx >= len(all_states):
                print(f"[!] Layer {layer_idx} out of bounds "
                      f"(max {len(all_states) - 1}). Skipping.")
                continue

            # mean of last k token positions
            seq_len = all_states[layer_idx].shape[1]
            k = min(k, seq_len)
            act = all_states[layer_idx][0, -k:, :].mean(dim=0)

            storage[layer_idx]["acts"].append(act.cpu().to(torch.float32))
    return storage

def main(args):
    device = get_device()
    print(f"[-] Running on device: {device}")

    # Load model & tokenizer
    print(f"[-] Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        output_hidden_states=True
    )
    model.eval()

    # --- Module 1: Data Preparation ---
    main_passages, calibration_passages = prepare_dataset(
        tokenizer,
        n_main=args.n_samples,
        n_calibration=args.n_calibration,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    # --- Module 2: Activation Extraction ---
    storage = extract_activations(
        model, tokenizer, main_passages,
        layers=args.layers,
        max_length=args.max_length,
    )

    # Save artifacts
    print("[-] Saving artifacts...")
    base_dir = Path("artifacts")
    base_dir.mkdir(exist_ok=True)

    for layer_idx, data in storage.items():
        if not data['acts']:
            continue
            
        layer_dir = base_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        # Stack into tensors
        acts_tensor = torch.stack(data['acts'])

        save_path = layer_dir / "activations.pt"
        torch.save({
            'activations': acts_tensor,
            'layer': layer_idx,
            'model': MODEL_NAME
        }, save_path)
        
        print(f"    -> Layer {layer_idx}: Saved {acts_tensor.shape} to {save_path}")

    # Save calibration passages for Module 7
    cal_path = base_dir / "calibration_passages.pt"
    torch.save(calibration_passages, cal_path)
    print(f"    -> Saved {len(calibration_passages)} calibration passages to {cal_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers", type=int, nargs='+', default=[10, 15, 20, 25], 
        help="List of layers to extract from (e.g., 10 15 20)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Number of main extraction passages",
    )
    parser.add_argument(
        "--n-calibration", type=int, default=1000,
        help="Number of held-out calibration passages",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=64,
        help="Minimum passage length in tokens",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Maximum passage length in tokens",
    )
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    main(args)
