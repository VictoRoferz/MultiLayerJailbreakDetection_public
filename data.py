"""
data.py

Extract activations from gemma-2-2b-it using mean pooling of last 5 tokens.
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
DATASET_HARMFUL = "TrustAIRLab/in-the-wild-jailbreak-prompts"
DATASET_BENIGN = "ag_news" # Changed from "daily_dialog" to avoid script loading error

def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	elif torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")

def main(args):
	device = get_device()
	print(f"[-] Running on device: {device}")

	# Retrieve HF token from environment (set by previous cell or Colab secrets)
	hf_token = os.environ.get("HF_TOKEN")
	if not hf_token:
		print("[!] Warning: HF_TOKEN environment variable not set. This may cause issues with gated models/datasets.")
	else:
		print("[-] HF_TOKEN found in environment and will be used for authentication.")

	# 1. Load Model & Tokenizer
	print(f"[-] Loading model: {MODEL_NAME}...")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_NAME,
		device_map="auto",
		torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
		output_hidden_states=True,
		token=hf_token
	)
	model.eval()

	# 2. Prepare Datasets
	print("[-] Loading datasets...")
	# Load Harmful (Label 1.0)
	ds_harmful = load_dataset(DATASET_HARMFUL, name='jailbreak_2023_05_07', split=f"train[:{args.n_samples}]", token=hf_token)
	# Load Benign (Label 0.0)
	# Note: 'ag_news' has a 'text' field.
	ds_benign = load_dataset(DATASET_BENIGN, split=f"train[:{args.n_samples}]", token=hf_token)

	# Combine into a processing queue: (text, label)
	# Note: Adjust field names based on dataset inspection if they change versions
	data_queue = []

	for item in ds_harmful:
		# Fallback for various dataset column names
		text = item.get('prompt') or item.get('text') or item.get('question')
		if text: data_queue.append((text, 1.0))

	for item in ds_benign:
		# 'ag_news' uses a 'text' field directly
		text = item.get('text')
		if text: data_queue.append((text, 0.0))

	print(f"[-] Total samples to process: {len(data_queue)}")

	# 3. Extraction Loop
	# Storage: { layer_idx: {'acts': [], 'labels': []} }
	storage = {l: {'acts': [], 'labels': []} for l in args.layers}

	for text, label in tqdm.tqdm(data_queue, desc="Extracting"):
		# Tokenize
		inputs = tokenizer(
			text,
			return_tensors="pt",
			truncation=True,
			max_length=args.max_length
		).to(model.device)

		with torch.no_grad():
			outputs = model(**inputs)

		# Hidden states tuple: (embeddings, layer_1, layer_2, ..., layer_N)
		# Note: hidden_states[0] is embeddings. hidden_states[i] is output of layer i-1.
		# We generally want the output of the layer, so we access index layer_idx + 1
		# or use standard index mapping. For clarity in transformers:
		# hidden_states is a tuple of length (num_layers + 1).

		all_states = outputs.hidden_states

		for layer_idx in args.layers:
			# Bounds check
			if layer_idx >= len(all_states):
				print(f"[!] Warning: Layer {layer_idx} out of bounds. Max {len(all_states)-1}.")
				continue

			# Heuristic: Take the mean of the last tokens (representing the "thought" of the sequence)
			# Taking the last token is common for causal LMs, but mean of last few is more robust.
			current_act = all_states[layer_idx][0, -5:, :].mean(dim=0)

			storage[layer_idx]['acts'].append(current_act.cpu())
			storage[layer_idx]['labels'].append(label)

	# 4. Save Artifacts
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
		labels_tensor = torch.tensor(data['labels'], dtype=torch.float32)

		save_path = layer_dir / "activations.pt"
		torch.save({
			'activations': acts_tensor,
			'labels': labels_tensor,
			'layer': layer_idx,
			'model': MODEL_NAME
		}, save_path)

		print(f"	 -> Layer {layer_idx}: Saved {acts_tensor.shape} to {save_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--layers", type=int, nargs='+', default=[10, 15, 20, 25],
						help="List of layers to extract from (e.g., 10 15 20)")
	parser.add_argument("--n-samples", type=int, default=500, help="Samples per class")
	parser.add_argument("--max-length", type=int, default=128)
	args = parser.parse_args()
	main(args)