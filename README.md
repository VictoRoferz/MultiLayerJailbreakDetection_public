# MultiLayerJailbreakDetection

Detects jailbreaks across multiple neural network layers using Activation Oracles

## Abstract
This work investigates whether a single high-level behavioral concept (e.g., a jailbreak behavior) corresponds to multiple distinct activation patterns inside a large language model (LLM), and aims to learn concept-conditioned mappings from natural-language descriptions to activation-level interventions that can elicit and explore these patterns. We train an agent that takes a natural-language concept and outputs activation-level interventions for a target LLM, enabling us to probe the structure of internal representations underlying that behavior.

## Contents
1. [Setup](#1-setup)
2. [Dataset](#2-dataset)
3. [Train](#3-train)
4. [Evaluation](#4-evaluation)

## 1. Setup
```bash
python -m venv ~/myenv
source ~/myenv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset
```bash
python src/data.py --layers 10 14 18 22
```

## 3. Train
```bash
python src/train.py --layer 14
```

## 4. Evaluation
```bash
python src/eval.py --layer 14
```
