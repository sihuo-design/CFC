# CFC: Coarse-to-Fine Open-Set Graph Node Classification

**The official code for "Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models" (AAAI 2026).**

---

## Overview

**CFC (Coarse-to-Fine Classification)** is an open-set graph node classification framework that leverages Large Language Models (LLMs) for both in-distribution (ID) recognition and out-of-distribution (OOD) labeling.

Unlike existing methods that treat all OOD samples as a single undifferentiated class, CFC assigns interpretable, probable labels to OOD nodes.

**Key capabilities:**
- In-distribution (ID) node classification
- OOD detection (seen vs. unseen class separation)
- Probable OOD label assignment for improved interpretability
- Support for multiple graph datasets and LLM backends

---

**Training techniques:**
- `denoising_mixup` (default): combines label propagation denoising with feature-space mixup between ID boundary nodes and pseudo-OOD nodes
- `denoising`: label propagation only
- `mixup`: feature mixup only
- `graph_mixup`: graph-structure-aware mixup

---

## Supported Datasets

| Dataset | Domain | Notes |
|---------|--------|-------|
| `cora` | Citation network | Default |
| `citeseer` | Citation network | |
| `dblp` | Citation network | |
| `pubmed` | Citation network | |
| `wikics` | Wikipedia graph | |
| `arxiv` | ArXiv paper network | |
| `elecomp` | Amazon Electronics (Computers) | |
| `elephoto` | Amazon Electronics (Photo) | |

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/sihuo-design/CFC.git
cd CFC
```

**2. Create and activate a Conda environment**

```bash
conda create -n cfc_env python=3.9
conda activate cfc_env
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** Requires CUDA-compatible GPU. Tested with PyTorch 2.1.0 + CUDA 11.8.

---

## Configuration

### OpenAI API Key

Set your OpenAI API key via the `--key` argument or by editing the default in `main.py`:

set your key in main.py

```bash
python main.py --key YOUR_OPENAI_API_KEY ...
```

### LLM Backends

| `--llm_name` | Description |
|---|---|
| `e5` (default) | E5-large-v2 text embeddings (1024-dim) |
| `ST` | Sentence-Transformers (all-mpnet-base-v2, 768-dim) |
| `llama2_7b` | LLaMA-2 7B embeddings (4096-dim) |
| `minilm` | MiniLM (384-dim, fastest) |
| `tfidf` | TF-IDF + SVD (300-dim, no GPU required) |

| `--llm_model_method` | Description |
|---|---|
| `gpt-4o` (default) | GPT-4o for OOD label generation |
| `gpt-3.5-turbo` | Faster, lower cost |

---

## Usage

### Basic Run (Cora, default settings)

```bash
mkdir -p results
python main.py \
    --dataset cora \
    --model_type llm \
    --lr 0.01 \
    --seen_unseen_classifier llm_test \
    --fine_method denoising_mixup \
    --seed 100 \
    --key YOUR_OPENAI_API_KEY
```

### Run All Seeds (reproduce paper results)

```bash
bash run.sh
```

### OOD Label Classification with Llama

```bash
bash ood.sh
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cora` | Graph dataset |
| `--model_type` | `llm` | Recognition method |
| `--backbone` | `GCN_model2` | GNN backbone (`GCN_model2` or `MLP`) |
| `--lr` | `0.01` | Learning rate |
| `--epoch` | `200` | Training epochs |
| `--seed` | `100` | Random seed |
| `--known_class` | `[0,2,3,5]` | ID class indices |
| `--fine_method` | `denoising_mixup` | Training augmentation method |
| `--seen_unseen_classifier` | `llm_train` | Coarse ID/OOD classifier (`llm_train` / `llm_test`) |
| `--llm_name` | `e5` | Text encoder for node features |
| `--llm_model_method` | `gpt-4o` | LLM for OOD label generation |
| `--key` | — | OpenAI API key |
| `--select_num` | `50` | Number of pseudo-OOD nodes to select |
| `--mixup_num` | `100` | Number of boundary nodes for mixup |
| `--temperature` | `0.1` | LLM decoding temperature |
| `--train_rate` | `0.5` | Training split ratio |
| `--valid_rate` | `0.3` | Validation split ratio |

---

## Evaluation Metrics

The framework reports three metrics on the test set:

- **Closed-set accuracy** — Classification accuracy on ID nodes
- **Open-set accuracy** — Detection accuracy for OOD nodes
- **Overall accuracy** — Combined accuracy across all test nodes

Results are saved to `results/<model_type>/<dataset>/<seed>/<fine_method>/`.

---

## Project Structure

```
CFC_github/
├── main.py              # Main training and evaluation script
├── model.py             # GNN model definitions (GCN, MLP, GIN, GAT)
├── llm.py               # LLM API calls (OpenAI, Llama)
├── utils.py             # SentenceEncoder, TF-IDF utilities
├── asyncapi.py          # Async OpenAI API for batch inference
├── run.sh               # Experiment runner (multiple seeds)
├── ood.sh               # OOD label classification with Llama
├── requirements.txt     # Python dependencies
├── models/
│   ├── GCN_model2.py    # GCN with dummy class support
│   ├── utils.py         # Training utilities
│   └── __init__.py
├── data/
│   ├── gen_data.py      # OFA-style graph dataset loader
│   └── ofa_data.py      # PyG dataset wrapper
├── openail/
│   ├── config.py        # API configuration
│   └── utils.py         # Label mapping utilities
├── main_results/        # Saved experiment results
└── result_1v1/          # LLM 1-vs-1 comparison results
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ma2026coarse,
  title={Coarse-to-fine open-set graph node classification with large language models},
  author={Ma, Xueqi and Ma, Xingjun and Erfani, Sarah Monazam and Mandic, Danilo and Bailey, James},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={43},
  pages={36714--36722},
  year={2026}
}
```
