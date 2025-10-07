# LLMs-from-scratch
LLM-from-scratch

Build, train, and evaluate a minimalist GPT-style Large Language Model from first principles. This repo walks through tokenization (byte-level BPE), Transformer blocks (multi-head causal self-attention + MLP), training loops in PyTorch, and text sampling. It prioritizes clarity over cleverness, so you can learn, hack, and extend easily.

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10%2B-blue" /> <img src="https://img.shields.io/badge/PyTorch-2.x-red" /> <img src="https://img.shields.io/badge/License-MIT-green" /> <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" /> </p>
Table of Contents

Highlights

Repo Structure

Setup

Quickstart

Training

Sampling / Inference

Configuration

Results

Troubleshooting

Roadmap

Contributing

License

Citation

Acknowledgements

Highlights

Tokenizer & BPE: Train a byte-level BPE tokenizer on any text corpus.

Transformer Core: GPT-like blocks with pre-norm LayerNorm, multi-head attention, GELU MLP.

Causal Masking: Proper autoregressive setup for language modeling.

Training Loop: Grad accumulation, AMP/bfloat16, cosine LR, checkpointing, logging.

Plug-and-Play Data: Start with Tiny Shakespeare, swap in your own .txt corpus.

Evaluation: Loss, perplexity, qualitative text generation (top-k / nucleus sampling).

Repo Structure
LLM-from-scratch/
├─ README.md
├─ requirements.txt
├─ config/
│  ├─ base.yaml
│  └─ tiny_shakespeare.yaml
├─ data/
│  └─ tiny_shakespeare.txt              # (auto-downloaded or place your own corpus)
├─ artifacts/
│  └─ tokenizer.json                    # saved after training tokenizer
├─ src/
│  ├─ tokenizers/
│  │  ├─ bpe_trainer.py                 # learns BPE merges from text
│  │  └─ byte_bpe_tokenizer.py          # encode/decode utilities
│  ├─ model/
│  │  ├─ blocks.py                      # attention + MLP blocks
│  │  ├─ transformer.py                 # GPT-like model definition
│  │  └─ rope.py                        # (optional) rotary embeddings
│  ├─ dataset.py                        # streaming & chunking into blocks
│  ├─ train.py                          # main training script
│  ├─ sampler.py                        # top-k / top-p sampling
│  └─ utils.py                          # logging, checkpointing, seed utils
└─ scripts/
   ├─ prepare_data.py                   # download/clean toy corpora
   ├─ train_tokenizer.py                # train byte-level BPE
   └─ sample.py                         # generate text from a checkpoint

Setup
# 1) Clone
git clone https://github.com/bhuvn24/LLM-from-scratch.git
cd LLM-from-scratch

# 2) (Optional) Create venv/conda
conda create -n llm-scratch python=3.10 -y
conda activate llm-scratch

# 3) Install PyTorch (adjust CUDA as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4) Install project deps
pip install -r requirements.txt


requirements.txt (example):

tqdm
numpy
pyyaml
matplotlib
sentencepiece
pydantic

Quickstart
Prepare a toy dataset
python scripts/prepare_data.py \
  --dataset tiny_shakespeare \
  --out data/tiny_shakespeare.txt

Train a byte-level BPE tokenizer
python scripts/train_tokenizer.py \
  --text_path data/tiny_shakespeare.txt \
  --vocab_size 2000 \
  --out_tokenizer_path artifacts/tokenizer.json

Train a small model (Tiny Shakespeare config)
python src/train.py \
  --config config/tiny_shakespeare.yaml \
  --tokenizer artifacts/tokenizer.json \
  --save_dir runs/ts-base \
  --compile true \
  --amp bf16

Training

Key YAML knobs (also settable via CLI):

data:
  text_path: data/tiny_shakespeare.txt
  train_ratio: 0.9
  block_size: 512

tokenizer:
  path: artifacts/tokenizer.json

model:
  n_layers: 8
  n_heads: 8
  d_model: 512
  d_mlp: 2048
  dropout: 0.0
  max_seq_len: 512
  rope: false

optim:
  lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  warmup_steps: 2000
  total_steps: 100000
  grad_clip: 1.0

train:
  batch_size_tokens: 131072
  eval_every: 1000
  checkpoint_every: 1000
  num_workers: 4
  amp: "bf16"          # "fp16" | "bf16" | "none"
  compile: true        # torch.compile for speed


Tips

Increase batch_size_tokens via gradient accumulation if GPU RAM is limited.

Use --amp bf16 or --amp fp16 and --compile true for speedups.

Save frequent checkpoints for longer runs.

Sampling / Inference

Generate text from a saved checkpoint:

python scripts/sample.py \
  --checkpoint runs/ts-base/ckpt_step_100000.pt \
  --tokenizer artifacts/tokenizer.json \
  --prompt "To be, or not to be" \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.95

Configuration

Two example configs are provided:

config/base.yaml – a template you can copy and modify.

config/tiny_shakespeare.yaml – ready to train on Tiny Shakespeare.

You can override any YAML value via CLI, e.g.:

python src/train.py \
  --config config/tiny_shakespeare.yaml \
  --model.n_layers 6 \
  --model.d_model 384

Results

Replace these with your actual numbers once you train.

Model	Params	Seq Len	Steps	Val Loss ↓	Val PPL ↓
ts-base	~38M	512	100k	~3.4	~30

Notes: Perplexity depends on tokenizer size, LR schedule, seeds, and AMP settings.

Troubleshooting

CUDA OOM
Reduce batch_size_tokens, max_seq_len, or model depth/width. Enable AMP (--amp bf16) and --compile true.

Tokenizer mismatch
Make sure artifacts/tokenizer.json used at sampling is the same one used in training.

Slow dataloading
Increase train.num_workers or pre-shard/cached datasets.

Instability / NaNs
Lower LR (optim.lr), enable gradient clipping, verify AMP mode, and ensure pre-norm LayerNorm.

Roadmap

 FlashAttention-2 backend (if available)

 ALiBi / RoPE toggle with benchmarks

 Mixture-of-Depth or SwiGLU MLP option

 Checkpoint EMA and better eval suite

 Simple RAG demo (FAISS/Chroma) post-pretraining

Contributing

PRs are welcome!

For bugs, open an Issue with clear repro steps.

For features, include tests and docstrings.

For docs, feel free to PR directly.

License

This project is released under the MIT License. See LICENSE
 for details.

Citation

If you find this helpful in research or learning, please cite:

@software{llm_from_scratch_2025,
  author = {Kodikonda Reddy Bhuvan},
  title  = {LLM-from-scratch: A minimalist GPT-style language model},
  year   = {2025},
  url    = {https://github.com/bhuvn24/LLM-from-scratch}
}

Acknowledgements

Inspired by educational “build-from-scratch” efforts across the community and classic Tiny Shakespeare demos. Thanks to the open-source ecosystem for tooling, papers, and ideas that make learning by building possible.
