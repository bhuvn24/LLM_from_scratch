# LLM-from-scratch

Build, train, and evaluate a minimalist GPT-style Large Language Model from first principles. This repo walks through tokenization (byte-level BPE), Transformer blocks (multi-head causal self-attention + MLP), training loops in PyTorch, and text sampling. It prioritizes clarity over cleverness, so you can learn, hack, and extend easily.

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10%2B-blue" /> <img src="https://img.shields.io/badge/PyTorch-2.x-red" /> <img src="https://img.shields.io/badge/License-MIT-green" /> <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" /> </p>



[![GitHub Repo stars](https://img.shields.io/github/stars/bhuvn24/LLM_from_scratch?style=social)](https://github.com/bhuvn24/LLM_from_scratch/stargazers)
[![GitHub license](https://img.shields.io/github/license/bhuvn24/LLM_from_scratch)](https://github.com/bhuvn24/LLM_from_scratch/blob/main/LICENSE)

## Overview

This repository provides a step-by-step implementation of a minimalist GPT-style Large Language Model (LLM) built from scratch using PyTorch. The focus is on educational value, emphasizing clarity and simplicity over optimization or performance tricks. You'll learn the core components of modern LLMs, including tokenization, Transformer architecture, training procedures, and inference.

Key features:
- **Tokenization**: Byte-level Byte Pair Encoding (BPE) for handling text data.
- **Transformer Blocks**: Multi-head causal self-attention and feed-forward MLPs.
- **Training Loop**: Full PyTorch-based training on a small dataset.
- **Text Sampling**: Generate text from the trained model.
- Designed for easy experimentation, hacking, and extension.

This project is inspired by resources like Andrej Karpathy's "nanoGPT" but aims to be even more beginner-friendly.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Generating Text](#generating-text)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Tokenization](#tokenization)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended for GPU acceleration)
- Basic knowledge of Python, deep learning, and PyTorch

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/bhuvn24/LLM_from_scratch.git
   cd LLM_from_scratch
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   (Note: If `requirements.txt` is not present, install core dependencies manually: `pip install torch numpy`)

3. (Optional) Prepare a small dataset, e.g., download Shakespeare's works or use any text corpus.

## Usage

### Training the Model

Run the training script with default parameters:
```
python train.py --dataset path/to/dataset.txt --epochs 10 --batch_size 32
```

Key arguments:
- `--dataset`: Path to the text file for training.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.
- `--model_size`: Size of the model (e.g., 'small', 'medium') – adjusts embedding dim, heads, etc.
- `--save_path`: Where to save the trained model checkpoint.

Example:
```
python train.py --dataset data/shakespeare.txt --epochs 5 --batch_size 64 --save_path models/mini_gpt.pth
```

### Generating Text

Load a trained model and generate samples:
```
python generate.py --model_path models/mini_gpt.pth --prompt "Once upon a time" --max_length 200
```

Key arguments:
- `--model_path`: Path to the saved model checkpoint.
- `--prompt`: Initial text to start generation.
- `--max_length`: Maximum number of tokens to generate.
- `--temperature`: Sampling temperature (higher = more creative, default 0.7).

Example output:
```
Once upon a time, in a land far away, there lived a brave knight...
```

## Project Structure

```
LLM_from_scratch/
├── data/                # Sample datasets (e.g., shakespeare.txt)
├── models/              # Saved model checkpoints
├── src/
│   ├── tokenizer.py     # Byte-level BPE tokenizer implementation
│   ├── model.py         # Transformer model definition
│   ├── train.py         # Training loop
│   └── generate.py      # Inference and sampling script
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── LICENSE              # License file (e.g., MIT)
```

## How It Works

### Tokenization

We implement a byte-level Byte Pair Encoding (BPE) tokenizer from scratch:
- Merges frequent byte pairs iteratively.
- Handles vocabulary building and encoding/decoding.
- Example: Tokenize text into subword units for efficient model input.

### Model Architecture

The core is a GPT-like Transformer:
- **Embedding Layer**: Token and position embeddings.
- **Transformer Blocks**: Stack of blocks with:
  - Multi-head causal self-attention (masked to prevent future peeking).
  - Feed-forward MLP (two linear layers with GELU activation).
- **Output Head**: Linear layer to predict next token logits.
- Hyperparameters: Configurable embedding dim, number of heads, layers, etc.

### Training

- Uses cross-entropy loss for next-token prediction.
- AdamW optimizer with learning rate scheduling.
- Supports GPU/CPU training.
- Logs loss and saves checkpoints periodically.

## Contributing

Contributions are welcome! Feel free to:
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Please ensure code is clean, commented, and follows PEP8 style.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Andrej Karpathy's nanoGPT and minGPT repositories.
- Thanks to the PyTorch team for an excellent framework.
- Dataset sources: Project Gutenberg (e.g., Shakespeare texts).
