# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a JAX implementation of Test-Time Training (TTT) layers for language modeling. TTT layers are a new class of sequence modeling layers with linear complexity that make the hidden state a machine learning model itself, updated through self-supervised learning at test time.

## Commands

### Training Models
```bash
# Basic training command structure
python3 -m ttt.train \
    --mesh_dim='!-1,1,1' \
    --dtype='bf16' \
    --load_model_config='125m-TTT' \
    --update_model_config="dict(seq_modeling_block='ttt_linear', ttt_base_lr=1.0)" \
    --dataset_path=/path/to/data \
    --dataset_name="the_pile" \
    --seq_length=2048 \
    --global_batch_size=64 \
    --exp_dir=/path/to/experiments \
    --exp_name=experiment_name
```

### Dataset Preparation
```bash
# Tokenize Pile dataset
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s ttt/dataloader/tokenization.py -k "pile"

# Tokenize Books3 dataset
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s ttt/dataloader/tokenization.py -k "books"
```

### Download Pre-tokenized Datasets
```bash
gsutil -m cp -r gs://llama-2-pile/* llama-2-pile/
gsutil -m cp -r gs://llama-2-books3/* llama-2-books3/
```

## Architecture

### Core Components

**TTT Layer Implementations** (`ttt/models/ttt_layer.py`):
- `TTTBase`: Base class for all TTT layers implementing the core test-time training logic
- `TTTLinear`: TTT layer with linear model as hidden state (Mamba backbone)
- `TTTMLP`: TTT layer with two-layer MLP as hidden state (Mamba backbone)
- `TTTLinearBase` / `TTTMLPBase`: Variants for Transformer backbone

**Model Configuration** (`ttt/models/model.py`):
- Contains `ModelConfig` class with all hyperparameters
- Pre-defined configs: '125m-TTT', '350m-TTT', '760m-TTT', '1.3b-TTT'
- Key parameter: `seq_modeling_block` controls which TTT variant to use

**Training Infrastructure** (`ttt/train.py`):
- Main training loop with JAX/Flax
- Handles distributed training via mesh parallelization
- Integrates with WandB for logging

### Key Training Parameters

- `mesh_dim`: Controls JAX parallelization across devices (see EasyLM docs for configuration)
- `seq_length` × `global_batch_size` should equal 0.5 million tokens per batch
- `ttt_base_lr`: Learning rate for TTT inner loop (1.0 for TTT-Linear, 0.1 for TTT-MLP)
- `ttt_base_lr_init` and `ttt_base_lr_warmup`: Additional parameters for TTT-MLP

### Sequence Modeling Block Options
- `ttt_linear`: TTT-Linear with Mamba backbone
- `ttt_mlp`: TTT-MLP with Mamba backbone
- `ttt_linear_base`: TTT-Linear with Transformer backbone
- `ttt_mlp_base`: TTT-MLP with Transformer backbone

## Development Notes

- Python 3.11 required
- JAX-based implementation, supports both GPU and TPU
- Based on EasyLM framework for distributed training
- Dataloader based on FlashAttention
- Scripts in `scripts/` folder organized by model type (ttt_linear, ttt_mlp, transformer)
- Experiments tracked via WandB (requires login)