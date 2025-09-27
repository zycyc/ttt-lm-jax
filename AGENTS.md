# AGENTS Guide

## Environment
- Python 3.11 with JAX; Linux systems tested on NVIDIA GPUs and Google Cloud TPU VMs.
- Use Conda or virtualenv to isolate dependencies; CUDA/cuDNN must match the JAX wheels you install.
- Set `PYTHONPATH=$PWD:$PYTHONPATH` before running scripts so intra-package imports resolve.
- Optional: `wandb login` if you want experiment telemetry.

## Installation
1. Create an environment, e.g. `conda create -n ttt-lm python=3.11` and `conda activate ttt-lm`.
2. Install GPU deps: `pip install -r requirements/gpu_requirements.txt` (for TPU use `tpu_requirements.txt`).
3. (GPU only) Install the matching JAX wheel if not pulled in automatically, e.g. `pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`.
4. Verify CUDA visibility with `python -c "import jax; print(jax.devices())"`.

## Datasets
- Preferred: download pre-tokenized corpora (`gsutil -m cp -r gs://llama-2-pile/* llama-2-pile/` etc.) and point `--dataset_path` at the directory containing `tokenizer_name-meta-llama/`.
- To self-tokenize The Pile or Books3, edit the TODO paths in `ttt/dataloader/tokenization.py`, export `PYTHONPATH`, and run the pytest commands below.

## Training / Experiments
- Primary entry point: `python -m ttt.train` with flags defined in `ttt/train.py`.
- Use the helper scripts in `scripts/{ttt_linear,ttt_mlp,transformer}` as templates; fill in `DATA_PATH`, `EXP_DIR`, `EXP_NAME`, and run after adjusting `mesh_dim`, `seq_length`, and batch sizing (product should be ~0.5M tokens per batch as in the paper).
- JAX distributed configs and model settings live in the training flags (`--load_model_config`, `--update_model_config`, etc.). Keep `mesh_dim` consistent with your hardware topology.
- Checkpoints write to `--exp_dir/--exp_name`; ensure the process has write permission.

## Testing
- Tokenizer/data pipeline regression: `pytest -q -s ttt/dataloader/tokenization.py -k "pile"` or `-k "books"`. Requires the raw datasets and enough disk to materialize the tokenized cache.
- No lightweight unit suite exists; expect multi-hour runs on large CPU machines when (re-)tokenizing. Consider running on subsets during development by editing the test constants locally.

## Tips
- Large-scale runs assume high-memory accelerators; monitor `mesh_dim` and optimizer hyperparameters when scaling down.
- Set `dataset_path`, `dataset_name`, and WandB environment variables (`WANDB_PROJECT`, etc.) explicitly in automation.
- For TPU VMs follow Google Cloud quick-start, then install `requirements/tpu_requirements.txt` before launching experiments.
