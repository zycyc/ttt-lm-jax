#!/bin/bash

DATA_PATH=/data/users/alan/ttt-lm-jax/llama-2-pile
DATA_NAME="the_pile" # "books3"

# Product should equal 0.5 million
SEQ_LEN=2048
BS=64

# Experiment details
EXP_NAME=ttt_linear_125m
EXP_DIR=/data/users/alan/ttt-lm-jax/experiments

export CUDA_VISIBLE_DEVICES=4,5,6,7

python3 -m ttt.train \
        --mesh_dim='!-1,1,1' \
        --dtype='bf16' \
        --total_steps=4800 \
        --save_checkpoint_freq=1000 \
        --save_milestone_freq=2000 \
        --load_model_config='125m-TTT' \
        --update_model_config="dict(seq_modeling_block='ttt_linear', ttt_base_lr=1.0)" \
        --dataset_path=${DATA_PATH} \
        --dataset_name=${DATA_NAME} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --optimizer.type='adamw' \
        --optimizer.adamw_optimizer.weight_decay=0.1 \
        --optimizer.adamw_optimizer.lr=3e-3 \
        --optimizer.adamw_optimizer.end_lr=1e-5 \
        --optimizer.adamw_optimizer.lr_warmup_steps=480 \
        --optimizer.adamw_optimizer.lr_decay_steps=4800 \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME}