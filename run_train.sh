#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --use_env --nproc_per_node=1 finetune_train.py
