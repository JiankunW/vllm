#!/bin/bash

# benchmark llamora with single gpus
export CUDA_VISIBLE_DEVICES=2
python vllm/entrypoints/api_server.py \
 --model /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
 --tokenizer /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
 --tokenizer-mode slow \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.8
