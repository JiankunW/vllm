#!/bin/bash

# benchmark llamora with single gpus
export CUDA_VISIBLE_DEVICES=2
python vllm/entrypoints/api_server.py \
 --model gpt2 \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.8
