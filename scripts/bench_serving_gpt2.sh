#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
python vllm/entrypoints/api_server.py \
 --model gpt2 \
 --tensor-parallel-size 2 \
 --gpu-memory-utilization 0.036
 