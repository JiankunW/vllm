#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
python vllm/entrypoints/api_server.py \
 --model /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
 --tokenizer /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
 --tokenizer-mode slow \
 --tensor-parallel-size 2 \
 --gpu-memory-utilization 0.4
