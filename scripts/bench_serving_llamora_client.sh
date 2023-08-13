#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
python benchmarks/benchmark_serving.py \
 --backend vllm-lp \
 --tokenizer /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
 --dataset /home/jiankun/randomfun/vllm/ShareGPT_V3_unfiltered_cleaned_split.json \
 --num-prompts 2000 \
 --request-rate 200
 