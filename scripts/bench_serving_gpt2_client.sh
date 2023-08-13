#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
python benchmarks/benchmark_serving.py \
 --backend vllm \
 --tokenizer gpt2 \
 --dataset /home/jiankun/randomfun/vllm/ShareGPT_V3_unfiltered_cleaned_split.json \
 --num-prompts 2000 \
 --request-rate 200