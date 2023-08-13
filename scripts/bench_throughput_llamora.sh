#!/bin/bash

# benchmark llamora with 2 gpus
export CUDA_VISIBLE_DEVICES=2,3
python benchmarks/benchmark_throughput.py \
 --backend vllm-lp \
 --dataset /home/jiankun/randomfun/vllm/ShareGPT_V3_unfiltered_cleaned_split.json \
 --model /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
 --tokenizer /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
 --tokenizer-mode slow \
 --num-prompts 4 \
 --tensor-parallel-size 2

#  # benchmark llamora with single gpus
# export CUDA_VISIBLE_DEVICES=2
# python benchmarks/benchmark_throughput.py \
#  --backend vllm-lp \
#  --dataset /home/jiankun/randomfun/vllm/ShareGPT_V3_unfiltered_cleaned_split.json \
#  --model /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
#  --tokenizer /home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter \
#  --tokenizer-mode slow \
#  --num-prompts 4 \
