# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs
from vllm.model_executor.utils import (divide_list_by_mask, 
                                       divide_tensor_by_mask, 
                                       merge_list_from_mask, 
                                       merge_tensor_from_mask, 
                                       pad_1d_tensor_to_alignment,
                                       pad_2d_tensor_to_alignment)


KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(hidden_size,
                                                 2 * intermediate_size,
                                                 bias=False,
                                                 gather_output=False,
                                                 perform_initialization=False)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           input_is_parallel=True,
                                           perform_initialization=False)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
        )
        self.attn = PagedAttentionWithRoPE(self.num_heads,
                                           self.head_dim,
                                           self.scaling,
                                           rotary_dim=self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            perform_initialization=False)
        # self.layers = nn.ModuleList([
        #     LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        # ])
        self.num_backbone_layers = config.init_lora_layer
        self.backbone = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(self.num_backbone_layers)
        ])
        self.tail_l = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers - self.num_backbone_layers)
        ])  # length predictor
        self.tail_r = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers - self.num_backbone_layers)
        ])  # response generator
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("cache_hs", torch.zeros(2560, config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(self.num_backbone_layers):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.backbone[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        assert cache_event is None, "NotImplementedError: not support not None cache_event for now. Fix in the future."
        kwargs_l, kwargs_r = self.divide_backbone_outputs(
            hidden_states, positions, input_metadata
        )

        # length predictor part
        hs_l = kwargs_l.pop("hidden_states")
        if hs_l.shape[0] > 0: 
            for i in range(len(self.tail_l)):
                layer = self.tail_l[i]
                hs_l = layer(
                    hidden_states=hs_l,
                    kv_cache=kv_caches[self.num_backbone_layers+i],
                    cache_event=None,
                    **kwargs_l
                )

        # response generator part
        hs_r = kwargs_r.pop("hidden_states")
        if hs_r.shape[0] > 0: 
            for i in range(len(self.tail_r)):
                layer = self.tail_r[i]
                hs_r = layer(
                    hidden_states=hs_r,
                    kv_cache=kv_caches[self.num_backbone_layers+i],
                    cache_event=None,
                    **kwargs_r
                )

        hidden_states = self.merge_hidden_states(
            hs_l, hs_r, kwargs_l["input_metadata"], kwargs_r["input_metadata"], input_metadata.tok_len_pred_flag)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    @staticmethod
    def divide_backbone_outputs(
        hidden_states: torch.Tensor, 
        positions: torch.Tensor, 
        input_metadata: InputMetadata
    ) -> Tuple[dict, dict]:
        input_length_wo_pad = input_metadata.tok_len_pred_flag.size(0)
        hs_l, hs_r = divide_tensor_by_mask(hidden_states[:input_length_wo_pad], input_metadata.tok_len_pred_flag)
        pos_l, pos_r = divide_tensor_by_mask(positions[:input_length_wo_pad], input_metadata.tok_len_pred_flag)
        hs_l = pad_2d_tensor_to_alignment(hs_l, 8, 0)
        hs_r = pad_2d_tensor_to_alignment(hs_r, 8, 0)
        pos_l = pad_1d_tensor_to_alignment(pos_l, 8)
        pos_r = pad_1d_tensor_to_alignment(pos_r, 8)
        meta_l, meta_r = InputMetadata.divide_for_length_prediction_and_generation(input_metadata)
        kwargs_l = {
            "positions": pos_l,
            "hidden_states": hs_l,
            "input_metadata": meta_l,
        }
        kwargs_r = {
            "positions": pos_r,
            "hidden_states": hs_r,
            "input_metadata": meta_r,
        }
        return kwargs_l, kwargs_r

    @staticmethod
    def merge_hidden_states(
        hs_l: torch.Tensor,
        hs_r: torch.Tensor,
        mt_l: InputMetadata,
        mt_r: InputMetadata,
        ori_tok_flag: torch.Tensor,
    ) -> torch.Tensor:
        input_length_wo_pad_l = mt_l.tok_len_pred_flag.size(0)
        input_length_wo_pad_r = mt_r.tok_len_pred_flag.size(0)
        pos_tensor, neg_tensor = hs_l[:input_length_wo_pad_l], hs_r[:input_length_wo_pad_r]
        hs = merge_tensor_from_mask(pos_tensor, neg_tensor, ori_tok_flag)
        hs = pad_2d_tensor_to_alignment(hs, 8, 0)
        return hs

    # def merge_hidden_states(
    #     self,
    #     hs_l: torch.Tensor,
    #     hs_r: torch.Tensor,
    #     mt_l: InputMetadata,
    #     mt_r: InputMetadata,
    #     ori_tok_flag: torch.Tensor,
    # ) -> torch.Tensor:
    #     input_length_wo_pad_l = mt_l.tok_len_pred_flag.size(0)
    #     input_length_wo_pad_r = mt_r.tok_len_pred_flag.size(0)
    #     pos_tensor, neg_tensor = hs_l[:input_length_wo_pad_l], hs_r[:input_length_wo_pad_r]

    #     if pos_tensor.shape[0] == self.cache_hs.shape[0]: 
    #         return self.cache_hs 

    #     import pdb; pdb.set_trace() 
    #     if pos_tensor.shape[0] > 0: 
    #         self.cache_hs[ori_tok_flag==1, :] = pos_tensor

    #     if neg_tensor.shape[0] > 0: 
    #         self.cache_hs[ori_tok_flag==0, :] = neg_tensor
    #     aligned_length = (-ori_tok_flag.size(0)) % 8
    #     self.cache_hs[len(ori_tok_flag):aligned_length] = 0
    #     # hs = merge_tensor_from_mask(pos_tensor, neg_tensor, ori_tok_flag)
    #     # hs = pad_2d_tensor_to_alignment(hs, 8, 1)
    #     return self.cache_hs[:aligned_length]


class LlamoraForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            gather_output=False,
                                            perform_initialization=False)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = [
        "embed_tokens.weight", "lm_head.weight", "qkv_proj.weight",
        "gate_proj.weight", "up_proj.weight"
    ]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def load_weights(self, loaded_model):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        sd = self.state_dict()

        # step 1: loading weights of huggingface base model
        loaded_model.base_model.unmerge_adapter()
        sd_hf = loaded_model.base_model.model.state_dict()
        for key_hf, wei_hf in sd_hf.items():
            if any(k in key_hf for k in ("rotary_emb.inv_freq", "lora_A", "lora_B")):
                continue

            is_attention_weight = False
            # say n = num_backbone_layers, loading attn weights
            # for attn weights in layer 0~n-1: 
            #   [copy] hf llama -> backbone
            # for attn weights in layer n~31: 
            #   [copy] hf llama -> tail_r
            for stride_id, att_weight_name in enumerate(
                ["q_proj.weight", "k_proj.weight", "v_proj.weight"]):
                if att_weight_name not in key_hf:
                    continue
                layer_num = int(key_hf.split('.')[2])
                if layer_num < self.model.num_backbone_layers:
                    net_name = "backbone"
                    new_layer_num = layer_num
                else:
                    net_name = "tail_r"
                    new_layer_num = layer_num - self.model.num_backbone_layers
                key = f"model.{net_name}.{new_layer_num}.self_attn.qkv_proj.weight"
                param = sd[key]
                shard_size = param.shape[0] // 3
                wei_hf = wei_hf[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param[shard_size * stride_id:shard_size *
                                        (stride_id + 1)]
                assert param_slice.shape == wei_hf.shape
                with torch.no_grad():
                    param_slice.copy_(wei_hf)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in key_hf:
                    continue
                layer_num = int(key_hf.split('.')[2])
                if layer_num < self.model.num_backbone_layers:
                    net_names = ("backbone",)
                    new_layer_num = layer_num
                else:
                    net_names = ("tail_l", "tail_r")
                    new_layer_num = layer_num - self.model.num_backbone_layers
                for net_name in net_names:
                    key = f"model.{net_name}.{new_layer_num}.mlp.gate_up_proj.weight"
                    param = sd[key]
                    shard_size = param.shape[0] // 2
                    wei_hf = wei_hf[
                        shard_size * tensor_model_parallel_rank:shard_size *
                        (tensor_model_parallel_rank + 1)]
                    param_slice = param[shard_size * stride_id:shard_size *
                                            (stride_id + 1)]
                    assert param_slice.shape == wei_hf.shape
                    with torch.no_grad():
                        param_slice.copy_(wei_hf)
                    is_gate_up_weight = True
                    break
            if is_gate_up_weight:
                continue

            if "layers" in key_hf:
                layer_num = int(key_hf.split('.')[2])
                if layer_num < self.model.num_backbone_layers:
                    net_names = ("backbone",)
                    new_layer_num = layer_num
                else:
                    net_names = ("tail_l", "tail_r")
                    new_layer_num = layer_num - self.model.num_backbone_layers
                for net_name in net_names:
                    key = key_hf.replace(f"layers.{layer_num}", f"{net_name}.{new_layer_num}")
                    param = sd[key]
                    load_tensor_parallel_weights(param, wei_hf, key_hf,
                                                self._column_parallel_weights,
                                                self._row_parallel_weights,
                                                tensor_model_parallel_rank)
            else:
                key = key_hf
                param = sd[key]
                load_tensor_parallel_weights(param, wei_hf, key_hf,
                                            self._column_parallel_weights,
                                            self._row_parallel_weights,
                                            tensor_model_parallel_rank)

        # step 2: loading weights of lora model
        # for attn weight in layer n~31: 
        #   [add] merged lora model -> tail_l
        loaded_model.base_model.merge_adapter()
        sd_hf = loaded_model.base_model.model.state_dict()
        for key_hf, wei_hf in sd_hf.items():
            for stride_id, att_weight_name in enumerate(
                ["q_proj.weight", "k_proj.weight", "v_proj.weight"]):
                if att_weight_name not in key_hf:
                    continue
                layer_num = int(key_hf.split('.')[2])
                if layer_num < self.model.num_backbone_layers:
                    continue
                else:
                    net_name = "tail_l"
                    new_layer_num = layer_num - self.model.num_backbone_layers
                    key = f"model.{net_name}.{new_layer_num}.self_attn.qkv_proj.weight"
                    param = sd[key]
                    shard_size = param.shape[0] // 3
                    wei_hf = wei_hf[
                        shard_size * tensor_model_parallel_rank:shard_size *
                        (tensor_model_parallel_rank + 1)]
                    param_slice = param[shard_size * stride_id:shard_size *
                                            (stride_id + 1)]
                    assert param_slice.shape == wei_hf.shape
                    with torch.no_grad():
                        param_slice.copy_(wei_hf)
                    break