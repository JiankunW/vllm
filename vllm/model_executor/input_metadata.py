from __future__ import annotations
from typing import Dict, List, Tuple

import torch
from xformers.ops import AttentionBias

from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData
from vllm.model_executor.utils import divide_list_by_mask, divide_tensor_by_mask


class InputMetadata:
    """Metadata for input sequences. Used for PagedAttention.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        context_lens: the length of attention context for each generation token.
        max_context_len: The maximum context length.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        prompt_lens: List[int],
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        block_tables: torch.Tensor,
        grp_len_pred_flag: List[int],
        tok_len_pred_flag: torch.Tensor,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables
        self.grp_len_pred_flag = grp_len_pred_flag
        self.tok_len_pred_flag = tok_len_pred_flag

        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = sum(prompt_lens)
        self.num_generation_tokens = context_lens.shape[0]
        self.num_valid_tokens = slot_mapping.shape[0]
        if block_tables.numel() > 0:
            self.max_num_blocks_per_seq = block_tables.shape[1]
        else:
            self.max_num_blocks_per_seq = 0
        assert block_tables.shape[0] == self.num_generation_tokens
        assert context_lens.shape[0] == self.num_generation_tokens

        # Set during the execution of the first attention op.
        self.attn_bias: List[AttentionBias] = []

    def __repr__(self) -> str:
        # Print only useful metadata.
        return (f'InputMetadata('
                f'num_valid_tokens={self.num_valid_tokens}, '
                f'num_prompt_tokens={self.num_prompt_tokens}, '
                f'num_prompts={self.num_prompts}, '
                f'prompt_lens={self.prompt_lens}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'context_lens={self.context_lens}, '
                f'max_context_len={self.max_context_len}), '
                f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
                f'block_tables={self.block_tables}), '
                f'slot_mapping={self.slot_mapping}, '
                f'grp_len_pred_flag={self.grp_len_pred_flag}, ' 
                f'tok_len_pred_flag={self.tok_len_pred_flag}')

    @classmethod
    def divide_for_length_prediction_and_generation(cls, meta: InputMetadata) -> Tuple[InputMetadata, InputMetadata]:
        # divide the metadata into two new metadata for length prediction and response generation.
        assert (meta.grp_len_pred_flag and meta.tok_len_pred_flag.numel()!=0), \
            "Missing length prediction masks, the division is rejected."

        length_seq_groups, response_seq_groups = divide_list_by_mask(meta.seq_groups, meta.grp_len_pred_flag)
        length_seq_data = {seq_id: meta.seq_data[seq_id] for seq_ids, _ in length_seq_groups for seq_id in seq_ids}
        response_seq_data = {seq_id: meta.seq_data[seq_id] for seq_ids, _ in response_seq_groups for seq_id in seq_ids}
        length_prompt_lens, response_prompt_lens = divide_list_by_mask(meta.prompt_lens, meta.grp_len_pred_flag)
        length_slot_mapping, response_slot_mapping = divide_tensor_by_mask(meta.slot_mapping, meta.tok_len_pred_flag)
        prompt_tok_sum = sum(meta.prompt_lens)
        length_context_lens, response_context_lens = divide_tensor_by_mask(meta.context_lens, 
                                                                           meta.tok_len_pred_flag[prompt_tok_sum:])
        length_max_context_len = length_context_lens.max().item() if length_context_lens.numel()!=0 else 0
        response_max_context_len = response_context_lens.max().item() if response_context_lens.numel()!=0 else 0
        length_block_tables, response_block_tables = divide_tensor_by_mask(meta.block_tables, 
                                                                           meta.tok_len_pred_flag[prompt_tok_sum:])
        length_grp_len_pred_flag, response_grp_len_pred_flag = divide_list_by_mask(meta.grp_len_pred_flag, 
                                                                                   meta.grp_len_pred_flag)
        length_tok_len_pred_flag, response_tok_len_pred_flag = divide_tensor_by_mask(meta.tok_len_pred_flag, 
                                                                                   meta.tok_len_pred_flag)

        length_meta = cls(
            length_seq_groups,
            length_seq_data,
            length_prompt_lens,
            length_slot_mapping,
            length_context_lens,
            length_max_context_len,
            length_block_tables,
            length_grp_len_pred_flag,
            length_tok_len_pred_flag,
        )

        response_meta = cls(
            response_seq_groups,
            response_seq_data,
            response_prompt_lens,
            response_slot_mapping,
            response_context_lens,
            response_max_context_len,
            response_block_tables,
            response_grp_len_pred_flag,
            response_tok_len_pred_flag,
        )

        return length_meta, response_meta

    @classmethod
    def merge_for_length_prediction_and_generation(cls, mt_l: InputMetadata, mt_r: InputMetadata, ori_grp_flag: List[int], ori_tok_flag: torch.Tensor) -> InputMetadata:
        # merge the metadata for (l)ength prediction and (r)esponse generation into a single metadata.
        pass