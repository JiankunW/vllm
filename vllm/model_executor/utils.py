"""Utils for model executor."""
import random
from typing import Optional, Tuple, List

import numpy as np
import torch

from vllm.model_executor.parallel_utils.parallel_state import model_parallel_is_initialized
from vllm.model_executor.parallel_utils.tensor_parallel import model_parallel_cuda_manual_seed


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_parallel_is_initialized():
        model_parallel_cuda_manual_seed(seed)

def divide_list_by_mask(input_list: list, mask_list: List[int]) -> Tuple[list, list]:
    pos_list = [x for x, mask in zip(input_list, mask_list) if mask]
    neg_list = [x for x, mask in zip(input_list, mask_list) if not mask]
    return pos_list, neg_list

def divide_tensor_by_mask(input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_tensor = input_tensor[mask_tensor==1]
    neg_tensor = input_tensor[mask_tensor==0]
    return pos_tensor, neg_tensor

def merge_list_from_mask(pos_list: list, neg_list: list, mask_list: List[int]) -> list:
    assert len(pos_list) + len(neg_list) == len(mask_list)
    combined_list = []

    for mask in mask_list:
        if mask:
            combined_list.append(pos_list.pop(0))
        else:
            combined_list.append(neg_list.pop(0))

    return combined_list

def merge_tensor_from_mask(pos_tensor: torch.Tensor, neg_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
    shape = (mask_tensor.size(0), pos_tensor.size(1))
    combined_tensor = torch.zeros(shape, dtype=pos_tensor.dtype, device=pos_tensor.device)

    combined_tensor[mask_tensor==1] = pos_tensor
    combined_tensor[mask_tensor==0] = neg_tensor

    return combined_tensor

def pad_1d_tensor_to_alignment(x: torch.Tensor, multiple_of: int) -> torch.Tensor:
    assert x.ndim==1
    pad = torch.zeros(((-x.size(0)) % multiple_of), dtype=x.dtype, device=x.device)
    return torch.cat((x, pad))

def pad_2d_tensor_to_alignment(x: torch.Tensor, multiple_of: int, dim: int) -> torch.Tensor:
    assert x.ndim==2
    if dim==1 or dim==-1:
        pad_shape = (x.size(0), (-x.size(1)) % multiple_of)
    elif dim==0:
        pad_shape = ((-x.size(0)) % multiple_of, x.size(1))
    else:
        raise IndexError(f"Dimension out of range (expected to be in range of [-1, 1], but got {dim})")
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, pad), dim=dim)