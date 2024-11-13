import torch.nn as nn

from typing import Optional
from utils.types import NonlinearityEnum
from math import ceil

SEED = 42
INPUT_SHAPE = (224, 224)

def select_nonlinearity(nl_enum: NonlinearityEnum) -> Optional[nn.Module]:
    match nl_enum:
        case NonlinearityEnum.SILU: return nn.SiLU()
        case NonlinearityEnum.RELU: return nn.ReLU()
        case NonlinearityEnum.GELU: return nn.GELU()
        case NonlinearityEnum.ELU: return nn.ELU()
        case NonlinearityEnum.LRELU: return nn.LeakyReLU()
        case _: return None

def convert_str_to_nonlinearity(s: str) -> Optional[NonlinearityEnum]:
    match s:
        case "silu": return NonlinearityEnum.SILU
        case "relu": return NonlinearityEnum.RELU
        case "gelu": return NonlinearityEnum.GELU
        case "elu": return NonlinearityEnum.ELU
        case "lrelu": return NonlinearityEnum.LRELU
        case _: return None

def compute_linear_size(input_shape: int, kernel_size: int, stride: int, padding: int) -> int:
    return (input_shape + 2 * padding - kernel_size) // stride + 1

def compute_linear_size_n(input_shape: int, kernel_size: int, stride: int, padding: int, n: int) -> int:
    cur_input_shape = input_shape
    for _ in range(n):
        cur_input_shape = compute_linear_size(cur_input_shape, kernel_size, stride, padding)
    return cur_input_shape

def get_padding(kernel_size):
    return ceil((223*stride - 224 + kernel_size)/2)

def get_output_size(stride, padding, kernel_size):
    return (224 + 2 * padding - kernel_size)//stride + 1