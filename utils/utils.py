import torch.nn as nn

from typing import Optional
from utils.types import NonlinearityEnum

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