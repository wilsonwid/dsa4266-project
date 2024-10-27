import torch.nn as nn

from typing import Optional
from utils.types import NonlinearityEnum

def select_nonlinearity(nl_enum: NonlinearityEnum) -> Optional[nn.Module]:
    match nl_enum:
        case NonlinearityEnum.SILU: return nn.SiLU
        case NonlinearityEnum.RELU: return nn.ReLU
        case NonlinearityEnum.GELU: return nn.GELU
        case NonlinearityEnum.ELU: return nn.ELU
        case NonlinearityEnum.LRELU: return nn.LeakyReLU
        case _: return None
