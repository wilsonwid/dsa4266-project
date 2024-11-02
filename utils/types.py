from enum import Enum

class FlipCode(Enum):
    FLIP = 1
    NO_FLIP = 0

class NonlinearityEnum(Enum):
    SILU = "silu"
    RELU = "relu"
    GELU = "gelu"
    ELU = "elu"
    LRELU = "lrelu"
