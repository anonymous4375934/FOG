import torch
from torch.nn import functional as F

from megatron.core.jit import jit_fuser


@jit_fuser
def scaled_swiglu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_1, y_2 = torch.chunk(x, 2, dim=-1)
    s = y_1.detach().abs().max(dim=-1, keepdim=True)[0]
    tmp = y_2/s
    return F.silu(y_1)*tmp, s
