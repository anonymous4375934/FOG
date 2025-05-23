from typing import Optional
import os
import torch.nn.functional as F
import torch
from torch import nn


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size() == 1:
        return input_
    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous())   # CHECK IF THE GROUP IS SET CORRECTLY

    return input_


class _LayerScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, parallel_input: bool) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.parallel_input = parallel_input
        return weight*x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight = ctx.saved_tensors
        if not ctx.parallel_input:
            return grad_output*weight, grad_output*x, None

        x_parallel = x
        grad_output_parallel = grad_output
        grad_x_parallel = grad_output_parallel*weight

        grad_weight_parallel = grad_output_parallel*weight
        grad_weight_parallel_reduced = torch.sum(grad_weight_parallel.view(-1, weight.size(-1)), dim=0)
        grad_weight = _reduce(grad_weight_parallel_reduced)

        return grad_x_parallel, grad_weight, None


class LayerScale(nn.Module):
    def __init__(self, hidden_size: Optional[int] = None, initial_value: float = 1.0, device=None, dtype=None,
                 sequence_parallel: bool = False):
        super().__init__()
        if hidden_size is None:
            self.weight = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        else:
            self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.sequence_parallel = sequence_parallel
        self.initial_value = initial_value
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        if x_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        y = _LayerScale.apply(x, self.weight, self.sequence_parallel)
        if x_dtype != self.weight.dtype:
            return y.to(x_dtype)
        return y
    
class ScaledSwiglu(nn.Module):
    '''
    References (adapted from deepspeed to transformers):
     - https://arxiv.org/abs/2409.12517 
     - https://github.com/Anonymous1252022/Megatron-DeepSpeed/blob/20029cf54da36558cf406843764a7998ff6e3410/experimental/fp8_linear.py#L220
    '''
    def __init__(self, delayed=False):
        super(ScaledSwiglu, self).__init__()
        self.delayed = delayed
        self.register_buffer('scale', torch.tensor(1.0))
    
    def forward(self, gate_proj, up_proj):
        if self.delayed:
            max_abs = up_proj.detach().abs().amax(dim=-1, keepdim=True)
            self.scale = max(self.scale, max_abs)
            scale = self.scale
        else:
            if os.getenv('DETACH_SCALED_SWIGLU', 'false').lower() == 'true':  # doesn't matter as we only use eval mode
                scale = up_proj.detach().abs().amax(dim=-1, keepdim=True)
            else:
                scale = up_proj.abs().amax(dim=-1, keepdim=True)
        
        scaled_up_proj = up_proj / scale.clamp(min=1e-12)
        return F.silu(gate_proj) * scaled_up_proj, scale


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x