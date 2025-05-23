# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

import torch

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version


class WrappedTorchNorm:
    """
    A conditional wrapper to initialize an instance of PyTorch's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        # TODO: unused arguments.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
        init_value: Optional[float] = None,
        learnable: bool = True,
    ):
        assert (
            not config.layernorm_zero_centered_gamma
        ), f"zero_centered_gamma not supported by torch LayerNorm"

        assert not config.persist_layer_norm, f"persist_layer_norm not supported by torch LayerNorm"

        assert not learnable or not config.sequence_parallel, f"sequence parallel not supported by torch LayerNorm"

        assert init_value is None or init_value == 1.0, "init_value is not supported by torch LayerNorm"

        assert (
            not config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        extra_kw = {}
        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
            assert learnable
        elif config.normalization == "RMSNorm":
            assert is_torch_min_version(
                "2.4.0a0"
            ), 'Torch RMSNorm requires PyTorch version >= 2.4.0'

            norm_cls = torch.nn.RMSNorm
            if not learnable:
                print("Non-learnable torch :)")
                extra_kw["elementwise_affine"] = False
        else:
            raise Exception("Only LayerNorm and RMSNorm are currently supported")

        return norm_cls(normalized_shape=hidden_size, eps=eps, **extra_kw)
