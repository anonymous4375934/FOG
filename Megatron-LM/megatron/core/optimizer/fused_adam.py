import torch
from transformer_engine.pytorch.optimizers import FusedAdam
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor


class Adam(FusedAdam):
    """Extends TE FusedAdam to have more fine-grained scaling behaviour."""

    def __init__(self, *args, exp_avg_sq_dtype=torch.float32, **kwargs):
        super().__init__(*args, exp_avg_sq_dtype=exp_avg_sq_dtype, **kwargs)

        self.e5m2 = set()
        if exp_avg_sq_dtype == torch.uint8:
            self.e5m2.add("exp_avg_sq")
        self.dtype_to_range_map["e5m2"] = torch.full([1], 57344.0, dtype=torch.float32)
        self.quant_eps = 1e-12  # Used in case of any 0 scaling.

    def _apply_scale(self, state_name, unscaled_state, scaled_state, scale):
        """Apply scaling on `unscaled_state`. `scaled_state` and `scale` will be written inplace.

        Arguments:
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            unscaled_state (torch.Tensor): An unscaled high-precision tensor.
            scaled_state (torch.Tensor): An scaled low-precision tensor.
            scale (torch.Tensor): A FP32 tensor representing the scaling factor.
        """
        assert unscaled_state.dtype == torch.float32
        if scaled_state.dtype == torch.bfloat16:
            scaled_state.copy_(unscaled_state.bfloat16())
            return

        dtype = self.name_to_dtype_map[state_name]
        if dtype == torch.uint8:
            assert isinstance(scaled_state, Float8Tensor)
            assert len(scaled_state._scale_inv) == 1, (
                "Only scaling with one scaling factor                per tensor is supported by the"
                " FusedAdam."
            )
        else:
            assert scaled_state.dtype == dtype

        # Changes to compute the max_range here!
        true_dtype = "e5m2" if dtype == torch.uint8 and state_name in self.e5m2 else dtype
        max_range = self.dtype_to_range_map[true_dtype]
        if max_range.device != scaled_state.device:
            max_range = max_range.to(scaled_state.device)
            self.dtype_to_range_map[true_dtype] = max_range

        if unscaled_state.device != scaled_state.device:
            unscaled_state = unscaled_state.to(scaled_state.device)
        min_val, max_val = torch.aminmax(unscaled_state)
        absmax = torch.maximum(-min_val, max_val)
        absmax = absmax.to(dtype=torch.float32, device=unscaled_state.device)
        torch.div(absmax, max_range, out=scale)
        if isinstance(scaled_state, Float8Tensor):
            if torch.any(scale < self.quant_eps):
                scale += self.quant_eps
            scaled_state._scale_inv.copy_(scale)
            scaled_state.copy_(unscaled_state)
        else:
            if torch.any(scale < self.quant_eps):
                scale += self.quant_eps
            rscale = torch.where(scale > 0, scale.reciprocal(), 0.0)
            unscaled_state.mul_(rscale)
            scaled_state.copy_(unscaled_state)

    def _initialize_state(
        self, param, state_name, zero_buffer: bool, store_param_remainders: bool = False
    ):
        """Initialize one of the optimizer states according to `state_name`.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            zero_buffer (bool): Whether to initialize the optimizer state with zeros.
            store_param_remainders (bool): Store only trailing remainder bits.
        """
        dtype = self.name_to_dtype_map[state_name]
        data = torch.empty_like(param, dtype=dtype)
        if zero_buffer:
            data.zero_()

        if dtype == torch.uint8:
            fp8_dtype = tex.DType.kFloat8E5M2 if state_name in self.e5m2 else tex.DType.kFloat8E4M3
            print(f"fp8 dtype for {state_name} is {fp8_dtype}")
            self.state[param][state_name] = Float8Tensor(
                data=data,
                dtype=torch.float32,
                fp8_scale_inv=torch.ones([1], dtype=torch.float32, device=param.device),
                fp8_dtype=fp8_dtype,
            )
        else:
            self.state[param][state_name] = data

        # Create scale if necessary.
        if dtype != torch.float32:
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param][state_name] = torch.ones(
                [1], dtype=torch.float32, device=param.device
            )
