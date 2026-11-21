# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from typing import Optional
import numbers
try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
except ImportError:
    LigerRMSNormFunction = None

__all__ = ["LayerNorm2d", "RMSNorm", "build_norm", "get_norm_name", "reset_bn", "remove_bn", "set_norm_eps"]

class LayerNorm2d(nn.LayerNorm):
    rmsnorm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x if LayerNorm2d.rmsnorm else x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, rmsnorm={self.rmsnorm}"


class RMSNorm(nn.Module):
    # Code from PyTorch
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
        \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}

    The RMS is taken over the last ``D`` dimensions, where ``D``
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the RMS is computed over
    the last 2 dimensions of the input.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: ``torch.finfo(x.dtype).eps``
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> rms_norm = nn.RMSNorm([2, 3])
        >>> input = torch.randn(2, 2, 3)
        >>> rms_norm(input)

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward pass.
        """
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Extra information about the module.
        """
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )

class LigerRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=True,
        row_mode=None,
        elementwise_affine=True,
    ):
        """
            Code adopted and modified from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/rms_norm.py
        """
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape) if init_fn == "ones" else torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place, self.row_mode = (
            eps,
            offset,
            casting_mode,
            in_place,
            row_mode,
        )
        assert LigerRMSNormFunction is not None, "LigerRMSNormFunction is not available"

    def forward(self, hidden_states):  
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
            self.row_mode,
        )
        
    def extra_repr(self):
        return f"weight_shape={tuple(self.weight.shape) if self.weight is not None else None}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}, row_mode={self.row_mode}"



# register normalization function here
#   name: module, kwargs with default values
REGISTERED_NORMALIZATION_DICT: dict[str, tuple[type, dict[str, any]]] = {
    "bn2d": (nn.BatchNorm2d, {"num_features": None, "eps": 1e-5, "momentum": 0.1, "affine": True}),
    "syncbn": (nn.SyncBatchNorm, {"num_features": None, "eps": 1e-5, "momentum": 0.1, "affine": True}),
    "ln": (nn.LayerNorm, {"normalized_shape": None, "eps": 1e-5, "elementwise_affine": True}),
    "ln2d": (LayerNorm2d, {"normalized_shape": None, "eps": 1e-5, "elementwise_affine": True}),
    "rms": (RMSNorm, {"normalized_shape": None, "eps": 1e-5, "elementwise_affine": True}),
    "liger-rms": (LigerRMSNorm, {"normalized_shape": None, "eps": 1e-5, "elementwise_affine": True}),
    "identity": (nn.Identity, {})
}


def build_norm(name="bn2d", num_features=None, affine=True, **kwargs) -> nn.Module or None:
    if name in ["ln", "ln2d", "rms", "liger-rms"]:
        kwargs["normalized_shape"] = num_features
        kwargs["elementwise_affine"] = affine
    else:
        kwargs["num_features"] = num_features
        kwargs["affine"] = affine
    if name in REGISTERED_NORMALIZATION_DICT:
        norm_cls, default_args = copy.deepcopy(REGISTERED_NORMALIZATION_DICT[name])
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return norm_cls(**default_args)
    elif name is None or name.lower() == "none":
        return None
    else:
        raise ValueError("do not support: %s" % name)


def get_norm_name(norm: nn.Module or None) -> str or None:
    if norm is None:
        return None
    module2name = {}
    for key, config in REGISTERED_NORMALIZATION_DICT.items():
        module2name[config[0].__name__] = key
    return module2name.get(type(norm).__name__, "unknown")


def reset_bn(
    model: nn.Module,
    data_loader: list,
    sync=True,
    progress_bar=False,
) -> None:
    import copy

    import torch.nn.functional as F
    from packages.apps.utils import AverageMeter, is_master, sync_tensor
    from packages.models.utils import get_device, list_join
    from tqdm import tqdm

    bn_mean = {}
    bn_var = {}

    tmp_model = copy.deepcopy(model)
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_mean[name] = AverageMeter(is_distributed=False)
            bn_var[name] = AverageMeter(is_distributed=False)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    x = x.contiguous()
                    if sync:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_mean = sync_tensor(batch_mean, reduce="cat")
                        batch_mean = torch.mean(batch_mean, dim=0, keepdim=True)

                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                        batch_var = sync_tensor(batch_var, reduce="cat")
                        batch_var = torch.mean(batch_var, dim=0, keepdim=True)
                    else:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.shape[0]
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # skip if there is no batch normalization layers in the network
    if len(bn_mean) == 0:
        return

    tmp_model.eval()
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc="reset bn", disable=not progress_bar or not is_master()) as t:
            for images in data_loader:
                images = images.to(get_device(tmp_model))
                tmp_model(images)
                t.set_postfix(
                    {
                        "bs": images.size(0),
                        "res": list_join(images.shape[-2:], "x"),
                    }
                )
                t.update()

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, _BatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def remove_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.weight = m.bias = None
            m.forward = lambda x: x


def set_norm_eps(model: nn.Module, eps: float or None = None, momentum: float or None = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps
            if momentum is not None:
                m.momentum = momentum



