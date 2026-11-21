from copy import deepcopy

import torch
import torch.nn as nn

from diffusion.model.nets.causal.causal_emage import (
    CausalEmage,
    DEFAULT_FUSE_INPUT_CONFIG,
    DEFAULT_INPUT_CONFIG,
)
from diffusion.utils.config import instantiate_from_config


class CausalEmageCleanDeep(CausalEmage):
    """
    P1-1 ablation model:
    Deepen clean-motion KV encoding with optional clean x-branch and clean fuse branch.
    """

    def __init__(
        self,
        *args,
        inputlayer_config=DEFAULT_INPUT_CONFIG,
        fuse_inputlayer_config=DEFAULT_FUSE_INPUT_CONFIG,
        clean_kv_use_xbranch: bool = True,
        clean_kv_use_fuse: bool = True,
        clean_kv_use_context: bool = True,
        clean_kv_detach_context: bool = False,
        clean_kv_use_real_expression: bool = True,
        **kwargs,
    ):
        # Preserve independent configs for clean branch before parent mutates configs in-place.
        clean_input_cfg = deepcopy(inputlayer_config)
        clean_fuse_cfg = deepcopy(fuse_inputlayer_config)

        super().__init__(
            *args,
            inputlayer_config=inputlayer_config,
            fuse_inputlayer_config=fuse_inputlayer_config,
            **kwargs,
        )

        self.clean_kv_use_xbranch = clean_kv_use_xbranch
        self.clean_kv_use_fuse = clean_kv_use_fuse
        self.clean_kv_use_context = clean_kv_use_context
        self.clean_kv_detach_context = clean_kv_detach_context
        self.clean_kv_use_real_expression = clean_kv_use_real_expression

        self._clean_kv_context_cache = None
        self._clean_kv_expression_cache = None

        if self.clean_kv_use_xbranch:
            clean_input_cfg["params"]["hidden_size"] = self.hidden_size
            clean_input_cfg["params"]["in_channels"] = (
                self.hand_in_channels
                + self.body_in_channels
                + self.expression_in_channels
                + self.trans_in_channels
            )

            self.clean_xhand_in = instantiate_from_config(clean_input_cfg)
            self.clean_xbody_in = instantiate_from_config(clean_input_cfg)
            self._initialize_extra_module(self.clean_xhand_in)
            self._initialize_extra_module(self.clean_xbody_in)
        else:
            self.clean_xhand_in = None
            self.clean_xbody_in = None

        if self.clean_kv_use_fuse:
            clean_fuse_cfg["params"]["hidden_size"] = self.hidden_size
            clean_fuse_cfg["params"]["in_channels"] = self.hidden_size * 3
            clean_fuse_cfg["params"]["enable_t_modulation"] = False
            if "mapping_size" not in clean_fuse_cfg["params"]:
                clean_fuse_cfg["params"]["mapping_size"] = self.mapping_size

            self.clean_hand_fuse = instantiate_from_config(clean_fuse_cfg)
            self.clean_body_fuse = instantiate_from_config(clean_fuse_cfg)
            self._initialize_extra_module(self.clean_hand_fuse)
            self._initialize_extra_module(self.clean_body_fuse)
        else:
            self.clean_hand_fuse = None
            self.clean_body_fuse = None

    @staticmethod
    def _initialize_extra_module(module: nn.Module):
        if hasattr(module, "initialize_weights"):
            module.initialize_weights()
            return

        for sub_module in module.modules():
            if isinstance(sub_module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(sub_module.weight)
                if sub_module.bias is not None:
                    nn.init.constant_(sub_module.bias, 0)

    def _cache_clean_context(self, args, kwargs):
        context = kwargs.get("context", None)
        if context is None and len(args) >= 3:
            context = args[2]
        self._clean_kv_context_cache = context

        expression = kwargs.get("expression", None)
        if expression is None and len(args) >= 4:
            expression = args[3]
        self._clean_kv_expression_cache = expression

    def _get_clean_expression_tokens(self, clean_motion: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = clean_motion.shape
        if (not self.clean_kv_use_real_expression) or (self._clean_kv_expression_cache is None):
            return clean_motion.new_zeros((bsz, seq_len, self.expression_in_channels))

        clean_expression = self._clean_kv_expression_cache.to(device=clean_motion.device, dtype=clean_motion.dtype)
        if clean_expression.shape[1] == 1 and seq_len > 1:
            clean_expression = clean_expression.expand(-1, seq_len, -1)
        elif clean_expression.shape[1] != seq_len:
            raise ValueError(
                f"clean expression length mismatch: expression={clean_expression.shape[1]}, clean_motion={seq_len}"
            )
        return clean_expression

    def _get_clean_context_tokens(self, reference_tokens: torch.Tensor) -> torch.Tensor:
        if (not self.clean_kv_use_context) or (self._clean_kv_context_cache is None):
            return torch.zeros_like(reference_tokens)

        context_tokens = self.context_embedder(self._clean_kv_context_cache)
        context_tokens = context_tokens.to(device=reference_tokens.device, dtype=reference_tokens.dtype)
        if self.clean_kv_detach_context:
            context_tokens = context_tokens.detach()

        if context_tokens.shape[1] == 1 and reference_tokens.shape[1] > 1:
            context_tokens = context_tokens.expand(-1, reference_tokens.shape[1], -1)
        elif context_tokens.shape[1] != reference_tokens.shape[1]:
            raise ValueError(
                f"clean context length mismatch: context={context_tokens.shape[1]}, "
                f"reference={reference_tokens.shape[1]}"
            )

        return context_tokens

    def forward_train(self, *args, **kwargs):
        self._cache_clean_context(args, kwargs)
        try:
            return super().forward_train(*args, **kwargs)
        finally:
            self._clean_kv_context_cache = None
            self._clean_kv_expression_cache = None

    def forward_inference(self, *args, **kwargs):
        self._cache_clean_context(args, kwargs)
        try:
            return super().forward_inference(*args, **kwargs)
        finally:
            self._clean_kv_context_cache = None
            self._clean_kv_expression_cache = None

    def _encode_clean_tokens_for_kv(
        self,
        clean_motion: torch.Tensor,
        hand_indices,
        body_indices,
    ):
        clean_motion = clean_motion.to(self.dtype)

        clean_hand = self.clean_hand_in(clean_motion[..., hand_indices])
        clean_body = self.clean_body_in(clean_motion[..., body_indices])

        clean_xhand = torch.zeros_like(clean_hand)
        clean_xbody = torch.zeros_like(clean_body)

        if self.clean_kv_use_xbranch:
            clean_expression = self._get_clean_expression_tokens(clean_motion)
            clean_x = torch.cat([clean_motion, clean_expression], dim=-1)
            clean_xhand = self.clean_xhand_in(clean_x)
            clean_xbody = self.clean_xbody_in(clean_x)

        if self.clean_kv_use_fuse:
            clean_context = self._get_clean_context_tokens(clean_hand)
            clean_hand = self.clean_hand_fuse(torch.cat([clean_hand, clean_xhand, clean_context], dim=-1))
            clean_body = self.clean_body_fuse(torch.cat([clean_body, clean_xbody, clean_context], dim=-1))
        elif self.clean_kv_use_xbranch:
            clean_hand = clean_hand + clean_xhand
            clean_body = clean_body + clean_xbody

        return clean_hand, clean_body

