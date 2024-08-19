# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
# transformers.models.phi.modeling_phi

from importlib import import_module

import torch.nn as nn
from torch import Tensor
from transformers.models.phi.configuration_phi import PhiConfig

from modules.phi_mlp import PhiMLP


class Block(nn.Module):
    def __init__(self, d_model, config, factory_kwargs, layer_idx, **kwargs):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.residual_in_fp32 = config.block_input.residual_in_fp32
        self.layer_idx = layer_idx

        # Mixer
        MixerClass = import_module(f"modules.mixers.{config.CoreType}").Mixer
        self.mixer = MixerClass(
            d_model=self.d_model,
            layer_idx=layer_idx,
            **kwargs,
            **config.core_input,
            **factory_kwargs,
        )

        # MLP + LayerNorm + Dropout
        self.mlp = PhiMLP(
            PhiConfig(
                hidden_size=self.d_model,
                intermediate_size=self.d_model * 4,
                hidden_act="gelu_new",
            )
        )
        self.input_layernorm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.resid_dropout = nn.Dropout(config.block_input.resid_dropout)

        return

    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        run_mlp_component=True,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        position_ids=None,
        **kwargs,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        outputs = {}

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer
        mamba_outputs = self.mixer(
            hidden_states,
            return_mixer_matrix=return_mixer_matrix,
            inference_params=inference_params,
            position_ids=position_ids,
        )
        mamba_outputs["hidden_states"] = mamba_outputs["hidden_states"].to(
            residual.dtype
        )

        if not run_mlp_component:
            return mamba_outputs

            # store outputs
        if return_mamba_outputs:
            outputs["mamba_hidden_states"] = mamba_outputs["hidden_states"]
        if return_mixer_matrix:
            outputs["transfer_matrix"] = mamba_outputs["transfer_matrix"]

        # Feed Forward
        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states)).to(
            residual.dtype
        )

        # Mixer output
        mixer_output = self.resid_dropout(mamba_outputs["hidden_states"])

        # sum all up (this is not sequential)
        outputs["hidden_states"] = mixer_output + feed_forward_hidden_states + residual

        return outputs

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if getattr(self.mixer, "allocate_inference_cache", None) is None:
            return
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
