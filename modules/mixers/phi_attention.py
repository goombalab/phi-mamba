# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
# transformers.models.phi.modeling_phi

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from modules.modeling_phi import PHI_ATTENTION_CLASSES, PhiConfig


class Mixer(nn.Module):
    def __init__(self, layer_idx, **kwargs):
        super().__init__()
        self.model_cfg = kwargs
        self.layer_idx = layer_idx

        # Select the attention class based on the configuration
        attn_type = kwargs.get("attn_type", "eager")
        self.self_attn = PHI_ATTENTION_CLASSES[attn_type](
            PhiConfig(self.model_cfg), layer_idx
        )
        self._attention_mask = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        return_mixer_matrix=False,
        inference_params=None,
        position_ids=None,
        **kwargs
    ):

        if attention_mask is None:
            attention_mask = self.create_mask(
                hidden_states,
                offset=0
                if inference_params is None
                else inference_params.seqlen_offset,
            )

        if inference_params is not None:
            self._check_states_from_cache(inference_params=inference_params)

        attn_output, attn_weights, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=return_mixer_matrix,
            past_key_value=inference_params,
            position_ids=position_ids.to(torch.long) if position_ids is not None else None,
        )

        return {"hidden_states": attn_output, "transfer_matrix": attn_weights}

    def create_mask(self, hidden_states, offset=0):
        batch_size, seq_length = hidden_states.size()[:2]
        # Is the attention mask already prepared?
        if self._attention_mask is not None and self._attention_mask.size() == (
            batch_size,
            1,
            seq_length,
            seq_length + offset,
        ):
            return self._attention_mask
        elif self._attention_mask is not None and self._attention_mask.size() != (
            batch_size,
            1,
            seq_length,
            seq_length + offset,
        ):
            self._attention_mask = None
        # Prepare the attention mask
        if self.self_attn.__class__ == PHI_ATTENTION_CLASSES["eager"]:
            # 2d mask is passed through the layers
            self._attention_mask = _prepare_4d_causal_attention_mask(
                self._attention_mask, (batch_size, seq_length), hidden_states, offset
            )
        else:
            # 4d mask is passed through the layers
            if self._attention_mask is not None and 0.0 in self._attention_mask:
                return self._attention_mask
            else:
                None
        return self._attention_mask

    def _check_states_from_cache(self, inference_params):
        """
        conv_state: (batch, d_conv, conv1d.weight.shape[0])
        ssm_state: (batch, nheads, headdim, d_state)
        """
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[
                self.layer_idx
            ] = self.allocate_inference_cache(
                inference_params.max_batch_size, inference_params.max_seqlen
            )

        if inference_params.seqlen_offset == 0:
            inference_params.key_value_memory_dict[self.layer_idx]["key_cache"].zero_()
            inference_params.key_value_memory_dict[self.layer_idx][
                "value_cache"
            ].zero_()
        assert self.layer_idx in inference_params.key_value_memory_dict

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        target_dtype = torch.float32
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
        return {
            "key_cache": torch.zeros(
                batch_size,
                self.self_attn.num_heads,
                max_seqlen,
                self.self_attn.head_dim,
                dtype=target_dtype if dtype is None else dtype,
                device=self.self_attn.q_proj.weight.device,
            ),
            "value_cache": torch.zeros(
                batch_size,
                self.self_attn.num_heads,
                max_seqlen,
                self.self_attn.head_dim,
                dtype=target_dtype if dtype is None else dtype,
                device=self.self_attn.q_proj.weight.device,
            ),
        }
