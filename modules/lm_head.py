# Copyright (c) 2024, Kevin Li, Aviv Bick.

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from mamba_ssm.utils.generation import GenerationMixin
from transformers.utils import ModelOutput

from modules.backbone import MixerModel
from utils.config import Config


@dataclass
class CustomMambaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_transfer_matrices: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_mamba_outputs: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class LMHeadModel(nn.Module, GenerationMixin, PyTorchModelHubMixin):
    def __init__(
        self, config: dict, initializer_cfg=None, device=None, dtype=None, **kwargs
    ) -> None:

        super().__init__()

        # Load config
        if not isinstance(config, Config):
            config = Config.from_dict(config)
        self.config = config

        # Factory kwargs
        factory_kwargs = {"device": device, "dtype": dtype}

        # Pad vocab size to be a multiple of pad_vocab_size_multiple
        vocab_size = config.LanguageModel.input.vocab_size
        pad_vocab_size_multiple = config.LanguageModel.input.pad_vocab_size_multiple
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.config.LanguageModel.input.vocab_size = vocab_size

        # Mixer model
        self.backbone = MixerModel(
            input_size=vocab_size,
            config=self.config,
            initializer_cfg=initializer_cfg,
            **factory_kwargs,
            **kwargs
        )

        # LM head
        d_model = self.config.MixerModel.input.d_model
        vocab_size = self.config.LanguageModel.input.vocab_size
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=True, **factory_kwargs
        )  # changed for Phi

        return

    def allocate_inference_cache(self, *args, **kwargs):
        return self.backbone.allocate_inference_cache(*args, **kwargs)

    def forward(
        self,
        input_ids,
        position_ids=None,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        return_hidden_states=False,
        return_logits=True,
        inference_params=None,
        num_last_tokens=0,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        outputs = self.backbone(
            input_ids,
            return_mixer_matrix=return_mixer_matrix,
            return_mamba_outputs=return_mamba_outputs,
            return_hidden_states=return_hidden_states,
            inference_params=inference_params,
            position_ids=position_ids,
        )

        if outputs["last_hidden_state"] is not None and return_logits:
            logits = self.lm_head(outputs["last_hidden_state"]).float()
            outputs["logits"] = (
                logits if num_last_tokens == 0 else logits[:, -num_last_tokens:]
            )
        else:
            outputs["logits"] = None

        return CustomMambaCausalLMOutput(
            loss=None,
            logits=outputs["logits"],
            all_hidden_states=outputs["all_hidden_states"],
            all_transfer_matrices=outputs["all_transfer_matrices"],
            all_mamba_outputs=outputs["all_mamba_outputs"],
            last_hidden_state=outputs["last_hidden_state"],
        )

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f)
