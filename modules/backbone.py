from importlib import import_module

import torch.nn as nn


class MixerModel(nn.Module):
    def __init__(
        self, input_size, config=None, device=None, dtype=None, **kwargs
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        n_layer = self.config.MixerModel.input.n_layer
        d_model = self.config.MixerModel.input.d_model

        self.embedding = nn.Embedding(input_size, d_model, **factory_kwargs)

        blocks = [
            self.config.__dict__[name]
            for name in self.config.__dict__.keys()
            if name.startswith("Block")
        ]
        self.layers = nn.ModuleList()
        for block_cfg in blocks:
            n_layers = block_cfg.n_layers
            Block = import_module(f"modules.{block_cfg.BlockType}").Block
            layers = nn.ModuleList(
                [
                    Block(
                        d_model=d_model,
                        config=block_cfg,
                        factory_kwargs=factory_kwargs,
                        layer_idx=i,
                        **kwargs,
                    ).to(device)
                    for i in range(len(self.layers), len(self.layers) + n_layers)
                ]
            )
            self.layers += layers
        assert len(self.layers) == n_layer

        # Initialize norm:
        norm_epsilon: float = 1e-5
        norm_cls = self.config.MixerModel.input.lm_head_prenorm
        if norm_cls == "layer":
            self.final_layernorm = nn.LayerNorm(d_model, eps=norm_epsilon).to(device)
        else:
            raise Exception(f"Norm class {norm_cls} is not valid.")

        return

    def allocate_inference_cache(self, *args, **kwargs):
        return {
            i: layer.allocate_inference_cache(*args, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        return_hidden_states=False,
        inference_params=None,
        position_ids=None,
    ):

        # Start running the layers
        hidden_states = self.embedding(input_ids)

        # Initialize outputs
        outputs = {
            "last_hidden_state": None,
            "all_hidden_states": (hidden_states,) if return_hidden_states else (),
            "all_transfer_matrices": tuple(),
            "all_mamba_outputs": tuple(),
        }

        # Run the layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                return_mixer_matrix=return_mixer_matrix,
                return_mamba_outputs=return_mamba_outputs,
                inference_params=inference_params,
                position_ids=position_ids,
            )
            # Record outputs
            hidden_states = layer_outputs["hidden_states"]
            if return_hidden_states:
                outputs["all_hidden_states"] += (hidden_states,)
            if return_mamba_outputs:
                outputs["all_mamba_outputs"] += (layer_outputs["mamba_hidden_states"],)
            if return_mixer_matrix:
                outputs["all_transfer_matrices"] += (layer_outputs["transfer_matrix"],)

        # Last layer, apply layer norm
        outputs["last_hidden_state"] = self.final_layernorm(hidden_states)
        return outputs
