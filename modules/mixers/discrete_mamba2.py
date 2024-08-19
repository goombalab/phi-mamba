import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nn.activation import Activation

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None

from modules.mixers.discrete_mamba2_ref_impl import (
    ssd_minimal_discrete,
    to_transfer_matrix_ssd,
)
from ops.triton.flashmamba import mamba_chunk_scan_fused


class Mixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        nheads=32,
        d_conv=4,
        expand=1,
        norm_cls="rms",
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=128,
        use_ref_impl=False,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        **kwargs,
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.nheads = nheads
        self.headdim = self.d_inner // self.nheads
        assert self.nheads == self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0
        self.norm_cls = norm_cls
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_ref_impl = use_ref_impl
        self.layer_idx = layer_idx
        self.bias = bias
        self.kwargs = kwargs

        # Projections
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + self.nheads * self.d_state * 2 + self.nheads,
            bias=bias,
            **factory_kwargs,
        )
        self.z_bias = (
            nn.Parameter(torch.zeros(self.d_inner, device=device)) if not bias else 0
        )  # make sure z_bias always exists

        # Convolutional layer
        conv_dim = self.d_inner + self.nheads * self.d_state * 2
        self.conv_bias = conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Activation after conv
        self.act = Activation(self.activation)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._optim = {"weight_decay": 0.0}

        # Norm before out_proj
        if self.norm_cls in ["rms", "rmsnorm"]:
            assert RMSNorm is not None
            self.norm = RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        elif self.norm_cls in ["layer", "layernorm"]:
            self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        elif self.norm_cls in ["none", "identity"]:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm class {self.norm_cls}")

        # out_proj
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def forward(self, u, return_mixer_matrix=False, inference_params=None, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        outputs = {}
        # assert state is None
        batch, seqlen, dim = u.shape

        state = None
        if inference_params is not None:
            state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # States are updated inplace
                out, _ = self.step(u, state)
                return {"hidden_states": out}

        # Hacky way to initialize state during inference
        chunk_size = self.chunk_size if state is None else seqlen

        # Pad input to nearest multiple of chunklen
        padded_len = (1 + (seqlen - 1) // chunk_size) * chunk_size
        u = F.pad(u, (0, 0, 0, padded_len - seqlen))

        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [self.d_inner + self.nheads * self.d_state * 2, self.d_inner, self.nheads],
            dim=-1,
        )
        z = z + self.z_bias

        A_log = -F.softplus(A_log).to(
            dtype=xBC.dtype
        )  # F.softplus(torch.tensor([0.5413]))=1

        if state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC[:, :seqlen, :], "b l d -> b d l")
            state["conv"].copy_(
                F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
            )  # Update state (B D W)

        # Convolutional layer
        xBC = self.convolutional_forward(xBC, padded_len)

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.nheads * self.d_state, self.nheads * self.d_state],
            dim=-1,
        )
        B, C = [rearrange(M, "b l (h n) -> b l h n", h=self.nheads) for M in (B, C)]

        # SSM
        y, ssm_state, T = self.ssm_forward(
            x, A_log, B, C, self.D, chunk_size, state, return_mixer_matrix
        )
        if state is not None:
            state["ssm"].copy_(ssm_state)

        # Norm and gate
        y = (
            self.norm(y, z)
            if self.norm_cls in ["rms", "rmsnorm"]
            else self.norm(y) * F.silu(z)
        )

        # Out proj
        out = self.out_proj(y)
        out = out[:, :seqlen, :]

        # store outputs
        outputs["hidden_states"] = out
        if return_mixer_matrix:
            outputs["transfer_matrix"] = T
        return outputs

    def step(self, u, state, **kwargs):
        # x: (B D)
        u = u.squeeze(1)

        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [self.d_inner + self.nheads * self.d_state * 2, self.d_inner, self.nheads],
            dim=-1,
        )
        z = z + self.z_bias

        A_log = -F.softplus(A_log).to(dtype=A_log.dtype)

        xBC, conv_state = self.convolutional_step(xBC, state["conv"])

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.nheads * self.d_state, self.nheads * self.d_state],
            dim=-1,
        )
        x, B, C = [
            rearrange(M, "b (h s) -> b h s", h=self.nheads) for M in (x, B, C)
        ]  # s is headdim for x and d_state for B and C

        y, ssm_state = self.ssm_step(
            X=x, A_log=A_log, B=B, C=C, initial_states=state["ssm"].to(x.dtype)
        )

        y = y + self.D[:, None] * x
        y = rearrange(y, "b h p -> b (h p)")

        # Norm and gate
        y = (
            self.norm(y, z)
            if self.norm_cls in ["rms", "rmsnorm"]
            else self.norm(y) * F.silu(z)
        )
        out = self.out_proj(y)

        # update state in place
        state["ssm"].copy_(ssm_state)
        state["conv"].copy_(conv_state)
        return out, {"conv": conv_state, "ssm": ssm_state}

    def ssm_forward(
        self, x, A_log, B, C, D, chunk_size, state, return_mixer_matrix=False
    ):
        """
        Arguments:
            x: (batch, seqlen, nheads, headdim)
            A_log: (batch, seqlen, nheads)
            B: (batch, seqlen, nheads, dstate)
            C: (batch, seqlen, nheads, dstate)
            D: (nheads)

        Return:
            y: (batch, seqlen, nheads, headdim)
            T: (batch, nheads, seqlen, seqlen)
        """
        if return_mixer_matrix:
            # Since the transfer matrix will be equated to the attention matrix,
            # we need to support the form: torch.matmul(attn_weights, value_states)
            T = to_transfer_matrix_ssd(A_log=A_log, B=B, C=C, D=D)
            T = rearrange(T, "b h z l -> b h l z")
            X = rearrange(x, "b l (h p) -> b h l p", h=self.nheads, p=self.headdim)
            y = torch.matmul(T, X)
            y = rearrange(y, "b h l p -> b l (h p)")
            return y, None, T

        X = rearrange(x, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        if self.use_ref_impl:
            y, ssm_state = ssd_minimal_discrete(
                X=X,
                A_log=A_log,
                B=B,
                C=C,
                block_len=chunk_size,
                initial_states=state["ssm"].to(X.dtype),
            )
        else:
            # This is a hacky way to use previous implementation without dt
            y, ssm_state = mamba_chunk_scan_fused(
                x=X / A_log.unsqueeze(-1),
                dt=rearrange(A_log, "b (c l) h -> b h c l", l=chunk_size),
                A=torch.ones(self.nheads, device=A_log.device),
                B=B,
                C=C,
                D=None,
                z=None,
            )
            ssm_state = ssm_state[:, -1]
        Du = torch.einsum("h,blhp->blhp", D, X)
        y = rearrange(y + Du, "b l h p -> b l (h p)")

        return y, ssm_state, None

    def ssm_step(self, X, A_log, B, C, initial_states=None):
        """
        Arguments:
            X: (batch, n_heads, d_head)
            A_log: (batch, n_heads)
            B: (batch, n_heads, d_state)
            C: (batch, n_heads, d_state)
            initial_states: (batch, n_heads, d_state, d_state) or None
        Return:
            Y: (batch, n_heads, d_head)
            final_state: (batch, n_heads, d_head, d_state)
        """
        # Compute Y:
        Bx = torch.einsum("bhn,bhp->bhpn", B, X)  # Bx
        Ah = torch.einsum("bh,bhpn->bhpn", torch.exp(A_log), initial_states)
        final_state = Ah + Bx
        Y = torch.einsum("bhn,bhpn->bhp", C, final_state)

        return Y, final_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.in_proj.weight.device
        # conv_state:
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_conv,
            self.conv1d.weight.shape[0],
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        # ssm_state:
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return {"conv": conv_state, "ssm": ssm_state}

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        """
        conv_state: (batch, d_conv, conv1d.weight.shape[0])
        ssm_state: (batch, nheads, headdim, d_state)
        """
        assert self.layer_idx is not None
        # Allocate memory if not exists
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[
                self.layer_idx
            ] = self.allocate_inference_cache(
                batch_size, inference_params.max_seqlen, dtype=torch.float32
            )
        # Get states
        states = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            states["conv"].zero_()
            states["ssm"].zero_()
        return states

    def convolutional_forward(self, xBC, padded_len):
        if causal_conv1d_fn is None or self.activation not in [
            "silu",
            "swish",
            "identity",
        ]:
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2)
            )
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)
        return xBC

    def convolutional_step(self, xBC, conv_state):
        # Convolutional layer
        conv_state = conv_state.to(xBC.dtype)
        if causal_conv1d_update:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation if self.activation != "identity" else None,
            )
        else:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv_bias:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(xBC.dtype)  # Some activations change dtype

        return xBC, conv_state
