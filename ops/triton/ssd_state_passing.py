# Copyright (c) 2023, Tri Dao.

"""We want triton==2.1.0 for this
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, repeat


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["dim"],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    dt_cs_ptr,
    # Matrix dimensions
    dim,
    nchunks,
    # Strides
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_dim,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_out_dim,
    stride_dt_cs_batch,
    stride_dt_cs_chunk,
    stride_dt_cs_head,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dt_cs_ptr += pid_b * stride_dt_cs_batch + pid_h * stride_dt_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    states_ptrs += stride_states_chunk
    dt_cs_ptr += stride_dt_cs_chunk
    out_ptrs += stride_out_chunk
    for _ in range(nchunks - 1):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dt_cs = tl.load(dt_cs_ptr).to(tl.float32)
        scale = tl.exp(dt_cs)
        states = scale * states + new_states
        tl.store(out_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dt_cs_ptr += stride_dt_cs_chunk
        out_ptrs += stride_out_chunk


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["dim"],
)
@triton.heuristics(
    {"CONVERT_STATES": lambda args: args["states_converted_ptr"] is not None}
)
@triton.jit
def _state_passing_bwd_kernel(
    # Pointers to matrices
    dout_ptr,
    out_ptr,
    dt_cs_ptr,
    dstates_ptr,
    ddt_cs_ptr,
    states_converted_ptr,
    # Matrix dimensions
    dim,
    nchunks,
    # Strides
    stride_dout_batch,
    stride_dout_chunk,
    stride_dout_head,
    stride_dout_dim,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_out_dim,
    stride_dt_cs_batch,
    stride_dt_cs_chunk,
    stride_dt_cs_head,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_dstates_head,
    stride_dstates_dim,
    stride_ddt_cs_batch,
    stride_ddt_cs_chunk,
    stride_ddt_cs_head,
    # Meta-parameters
    FUSED: tl.constexpr,
    CONVERT_STATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dstates_ptr += (
        pid_b * stride_dstates_batch
        + pid_h * stride_dstates_head
        + (nchunks - 1) * stride_dstates_chunk
    )
    dt_cs_ptr += (
        pid_b * stride_dt_cs_batch
        + pid_h * stride_dt_cs_head
        + (nchunks - 1) * stride_dt_cs_chunk
    )
    ddt_cs_ptr += (
        pid_b * stride_ddt_cs_batch
        + pid_h * stride_ddt_cs_head
        + (nchunks - 1) * stride_ddt_cs_chunk
        + pid_m
    )
    out_ptr += (
        pid_b * stride_out_batch
        + pid_h * stride_out_head
        + (nchunks - 1) * stride_out_chunk
    )
    dout_ptr += (
        pid_b * stride_dout_batch
        + pid_h * stride_dout_head
        + (nchunks - 1) * stride_dout_chunk
    )
    if CONVERT_STATES:
        states_converted_ptr += (
            pid_b * stride_out_batch
            + pid_h * stride_out_head
            + (nchunks - 1) * stride_out_chunk
        )

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dstates_ptrs = dstates_ptr + offs_m * stride_dstates_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    dout_ptrs = dout_ptr + offs_m * stride_dout_dim
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_m * stride_out_dim

    if not FUSED:
        dstates = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
        if CONVERT_STATES:
            out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            tl.store(states_converted_ptrs, out, mask=offs_m < dim)
        dout_ptrs -= stride_dout_chunk
        out_ptrs -= stride_out_chunk
        dstates_ptrs -= stride_dstates_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk
        for _ in range(nchunks - 1):
            dt_cs = tl.load(dt_cs_ptr).to(tl.float32)
            scale = tl.exp(dt_cs)
            out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if CONVERT_STATES:
                tl.store(states_converted_ptrs, out, mask=offs_m < dim)
            ddA = tl.sum(out * dstates) * scale
            tl.store(ddt_cs_ptr, ddA)
            dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            dstates = scale * dstates + dout
            tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
            dout_ptrs -= stride_dout_chunk
            dstates_ptrs -= stride_dstates_chunk
            dt_cs_ptr -= stride_dt_cs_chunk
            ddt_cs_ptr -= stride_ddt_cs_chunk
            out_ptrs -= stride_out_chunk
            if CONVERT_STATES:
                states_converted_ptrs -= stride_out_chunk
        tl.store(ddt_cs_ptr, 0.0)
    else:
        dstates = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=offs_m < dim)
        ddA = tl.sum(out * dstates)
        tl.store(ddt_cs_ptr, ddA)
        dout_ptrs -= stride_dout_chunk
        out_ptrs -= stride_out_chunk
        dstates_ptrs -= stride_dstates_chunk
        ddt_cs_ptr -= stride_ddt_cs_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk
        for _ in range(nchunks - 1):
            dt_cs = tl.load(dt_cs_ptr).to(tl.float32)
            scale = tl.exp(dt_cs)
            dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            dstates = scale * dstates + dout
            tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
            out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if CONVERT_STATES:
                tl.store(states_converted_ptrs, out, mask=offs_m < dim)
            ddA = tl.sum(out * dstates)
            tl.store(ddt_cs_ptr, ddA)
            dout_ptrs -= stride_dout_chunk
            dstates_ptrs -= stride_dstates_chunk
            dt_cs_ptr -= stride_dt_cs_chunk
            ddt_cs_ptr -= stride_ddt_cs_chunk
            out_ptrs -= stride_out_chunk
            if CONVERT_STATES:
                states_converted_ptrs -= stride_out_chunk


def _state_passing_fwd(states, dA_chunk_cumsum):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    out = torch.empty(
        (batch, nchunks, nheads, dim), device=states.device, dtype=states.dtype
    )
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states,
            out,
            dA_chunk_cumsum,
            dim,
            nchunks,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            dA_chunk_cumsum.stride(0),
            dA_chunk_cumsum.stride(2),
            dA_chunk_cumsum.stride(1),
        )
    return out


def _state_passing_bwd(
    states, dA_chunk_cumsum, dout, fused=False, dstates_dtype=None, states_dtype=None
):
    """
    If fused, the gradient ddA_chunk_cumsum here will also contain the contribution from chunk_state.
    """
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    assert dout.shape == states.shape
    dstates = torch.empty_like(
        dout, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype
    )
    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(
            dout, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype
        )
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None
    BLOCK_SIZE_min = 64
    n_blocks = (dim + BLOCK_SIZE_min - 1) // BLOCK_SIZE_min
    ddA_chunk_cumsum = torch.empty(
        batch,
        nheads,
        nchunks,
        n_blocks,
        dtype=torch.float32,
        device=dA_chunk_cumsum.device,
    )
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)
    with torch.cuda.device(dout.device.index):
        _state_passing_bwd_kernel[grid](
            dout,
            states,
            dA_chunk_cumsum,
            dstates,
            ddA_chunk_cumsum,
            states_converted,
            dim,
            nchunks,
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            dA_chunk_cumsum.stride(0),
            dA_chunk_cumsum.stride(2),
            dA_chunk_cumsum.stride(1),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            ddA_chunk_cumsum.stride(0),
            ddA_chunk_cumsum.stride(2),
            ddA_chunk_cumsum.stride(1),
            fused,
        )
    BLOCK_SIZE_actual = _state_passing_bwd_kernel.best_config.kwargs["BLOCK_SIZE"]
    n_valid_blocks = (dim + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    ddA_chunk_cumsum = (
        ddA_chunk_cumsum[..., :n_valid_blocks]
        .sum(dim=-1)
        .to(dtype=dA_chunk_cumsum.dtype)
    )
    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states
    return (
        (dstates, ddA_chunk_cumsum)
        if states_dtype is None
        else (dstates, ddA_chunk_cumsum, states_converted)
    )


class StatePassingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, states, dA_chunk_cumsum):
        batch, nchunks, nheads, dim = states.shape
        assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
        if states.stride(-1) != 1:
            states = states.contiguous()
        out = _state_passing_fwd(states, dA_chunk_cumsum)
        ctx.save_for_backward(out, dA_chunk_cumsum)
        return out

    @staticmethod
    def backward(ctx, dout):
        out, dA_chunk_cumsum = ctx.saved_tensors
        batch, nchunks, nheads, dim = out.shape
        assert dout.shape == out.shape
        assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        dstates, ddA_chunk_cumsum = _state_passing_bwd(out, dA_chunk_cumsum, dout)
        return dstates, ddA_chunk_cumsum


def state_passing(states, dA_chunk_cumsum):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
    Return:
        out: (batch, nchunks, nheads, dim)
    """
    return StatePassingFn.apply(states, dA_chunk_cumsum)


def state_passing_ref(states, dA_chunk_cumsum):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
    Return:
        out: (batch, nchunks, nheads, dim)
    """
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    # (batch, nheads, nchunks, nchunks)
    dt_chunk_segment_sum = (
        dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    )
    # (batch, nheads, nchunks, nchunks)
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(
        torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0
    )
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    return torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
