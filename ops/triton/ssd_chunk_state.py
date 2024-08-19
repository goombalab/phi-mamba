# Copyright (c) 2023, Tri Dao.

"""We want triton==2.1.0 for this
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange


def init_to_zero(names):
    return lambda nargs: [
        nargs[name].zero_() for name in names if nargs[name] is not None
    ]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dt_cumsum_ptr,
    # Matrix dimensions
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dt_cs_batch,
    stride_dt_cs_chunk,
    stride_dt_cs_head,
    stride_dt_cs_csize,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + pid_h * stride_b_head
    )
    x_ptr += (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dt_cumsum_ptr += (
        pid_b * stride_dt_cs_batch
        + pid_c * stride_dt_cs_chunk
        + pid_h * stride_dt_cs_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dt_cs_last = tl.load(dt_cumsum_ptr + (chunk_size - 1) * stride_dt_cs_csize).to(
        tl.float32
    )
    dt_cumsum_ptrs = dt_cumsum_ptr + offs_k * stride_dt_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dt_cs_k = tl.load(
            dt_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        b *= (tl.exp((dt_cs_last - dt_cs_k)) * dt_k)[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dt_cumsum_ptrs += BLOCK_SIZE_K * stride_dt_cs_csize
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += (
        pid_b * stride_states_batch
        + pid_c * stride_states_chunk
        + pid_h * stride_states_head
    )
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr", "ddt_cumsum_ptr"]),
        ),
    ],
    key=["chunk_size", "hdim", "dstate"],
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: max(triton.next_power_of_2(args["dstate"]), 16)}
)
@triton.jit
def _chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    dstates_ptr,
    dt_ptr,
    dt_cumsum_ptr,
    dx_ptr,
    ddt_ptr,
    ddt_cumsum_ptr,
    # Matrix dimensions
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dt_cs_batch,
    stride_dt_cs_chunk,
    stride_dt_cs_head,
    stride_dt_cs_csize,
    stride_dx_batch,
    stride_dx_seqlen,
    stride_dx_head,
    stride_dx_hdim,
    stride_ddt_batch,
    stride_ddt_chunk,
    stride_ddt_head,
    stride_ddt_csize,
    stride_ddt_cs_batch,
    stride_ddt_cs_chunk,
    stride_ddt_cs_head,
    stride_ddt_cs_csize,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + pid_h * stride_b_head
    )
    dstates_ptr += (
        pid_b * stride_dstates_batch
        + pid_c * stride_dstates_chunk
        + pid_h * stride_states_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += (
        pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    )
    ddt_cumsum_ptr += (
        pid_b * stride_ddt_cs_batch
        + pid_c * stride_ddt_cs_chunk
        + pid_h * stride_ddt_cs_head
    )
    dt_cumsum_ptr += (
        pid_b * stride_dt_cs_batch
        + pid_c * stride_dt_cs_chunk
        + pid_h * stride_dt_cs_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k = tl.arange(
        0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
    )
    b_ptrs = b_ptr + (
        offs_m[:, None] * stride_b_seqlen + offs_k[None, :] * stride_b_dstate
    )
    dstates_ptrs = dstates_ptr + (
        offs_n[None, :] * stride_states_hdim + offs_k[:, None] * stride_states_dstate
    )
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(
            b_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate),
            other=0.0,
        )
        dstates = tl.load(
            dstates_ptrs,
            mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(
                b_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_k[None, :] < dstate - k),
                other=0.0,
            )
            dstates = tl.load(
                dstates_ptrs,
                mask=(offs_k[:, None] < dstate - k) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dt_cs_last = tl.load(dt_cumsum_ptr + (chunk_size - 1) * stride_dt_cs_csize).to(
        tl.float32
    )
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_cumsum_ptrs = dt_cumsum_ptr + offs_m * stride_dt_cs_csize
    dt_cs_m = tl.load(dt_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(
        tl.float32
    )
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    acc *= tl.exp(dt_cs_last - dt_cs_m)[:, None]

    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    x = tl.load(
        x_ptrs,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
        other=0.0,
    ).to(tl.float32)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)
    ddt_cs = -(ddt * dt_m)
    ddt_cs_last = -tl.sum(ddt_cs)
    ddt_cumsum_ptrs = ddt_cumsum_ptr + offs_m * stride_ddt_cs_csize
    tl.atomic_add(ddt_cumsum_ptrs, ddt_cs, mask=offs_m < chunk_size)
    tl.atomic_add(ddt_cumsum_ptr + (chunk_size - 1) * stride_ddt_cs_csize, ddt_cs_last)

    dx = (acc * dt_m[:, None]).to(dx_ptr.dtype.element_ty)
    dx_ptr += (
        pid_b * stride_dx_batch
        + pid_c * chunk_size * stride_dx_seqlen
        + pid_h * stride_dx_head
    )
    dx_ptrs = dx_ptr + (
        offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim
    )
    tl.store(
        dx_ptrs,
        dx,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_stages=3, num_warps=4
        ),
    ],
    key=["chunk_size", "dstate", "hdim"],
)
@triton.heuristics(
    {"BLOCK_SIZE_K": lambda args: max(triton.next_power_of_2(args["hdim"]), 16)}
)
@triton.jit
def _chunk_state_bwd_db_kernel(
    # Pointers to matrices
    x_ptr,
    dstates_ptr,
    dt_ptr,
    dt_cumsum_ptr,
    db_ptr,
    # Matrix dimensions
    chunk_size,
    dstate,
    hdim,
    batch,
    seqlen,
    nheads,
    nheads_per_program,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dt_cs_batch,
    stride_dt_cs_chunk,
    stride_dt_cs_head,
    stride_dt_cs_csize,
    stride_db_batch,
    stride_db_seqlen,
    stride_db_head,
    stride_db_dstate,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * nheads_per_program * stride_x_head
    )
    db_ptr += (
        pid_b * stride_db_batch
        + pid_c * chunk_size * stride_db_seqlen
        + pid_h * stride_db_head
    )
    dstates_ptr += (
        pid_b * stride_dstates_batch
        + pid_c * stride_dstates_chunk
        + pid_h * nheads_per_program * stride_states_head
    )
    dt_ptr += (
        pid_b * stride_dt_batch
        + pid_c * stride_dt_chunk
        + pid_h * nheads_per_program * stride_dt_head
    )
    dt_cumsum_ptr += (
        pid_b * stride_dt_cs_batch
        + pid_c * stride_dt_cs_chunk
        + pid_h * nheads_per_program * stride_dt_cs_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_seqlen + offs_k[None, :] * stride_x_hdim
    )
    dstates_ptrs = dstates_ptr + (
        offs_n[None, :] * stride_states_dstate + offs_k[:, None] * stride_states_hdim
    )
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_cumsum_ptrs = dt_cumsum_ptr + offs_m * stride_dt_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    nheads_iter = min(nheads_per_program, nheads - pid_h * nheads_per_program)
    for h in range(nheads_iter):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim),
            other=0.0,
        )
        dstates = tl.load(
            dstates_ptrs,
            mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate),
            other=0.0,
        )
        dstates = dstates.to(x_ptrs.dtype.element_ty)
        db = tl.dot(x, dstates)
        dt_cs_last = tl.load(dt_cumsum_ptr + (chunk_size - 1) * stride_dt_cs_csize).to(
            tl.float32
        )
        dt_cs_m = tl.load(dt_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(
            tl.float32
        )
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        scale = tl.exp(dt_cs_last - dt_cs_m)
        acc += db * (scale * dt_m)[:, None]
        x_ptrs += stride_x_head
        dstates_ptrs += stride_states_head
        dt_ptrs += stride_dt_head
        dt_cumsum_ptr += stride_dt_cs_head
        dt_cumsum_ptrs += stride_dt_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    db_ptrs = db_ptr + (
        offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_dstate
    )
    tl.store(
        db_ptrs,
        acc,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate),
    )


def _chunk_state_fwd(B, x, dt, dA_cumsum, states=None, states_in_fp32=True):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = B.shape[-1]
    assert B.shape == (batch, seqlen, dstate) or B.shape == (
        batch,
        seqlen,
        nheads,
        dstate,
    )
    B_has_head = B.dim() == 4
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (batch, nchunks, nheads, headdim, dstate),
            device=x.device,
            dtype=states_dtype,
        )
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x,
            B,
            states,
            dt,
            dA_cumsum,
            headdim,
            dstate,
            chunk_size,
            batch,
            seqlen,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            B.stride(0),
            B.stride(1),
            0 if not B_has_head else B.stride(2),
            B.stride(-1),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
        )
    return states


def _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = B.shape[-1]
    assert B.shape == (batch, seqlen, dstate) or B.shape == (
        batch,
        seqlen,
        nheads,
        dstate,
    )
    B_has_head = B.dim() == 4
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if dx is not None:
        assert dx.shape == x.shape
    else:
        dx = torch.empty_like(x)
    ddt = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    ddA_cumsum = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dA_cumsum.device, dtype=torch.float32
    )
    grid_dx = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_dx_kernel[grid_dx](
            x,
            B,
            dstates,
            dt,
            dA_cumsum,
            dx,
            ddt,
            ddA_cumsum,
            chunk_size,
            headdim,
            dstate,
            batch,
            seqlen,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            B.stride(0),
            B.stride(1),
            0 if not B_has_head else B.stride(2),
            B.stride(-1),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            dstates.stride(4),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dx.stride(3),
            ddt.stride(0),
            ddt.stride(2),
            ddt.stride(1),
            ddt.stride(3),
            ddA_cumsum.stride(0),
            ddA_cumsum.stride(2),
            ddA_cumsum.stride(1),
            ddA_cumsum.stride(3),
        )
    return dx, ddt.to(dt.dtype), ddA_cumsum.to(dA_cumsum.dtype)


def _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, dB=None, B_has_head=False):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    dB_og = dB
    if dB_og is not None:
        assert (
            dB_og.shape == (batch, seqlen, dstate)
            if not B_has_head
            else (batch, seqlen, nheads, dstate)
        )
    if not B_has_head:
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        nheads_per_program = max(
            min(math.ceil(batch * nchunks * nheads / sm_count), nheads), 1
        )
    else:
        nheads_per_program = 1
    nsplits = triton.cdiv(nheads, nheads_per_program)
    dB = torch.empty(
        batch, seqlen, nsplits, dstate, device=x.device, dtype=torch.float32
    )
    grid_db = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nsplits,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_db_kernel[grid_db](
            x,
            dstates,
            dt,
            dA_cumsum,
            dB,
            chunk_size,
            dstate,
            headdim,
            batch,
            seqlen,
            nheads,
            nheads_per_program,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            dstates.stride(4),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            dB.stride(0),
            dB.stride(1),
            dB.stride(2),
            dB.stride(3),
        )
    if not B_has_head:
        dB = dB.sum(-2)
    if dB_og is not None:
        with torch.no_grad():
            dB_og.copy_(dB)
    return dB if dB_og is None else dB_og


class ChunkStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, x, dt, dA_cumsum, states_in_fp32=True):
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        dstate = B.shape[-1]
        assert seqlen <= nchunks * chunk_size
        assert B.shape == (batch, seqlen, dstate) or B.shape == (
            batch,
            seqlen,
            nheads,
            dstate,
        )
        B_has_head = B.dim() == 4
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if B.stride(-1) != 1:
            B = B.contiguous()
        if (
            x.stride(-1) != 1 and x.stride(1) != 1
        ):  # Either M or K dimension should be contiguous
            x = x.contiguous()
        states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=states_in_fp32)
        ctx.save_for_backward(B, x, dt, dA_cumsum)
        return states

    @staticmethod
    def backward(ctx, dstates):
        B, x, dt, dA_cumsum = ctx.saved_tensors
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        dstate = B.shape[-1]
        B_has_head = B.dim() == 4
        assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
        if dstates.stride(-1) != 1:
            dstates = dstates.contiguous()
        dx, ddt, ddA_cumsum = _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates)
        dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, B_has_head=B_has_head)
        dB = dB.to(B.dtype)
        return dB, dx, ddt, ddA_cumsum, None


def chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True):
    """
    Argument:
        B: (batch, seqlen, dstate) or (batch, seqlen, nheads, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    return ChunkStateFn.apply(B, x, dt, dA_cumsum, states_in_fp32)


def chunk_state_ref(B, x, dt, dA_cumsum):
    """
    Argument:
        B: (batch, seqlen, dstate) or (batch, seqlen, nheads, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert B.shape == (batch, seqlen, dstate) or B.shape == (
        batch,
        seqlen,
        nheads,
        dstate,
    )
    B_has_head = B.dim() == 4
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        if not B_has_head:
            B = F.pad(B, (0, 0, 0, nchunks * chunk_size - seqlen))
        else:
            B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    if not B_has_head:
        return torch.einsum(
            "bcln,bhcl,bhcl,bclhp->bchpn",
            B.to(x.dtype),
            decay_states.to(x.dtype),
            dt.to(x.dtype),
            x,
        )
    else:
        return torch.einsum(
            "bclhn,bhcl,bhcl,bclhp->bchpn",
            B.to(x.dtype),
            decay_states.to(x.dtype),
            dt.to(x.dtype),
            x,
        )
