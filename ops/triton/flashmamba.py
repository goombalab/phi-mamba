# Copyright (c) 2023, Tri Dao.

"""We want triton==2.1.0 for this
"""

import math

import causal_conv1d
import causal_conv1d_cuda
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from causal_conv1d import causal_conv1d_fn
from einops import rearrange, repeat
from torch.cuda.amp import custom_bwd, custom_fwd

from ops.triton.bmm import bmm
from ops.triton.layernorm import _layer_norm_bwd, _layer_norm_fwd, rms_norm_fn
from ops.triton.ssd_chunk_scan import (
    _chunk_scan_bwd_dC,
    _chunk_scan_bwd_dcb,
    _chunk_scan_bwd_ddAcs_stable,
    _chunk_scan_bwd_ddtcs,
    _chunk_scan_bwd_dstates,
    _chunk_scan_bwd_dx,
    _chunk_scan_bwd_dz,
    _chunk_scan_fwd,
    chunk_scan,
    chunk_scan_ref,
)
from ops.triton.ssd_chunk_state import (
    _chunk_state_bwd_db,
    _chunk_state_bwd_dx,
    _chunk_state_fwd,
    chunk_state,
    chunk_state_ref,
)
from ops.triton.ssd_state_passing import (
    _state_passing_bwd,
    _state_passing_fwd,
    state_passing,
    state_passing_ref,
)
from ops.triton.triton_matmul import matmul as matmul_triton


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
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
    ],
    key=["chunk_size", "hdim", "dstate"],
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: max(triton.next_power_of_2(args["dstate"]), 16)}
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr,
    cb_ptr,
    dout_ptr,
    dt_ptr,
    dt_cumsum_ptr,
    D_ptr,
    b_ptr,
    dstates_ptr,
    dx_ptr,
    ddt_ptr,
    dD_ptr,
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
    stride_cb_batch,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_dout_batch,
    stride_dout_seqlen,
    stride_dout_head,
    stride_dout_hdim,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dt_cs_batch,
    stride_dt_cs_chunk,
    stride_dt_cs_head,
    stride_dt_cs_csize,
    stride_D_head,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_dstates_head,
    stride_dstates_hdim,
    stride_dstates_dstate,
    stride_dx_batch,
    stride_dx_seqlen,
    stride_dx_head,
    stride_dx_hdim,
    stride_ddt_batch,
    stride_ddt_chunk,
    stride_ddt_head,
    stride_ddt_csize,
    stride_dD_batch,
    stride_dD_chunk,
    stride_dD_head,
    stride_dD_csize,
    stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
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
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_h * stride_cb_head
    dout_ptr += (
        pid_b * stride_dout_batch
        + pid_c * chunk_size * stride_dout_seqlen
        + pid_h * stride_dout_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += (
        pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    )
    dt_cumsum_ptr += (
        pid_b * stride_dt_cs_batch
        + pid_c * stride_dt_cs_chunk
        + pid_h * stride_dt_cs_head
    )
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + pid_h * stride_b_head
    )
    dstates_ptr += (
        pid_b * stride_dstates_batch
        + pid_c * stride_dstates_chunk
        + pid_h * stride_dstates_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dt_cs_m = tl.load(
        dt_cumsum_ptr + offs_m * stride_dt_cs_csize,
        mask=offs_m < chunk_size_limit,
        other=0.0,
    ).to(tl.float32)
    dt_cs_last = tl.load(dt_cumsum_ptr + (chunk_size - 1) * stride_dt_cs_csize).to(
        tl.float32
    )
    scale = tl.exp(dt_cs_last - dt_cs_m)
    # Might be faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    # However, we're getting error with the Triton compiler for that code path:
    # Unexpected mma -> mma layout conversion
    # So we're disabling that code path for now.
    # offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if False else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (
        offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate
    )
    dstates_ptrs = dstates_ptr + (
        offs_n[None, :] * stride_dstates_hdim
        + offs_dstate[:, None] * stride_dstates_dstate
    )
    # if BLOCK_SIZE_DSTATE <= 128:
    if False:
        b = tl.load(
            b_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate),
            other=0.0,
        )
        dstates = tl.load(
            dstates_ptrs,
            mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(
                b_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_dstate[None, :] < dstate - k),
                other=0.0,
            )
            dstates = tl.load(
                dstates_ptrs,
                mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (
        offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
    )
    dout_ptrs = dout_ptr + (
        offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim
    )
    dt_cumsum_ptrs = dt_cumsum_ptr + offs_k * stride_dt_cs_csize
    # Idk why limiting K_MAX gives wrong results, is it a Triton bug?
    # K_MAX = min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    K_MAX = chunk_size_limit
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        # For some reason setting mask to (offs_m[:, None] < chunk_size_limit) is much slower
        cb = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k),
            other=0.0,
        )
        dout = tl.load(
            dout_ptrs,
            mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        dt_cs_k = tl.load(dt_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0).to(
            tl.float32
        )
        cb *= tl.exp(dt_cs_k[None, :] - dt_cs_m[:, None])
        mask = k + offs_k[None, :] >= offs_m[:, None]
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dt_cumsum_ptrs += BLOCK_SIZE_K * stride_dt_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += (
        pid_b * stride_dx_batch
        + pid_c * chunk_size * stride_dx_seqlen
        + pid_h * stride_dx_head
    )
    dx_ptrs = dx_ptr + (
        offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim
    )
    if HAS_D:
        dout_res_ptrs = dout_ptr + (
            offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim
        )
        dout_res = tl.load(
            dout_res_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(
                D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0
            ).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(
        dx_ptrs,
        dx,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
    )

    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    x = tl.load(
        x_ptrs,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
        other=0.0,
    ).to(tl.float32)
    if HAS_D:
        dD_ptr += (
            pid_b * stride_dD_batch
            + pid_c * stride_dD_chunk
            + pid_h * stride_dD_head
            + pid_m * stride_dD_csize
        )
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=8,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=8,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=8,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=8,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
    ],
    key=["chunk_size", "hdim", "dstate"],
)
@triton.heuristics(
    {"BLOCK_SIZE_M": lambda args: max(triton.next_power_of_2(args["chunk_size"]), 16)}
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: max(triton.next_power_of_2(args["dstate"]), 16)}
)
@triton.jit
def _chunk_state_bwd_ddAcs_stable_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    dstates_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    ddA_cumsum_ptr,
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
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_ddA_cs_batch,
    stride_ddA_cs_chunk,
    stride_ddA_cs_head,
    stride_ddA_cs_csize,
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
    ddA_cumsum_ptr += (
        pid_b * stride_ddA_cs_batch
        + pid_c * stride_ddA_cs_chunk
        + pid_h * stride_ddA_cs_head
    )
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
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

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(
        tl.float32
    )
    acc *= tl.exp(dA_cs_last - dA_cs_m)[:, None]

    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    x = tl.load(
        x_ptrs,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
        other=0.0,
    ).to(tl.float32)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    ddt = tl.sum(acc * x, axis=1)
    # ddA_cs = -(ddt * dt_m)
    ddA_cs = tl.cumsum(ddt * dt_m)
    ddt_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    # tl.atomic_add(ddt_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
    tl.atomic_add(
        ddt_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size - 1
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
        ),
    ],
    key=["chunk_size", "dstate", "hdim"],
)
@triton.heuristics(
    {"BLOCK_SIZE_K": lambda args: max(triton.next_power_of_2(args["hdim"]), 16)}
)
@triton.jit
def _chunk_scan_bwd_ddAcs_prev_kernel(
    # Pointers to matrices
    dout_ptr,
    prev_states_ptr,
    C_ptr,
    dA_cumsum_ptr,
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size,
    dstate,
    hdim,
    batch,
    seqlen,
    nchunks,
    # Strides
    stride_dout_batch,
    stride_dout_seqlen,
    stride_dout_head,
    stride_dout_hdim,
    stride_prev_states_batch,
    stride_prev_states_chunk,
    stride_prev_states_head,
    stride_prev_states_hdim,
    stride_prev_states_dstate,
    stride_C_batch,
    stride_C_seqlen,
    stride_C_head,
    stride_C_dstate,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_ddA_cs_batch,
    stride_ddA_cs_chunk,
    stride_ddA_cs_head,
    stride_ddA_cs_csize,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    if pid_c == 0:
        return
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    dout_ptr += (
        pid_b * stride_dout_batch
        + pid_c * chunk_size * stride_dout_seqlen
        + pid_h * stride_dout_head
    )
    prev_states_ptr += (
        pid_b * stride_prev_states_batch
        + (pid_c - 1) * stride_prev_states_chunk
        + pid_h * stride_prev_states_head
    )
    C_ptr += (
        pid_b * stride_C_batch
        + pid_c * chunk_size * stride_C_seqlen
        + pid_h * stride_C_head
    )
    ddA_cumsum_ptr += (
        pid_b * stride_ddA_cs_batch
        + pid_c * stride_ddA_cs_chunk
        + pid_h * stride_ddA_cs_head
    )
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (
        offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim
    )
    prev_states_ptrs = prev_states_ptr + (
        offs_n[None, :] * stride_prev_states_dstate
        + offs_k[:, None] * stride_prev_states_hdim
    )
    C_ptrs = C_ptr + (
        offs_m[:, None] * stride_C_seqlen + offs_n[None, :] * stride_C_dstate
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(
        dout_ptrs,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim),
        other=0.0,
    )
    prev_states = tl.load(
        prev_states_ptrs,
        mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate),
        other=0.0,
    )
    prev_states = prev_states.to(dout_ptrs.dtype.element_ty)
    acc = tl.dot(dout, prev_states)
    c = tl.load(
        C_ptrs,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate),
        other=0.0,
    ).to(tl.float32)
    ddA_cs = tl.sum(acc * c, axis=1)
    dt_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(
        tl.float32
    )
    ddA_cs *= tl.exp(dt_cs_m)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ddt_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    tl.atomic_add(ddt_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)


def _chunk_scan_chunk_state_bwd_dx(
    x, dt, dt_cumsum, B, CB, dout, dstates, D=None, dx=None
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = B.shape[-1]
    assert B.shape == (batch, seqlen, dstate) or B.shape == (
        batch,
        seqlen,
        nheads,
        dstate,
    )
    BC_has_head = B.dim() == 4
    assert (
        CB.shape == (batch, nchunks, chunk_size, chunk_size)
        if not BC_has_head
        else (batch, nchunks, nheads, chunk_size, chunk_size)
    )
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dt_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(
            triton.cdiv(chunk_size, BLOCK_SIZE_min),
            batch,
            nchunks,
            nheads,
            headdim if D.dim() == 2 else 1,
            device=D.device,
            dtype=torch.float32,
        )
    else:
        dD = None
    dD_strides = (
        (dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
        if D is not None
        else (0, 0, 0, 0, 0)
    )
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    ddt = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32
    )
    grid_dx = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x,
            CB,
            dout,
            dt,
            dt_cumsum,
            D,
            B,
            dstates,
            dx,
            ddt,
            dD,
            chunk_size,
            headdim,
            dstate,
            batch,
            seqlen,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            CB.stride(0),
            CB.stride(1),
            0 if not BC_has_head else CB.stride(2),
            CB.stride(-1),
            CB.stride(-2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dt_cumsum.stride(0),
            dt_cumsum.stride(2),
            dt_cumsum.stride(1),
            dt_cumsum.stride(3),
            D.stride(0) if D is not None else 0,
            B.stride(0),
            B.stride(1),
            0 if not BC_has_head else B.stride(2),
            B.stride(-1),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            dstates.stride(4),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dx.stride(3),
            ddt.stride(0),
            ddt.stride(2),
            ddt.stride(1),
            ddt.stride(3),
            dD_strides[1],
            dD_strides[2],
            dD_strides[3],
            dD_strides[0],
            dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs[
            "BLOCK_SIZE_M"
        ]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return dx, ddt.to(dtype=dt.dtype), dD


def _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates):
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
    ddA_cumsum = torch.empty(
        batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32
    )
    grid_ddtcs = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_ddAcs_stable_kernel[grid_ddtcs](
            x,
            B,
            dstates,
            dt,
            dA_cumsum,
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
            ddA_cumsum.stride(0),
            ddA_cumsum.stride(2),
            ddA_cumsum.stride(1),
            ddA_cumsum.stride(3),
        )
    return ddA_cumsum


def _chunk_scan_bwd_ddAcs_prev(prev_states, C, dout, dA_cumsum):
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert C.shape == (batch, seqlen, dstate) or C.shape == (
        batch,
        seqlen,
        nheads,
        dstate,
    )
    C_has_head = C.dim() == 4
    ddA_cumsum_prev = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32
    )
    grid_ddAcs = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_ddAcs_prev_kernel[grid_ddAcs](
            dout,
            prev_states,
            C,
            dA_cumsum,
            ddA_cumsum_prev,
            chunk_size,
            dstate,
            headdim,
            batch,
            seqlen,
            nchunks,
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            prev_states.stride(0),
            prev_states.stride(1),
            prev_states.stride(2),
            prev_states.stride(3),
            prev_states.stride(4),
            C.stride(0),
            C.stride(1),
            0 if not C_has_head else C.stride(2),
            C.stride(-1),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            ddA_cumsum_prev.stride(0),
            ddA_cumsum_prev.stride(2),
            ddA_cumsum_prev.stride(1),
            ddA_cumsum_prev.stride(3),
        )
    return ddA_cumsum_prev


def _mamba_chunk_scan_fused_fwd(x, dt, A, B, C, D=None, z=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = B.shape[-1]
    assert B.shape == (batch, seqlen, dstate) or B.shape == (
        batch,
        seqlen,
        nheads,
        dstate,
    )
    BC_has_head = B.dim() == 4
    assert seqlen == nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    dt = dt.float()  # We want high precision for this before cumsum
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if (
        x.stride(-1) != 1 and x.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=True)
    states = rearrange(
        _state_passing_fwd(
            rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1]
        ),
        "... (p n) -> ... p n",
        n=dstate,
    )
    if not BC_has_head:
        CB = bmm(
            rearrange(C, "b (c l) n -> b c l n", c=nchunks),
            rearrange(B, "b (c l) n -> b c l n", c=nchunks),
            output_dtype=torch.float32,
        )
    else:
        CB = bmm(
            rearrange(C, "b (c l) h n -> b c h l n", c=nchunks),
            rearrange(B, "b (c l) h n -> b c h l n", c=nchunks),
            output_dtype=torch.float32,
        )
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z)
    return out, out_x, dt, dA_cumsum, CB, states


def _mamba_chunk_scan_fused_bwd(
    dout,
    x,
    dt,
    A,
    B,
    C,
    out,
    dA_cumsum,
    CB,
    states,
    dt_dtype,
    D=None,
    z=None,
    dx=None,
    ddt=None,
    dB=None,
    dC=None,
    dz=None,
    recompute_output=False,
):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = B.shape[-1]
    BC_has_head = B.dim() == 4
    assert dout.shape == (batch, seqlen, nheads, headdim)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt, dtype=dt_dtype)
    if z is not None:
        # dz, dout, dD, ddA_cumsum = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, D=D, dz=dz)
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(
            x,
            z,
            out,
            dout,
            chunk_size=chunk_size,
            has_ddAcs=False,
            D=D,
            dz=dz,
            recompute_output=recompute_output,
        )
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out
    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, dtype=states.dtype)
    # Do computation in fp32 but convert dstates and states to fp16/bf16 since dstates and states
    # will be used in matmul in the next kernels.
    dstates, ddA_chunk_cumsum, states = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        rearrange(dstates, "... p n -> ... (p n)"),
        # fused=True,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
    )
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
    dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, B_has_head=BC_has_head)
    dC = _chunk_scan_bwd_dC(states.to(x.dtype), dA_cumsum, dout, C_has_head=BC_has_head)
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, BC_has_head=BC_has_head)
    dCB = dCB.to(CB.dtype)
    if not BC_has_head:
        bmm(
            rearrange(dCB, "b c l s -> b c s l"),
            rearrange(C, "b (c l) n -> b c n l", c=nchunks),
            residual=rearrange(dB, "b (c s) n -> b c s n", c=nchunks),
            out=rearrange(dB_given, "b (c s) n -> b c s n", c=nchunks),
        )
        bmm(
            rearrange(dCB, "b c l s -> b c l s"),
            rearrange(B, "b (c s) n -> b c n s", c=nchunks),
            residual=rearrange(dC, "b (c l) n -> b c l n", c=nchunks),
            out=rearrange(dC_given, "b (c l) n -> b c l n", c=nchunks),
        )
    else:
        bmm(
            rearrange(dCB, "b c h l s -> b c h s l"),
            rearrange(C, "b (c l) h n -> b c h n l", c=nchunks),
            residual=rearrange(dB, "b (c s) h n -> b c h s n", c=nchunks),
            out=rearrange(dB_given, "b (c s) h n -> b c h s n", c=nchunks),
        )
        bmm(
            rearrange(dCB, "b c h l s -> b c h l s"),
            rearrange(B, "b (c s) h n -> b c h n s", c=nchunks),
            residual=rearrange(dC, "b (c l) h n -> b c h l n", c=nchunks),
            out=rearrange(dC_given, "b (c l) h n -> b c h l n", c=nchunks),
        )
    # This function will not compute the contribution to ddA_cumsum[:, :, :, -1]. Instead this was
    # computed during _state_passing_bwd
    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(
        x, dt, dA_cumsum, B, CB, dout, dstates, D=D, dx=dx
    )
    # If we have z, then dout_x is recomputed in fp32 so dD = (dout_x * x).sum() is more accurate
    # then dD_from_x = (dout_x * x).sum() where dout_x is in fp16/bf16
    if z is None:
        dD = dD_from_x

    ddA_cumsum_prev = _chunk_scan_bwd_ddAcs_prev(states, C, dout, dA_cumsum)
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    ddA_next = _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates)

    ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB) + ddA_next + ddA_prev

    dA = (ddA * dt).sum((-1, -2)).to(A.dtype)
    torch.add(ddt, ddA * rearrange(A, "h -> h 1 1"), out=ddt_given)

    return_vals = (dx, ddt_given, dA, dB_given, dC_given, dD, dz)
    return return_vals if not recompute_output else (*return_vals, rest[0])


def selective_scan_bwd(dout, x, dt, A, B, C, D=None, z=None):
    """
    Argument:
        dout: (batch, seqlen, nheads, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size) or (batch, nheads, headdim, nchunks, chunk_size)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    import selective_scan

    batch, seqlen, nheads, headdim = x.shape
    chunk_size = dt.shape[-1]
    dstate = B.shape[-1]
    x = rearrange(x, "b l h p -> b (h p) l")
    squeeze_dt = dt.dim() == 4
    if dt.dim() == 4:
        dt = repeat(dt, "b h c l -> b h p c l", p=headdim)
    dt = rearrange(dt, "b h p c l -> b (h p) (c l)", p=headdim)
    squeeze_A = A.dim() == 1
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    BC_has_head = B.dim() == 4
    if not BC_has_head:
        B = rearrange(B, "b l n -> b n l")
        C = rearrange(C, "b l n -> b n l")
    else:
        B = rearrange(B, "b l h n -> b h n l")
        C = rearrange(C, "b l h n -> b h n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")

    if x.stride(-1) != 1:
        x = x.contiguous()
    if dt.stride(-1) != 1:
        dt = dt.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    if B.dim() == 3:
        B = rearrange(B, "b dstate l -> b 1 dstate l")
    if C.dim() == 3:
        C = rearrange(C, "b dstate l -> b 1 dstate l")
    _, intermediate, *rest = selective_scan.fwd(
        x, dt.to(dtype=x.dtype), A, B, C, D, z, None, False
    )
    if z is not None:
        out = rest[0]
    else:
        out = None

    dout = rearrange(dout, "b l h p -> b (h p) l")

    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
    # backward of selective_scan with the backward of chunk).
    # Here we just pass in None and dz will be allocated in the C++ code.
    _, ddt, dA, *rest = selective_scan.bwd(
        x,
        dt.to(dtype=x.dtype),
        A,
        B,
        C,
        D,
        z,
        None,
        dout,
        intermediate,
        out,
        None,
        False,
        False,  # option to recompute out_z, not used here
    )
    ddt = rearrange(ddt, "b (h p) (c l) -> b h p c l", p=headdim, l=chunk_size)
    if squeeze_dt:
        ddt = ddt.float().sum(dim=2)
    if squeeze_A:
        dA = rearrange(dA, "(h p) n -> h p n", p=headdim).sum(dim=(1, 2))
    return ddt, dA


class MambaChunkScanFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, D=None, z=None):
        ctx.dt_dtype = dt.dtype
        out, out_x, dt, dA_cumsum, CB, states = _mamba_chunk_scan_fused_fwd(
            x, dt, A, B, C, D=D, z=z
        )
        ctx.save_for_backward(
            out if z is None else out_x, x, dt, dA_cumsum, A, B, C, CB, states, D, z
        )
        return out, states

    @staticmethod
    def backward(ctx, dout):
        out, x, dt, dA_cumsum, A, B, C, CB, states, D, z = ctx.saved_tensors
        dx, ddt, dA, dB, dC, dD, dz = _mamba_chunk_scan_fused_bwd(
            dout, x, dt, A, B, C, out, dA_cumsum, CB, states, ctx.dt_dtype, D=D, z=z
        )
        return dx, ddt, dA, dB, dC, dD, dz


def mamba_chunk_scan_fused(x, dt, A, B, C, D=None, z=None):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        A: (nheads)
        B: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    return MambaChunkScanFusedFn.apply(x, dt, A, B, C, D, z)


def mamba_chunk_scan(x, dt, A, B, C, D=None, z=None):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        A: (nheads)
        B: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    dstate = B.shape[-1]
    dt = dt.float()  # We want high precision for this before cumsum
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True)
    # 2. Pass the state to all the chunks by weighted cumsum.
    states = rearrange(
        state_passing(
            rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1]
        ),
        "... (p n) -> ... p n",
        n=dstate,
    )
    # 3. Compute the output for each chunk
    out = chunk_scan(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out


def mamba_chunk_scan_ref(x, dt, A, B, C, D=None, z=None):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        A: (nheads)
        B: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    dstate = B.shape[-1]
    dt = dt.float()  # We want high precision for this before cumsum
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state_ref(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
    # 2. Pass the state to all the chunks by weighted cumsum.
    # state_passing_ref is much less numerically stable
    states = rearrange(
        state_passing(
            rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1]
        ),
        "... (p n) -> ... p n",
        n=dstate,
    )
    states = states.to(states_dtype)
    # 3. Compute the output for each chunk
    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out


def ssd_selective_scan(x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size) or (batch, nheads, headdim, nchunks, chunk_size)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, nheads, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,) or (nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    from src.ops.selective_scan_interface import selective_scan_fn

    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    x = rearrange(x, "b l h p -> b (h p) l")
    if dt.dim() == 4:
        dt = repeat(dt, "b h c l -> b h p c l", p=headdim)
    dt = rearrange(dt, "b h p c l -> b (h p) (c l)", p=headdim)
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    BC_has_head = B.dim() == 4
    if not BC_has_head:
        B = rearrange(B, "b l n -> b n l")
        C = rearrange(C, "b l n -> b n l")
    else:
        B = rearrange(B, "b l h n -> b h n l")
        C = rearrange(C, "b l h n -> b h n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")
    if dt_bias is not None:
        if dt_bias.dim() == 1:
            dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
        dt_bias = rearrange(dt_bias, "h p -> (h p)")
    out = selective_scan_fn(
        x, dt, A, B, C, D=D, z=z, delta_bias=dt_bias, delta_softplus=dt_softplus
    )
    return rearrange(out, "b (h p) l -> b l h p", p=headdim)


class MambaConv1dScanFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xBC,
        conv1d_weight,
        conv1d_bias,
        dt,
        A,
        D,
        z=None,
        activation="silu",
        headdim=None,
    ):
        assert activation in [None, "silu", "swish"]
        batch, nheads, nchunks, chunk_size = dt.shape
        if z is not None:
            dim = z.shape[-1]
            assert dim % nheads == 0
            headdim = dim // nheads
        else:
            if D.dim() == 1:
                assert headdim is not None
            else:
                headdim = D.shape[1]
            dim = nheads * headdim
        ctx.dt_dtype = dt.dtype
        dstate = (xBC.shape[-1] - dim) // 2
        # xBC_conv = rearrange(
        #     causal_conv1d_cuda.causal_conv1d_fwd(rearrange(xBC, "b s d -> b d s"),
        #                                     conv1d_weight, conv1d_bias, activation in ["silu", "swish"]),
        #     "b d s -> b s d"
        # )
        xBC_conv = rearrange(
            causal_conv1d_cuda.causal_conv1d_fwd(
                rearrange(xBC, "b s d -> b d s"),
                conv1d_weight,
                conv1d_bias,
                None,
                None,
                activation in ["silu", "swish"],
            ),
            "b d s -> b s d",
        )
        x, B, C = torch.split(xBC_conv, [dim, dstate, dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        out, out_x, dt, dA_cumsum, CB, states = _mamba_chunk_scan_fused_fwd(
            x, dt, A, B, C, D=D, z=z
        )
        ctx.save_for_backward(
            xBC,
            conv1d_weight,
            conv1d_bias,
            out if z is None else out_x,
            dt,
            dA_cumsum,
            A,
            CB,
            states,
            D,
            z,
        )
        ctx.activation = activation
        ctx.headdim = headdim
        return rearrange(out, "b s h p -> b s (h p)")

    @staticmethod
    def backward(ctx, dout):
        (
            xBC,
            conv1d_weight,
            conv1d_bias,
            out,
            dt,
            dA_cumsum,
            A,
            CB,
            states,
            D,
            z,
        ) = ctx.saved_tensors
        nheads = D.shape[0]
        headdim = ctx.headdim
        dim = nheads * headdim
        dstate = (xBC.shape[-1] - dim) // 2
        # Recompute x, B, C
        xBC_conv = rearrange(
            causal_conv1d_cuda.causal_conv1d_fwd(
                rearrange(xBC, "b s d -> b d s"),
                conv1d_weight,
                conv1d_bias,
                ctx.activation in ["silu", "swish"],
            ),
            "b d s -> b s d",
        )
        x, B, C = torch.split(xBC_conv, [dim, dstate, dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, dstate, dstate], dim=-1)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        dx, ddt, dA, dB, dC, dD, dz = _mamba_chunk_scan_fused_bwd(
            dout,
            x,
            dt,
            A,
            B,
            C,
            out,
            dA_cumsum,
            CB,
            states,
            ctx.dt_dtype,
            D=D,
            z=z,
            dx=dx,
            dB=dB,
            dC=dC,
        )
        dz = rearrange(dz, "b l h p -> b l (h p)") if dz is not None else None
        dxBC, dweight, dbias = causal_conv1d.causal_conv1d_bwd(
            rearrange(xBC, "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
            rearrange(dxBC, "b s d -> b d s"),
            None,
            ctx.activation in ["silu", "swish"],
        )
        dxBC = rearrange(dxBC, "b d s -> b s d")
        return dxBC, dweight, dbias, ddt, dA, dD, dz, None, None


def mamba_conv1d_scan_fused(
    xBC, conv1d_weight, conv1d_bias, dt, A, D, z=None, activation="silu", headdim=None
):
    """
    Argument:
        xBC: (batch, seqlen, dim + 2 * dstate) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * dstate, width)
        conv1d_bias: (dim + 2 * dstate,)
        dt: (batch, nheads, nchunks, chunk_size)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, dim)
        headdim: if D is 1D and z is None, headdim must be passed in
    Return:
        out: (batch, seqlen, dim)
    """
    return MambaConv1dScanFusedFn.apply(
        xBC, conv1d_weight, conv1d_bias, dt, A, D, z, activation, headdim
    )


def mamba_conv1d_scan_ref(
    xBC,
    conv1d_weight,
    conv1d_bias,
    dt,
    A,
    D,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    activation="silu",
    headdim=None,
):
    """
    Argument:
        xBC: (batch, seqlen, dim + 2 * dstate) where dim == nheads * headdim
        dt: (batch, nheads, nchunks, chunk_size) or (batch, nheads, headdim, nchunks, chunk_size)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, dim)
        dt_bias: (nheads) or (nheads, headdim)
        headdim: if D is 1D and z is None, headdim must be passed in
    Return:
        out: (batch, seqlen, dim)
    """
    batch, nheads, nchunks, chunk_size = dt.shape
    if z is not None:
        dim = z.shape[-1]
        assert dim % nheads == 0
        headdim = dim // nheads
    else:
        if D.dim() == 1:
            assert headdim is not None
        else:
            headdim = D.shape[1]
        dim = nheads * headdim
    xBC = rearrange(
        causal_conv1d_fn(
            rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation
        ),
        "b d s -> b s d",
    )
    dstate = (xBC.shape[-1] - dim) // 2
    x, B, C = torch.split(xBC, [dim, dstate, dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
    out = ssd_selective_scan(
        x,
        dt.to(x.dtype),
        A,
        B,
        C,
        D=D.float(),
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
    )
    return rearrange(out, "b s h p -> b s (h p)")


class MambaSplitConv1dScanFusedFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size,
        activation="silu",
        rmsnorm_weight=None,
        rmsnorm_eps=1e-6,
        outproj_weight=None,
        outproj_bias=None,
        headdim=None,
    ):
        assert activation in [None, "silu", "swish"]
        if D.dim() == 1:
            assert headdim is not None
            (nheads,) = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // 2
        assert seqlen % chunk_size == 0
        assert zxbcdt.shape == (batch, seqlen, 2 * dim + 2 * dstate + nheads)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        z, xBC, dt = torch.split(zxbcdt, [dim, dim + dstate * 2, nheads], dim=-1)
        xBC_conv = rearrange(
            causal_conv1d.causal_conv1d_fn(
                x=rearrange(xBC, "b s d -> b d s"),
                weight=conv1d_weight,
                bias=conv1d_bias,
                activation=activation,
            ),
            "b d s -> b s d",
        )
        x, B, C = torch.split(xBC_conv, [dim, dstate, dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        dt_in = dt.float() + dt_bias
        dt = F.softplus(dt_in)  # (B, L, nheads)
        dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size).contiguous()
        if rmsnorm_weight is None:
            out, out_x, dt, dA_cumsum, CB, states = _mamba_chunk_scan_fused_fwd(
                x, dt, A, B, C, D=D, z=z
            )
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
        else:
            out_x, _, dt, dA_cumsum, CB, states = _mamba_chunk_scan_fused_fwd(
                x, dt, A, B, C, D=D, z=None
            )
            # reshape input data into 2D tensor
            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            out, _, rstd = _layer_norm_fwd(
                x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, is_rms_norm=True
            )
            out = rearrange(out, "(b s) d -> b s d", b=batch)
        ctx.outproj_weight_dtype = (
            outproj_weight.dtype if outproj_weight is not None else None
        )
        if outproj_weight is not None:
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                bias = outproj_bias.to(dtype) if outproj_bias is not None else None
            out = F.linear(out, outproj_weight, outproj_bias)
        else:
            assert outproj_bias is None
        ctx.save_for_backward(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            out_x,
            dt,
            dt_in,
            dA_cumsum,
            A,
            CB,
            states,
            D,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
        )
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.headdim = headdim
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        (
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            out,
            dt,
            dt_in,
            dA_cumsum,
            A,
            CB,
            states,
            D,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
        ) = ctx.saved_tensors
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // 2
        recompute_output = outproj_weight is not None
        z, xBC, _ = torch.split(zxbcdt, [dim, dim + dstate * 2, nheads], dim=-1)
        # Recompute x, B, C
        # xBC_conv = rearrange(
        #     causal_conv1d_cuda.causal_conv1d_fwd(rearrange(xBC, "b s d -> b d s"),
        #                                     conv1d_weight, conv1d_bias, ctx.activation in ["silu", "swish"]),
        #     "b d s -> b s d"
        # )
        xBC_conv = rearrange(
            causal_conv1d_cuda.causal_conv1d_fwd(
                rearrange(xBC, "b s d -> b d s"),
                conv1d_weight,
                conv1d_bias,
                None,
                None,
                None,
                ctx.activation in ["silu", "swish"],
            ),
            "b d s -> b s d",
        )
        x, B, C = torch.split(xBC_conv, [dim, dstate, dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        dzxbcdt = torch.empty_like(zxbcdt)
        dz, dxBC_given, ddt_given = torch.split(
            dzxbcdt, [dim, dim + dstate * 2, nheads], dim=-1
        )
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, dstate, dstate], dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        if outproj_weight is not None:
            dout_og = dout
            dout = F.linear(dout, outproj_weight.t())
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        if rmsnorm_weight is None:
            dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
            dx, ddt, dA, dB, dC, dD, dz, *rest = _mamba_chunk_scan_fused_bwd(
                dout,
                x,
                dt,
                A,
                B,
                C,
                out,
                dA_cumsum,
                CB,
                states,
                dt.dtype,
                D=D,
                z=z,
                dx=dx,
                dB=dB,
                dC=dC,
                dz=dz,
                recompute_output=recompute_output,
            )
            out_for_linear = (
                rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
            )
            drmsnorm_weight = None
        else:
            batch = dout.shape[0]
            dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
            dz = rearrange(dz, "b l d -> (b l) d")
            x_rms = rearrange(out, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            dout, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(
                dy_rms,
                x_rms,
                rmsnorm_weight,
                None,
                ctx.rmsnorm_eps,
                None,
                rstd,
                z_rms,
                is_rms_norm=True,
                recompute_output=recompute_output,
                dz=dz,
            )
            out_for_linear = (
                rearrange(rest[0], "(b s) d -> b s d", b=batch)
                if recompute_output
                else None
            )
            dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
            dx, ddt, dA, dB, dC, dD, _ = _mamba_chunk_scan_fused_bwd(
                dout,
                x,
                dt,
                A,
                B,
                C,
                out,
                dA_cumsum,
                CB,
                states,
                dt.dtype,
                D=D,
                z=None,
                dx=dx,
                dB=dB,
                dC=dC,
            )

        if outproj_weight is not None:
            # matmul_triton can keep output dtype in fp32 to avoid a bf16/fp16->fp32 cast if using AMP
            # However, it's only faster on A100 and not H100
            if torch.cuda.get_device_capability(outproj_weight.device) == (9, 0):
                doutproj_weight = torch.einsum("bso,bsd->od", dout_og, out_for_linear)
            else:
                doutproj_weight = matmul_triton(
                    rearrange(dout_og, "b s o -> o (b s)"),
                    rearrange(out_for_linear, "b s d -> (b s) d"),
                    ctx.outproj_weight_dtype,
                )
            doutproj_bias = (
                dout_og.sum(dim=(0, 1)) if outproj_bias is not None else None
            )
        else:
            doutproj_weight, doutproj_bias = None, None
        dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
        dxBC_given, dweight, dbias = causal_conv1d_cuda.causal_conv1d_bwd(
            rearrange(xBC, "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
            None,
            rearrange(dxBC, "b s d -> b d s"),
            dxBC_given,
            None,
            ctx.activation in ["silu", "swish"],
        )
        dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        ddt = torch.sigmoid(dt_in) * rearrange(ddt, "b h c l -> b (c l) h")
        ddt_given.copy_(ddt)
        ddt_bias = ddt.sum(dim=(0, 1))  # (nheads,)
        return (
            dzxbcdt,
            dweight,
            dbias,
            ddt_bias,
            dA,
            dD,
            None,
            None,
            drmsnorm_weight,
            None,
            doutproj_weight,
            doutproj_bias,
            None,
        )


def mamba_split_conv1d_scan_fused(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * dstate, width)
        conv1d_bias: (dim + 2 * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
    Return:
        out: (batch, seqlen, dim)
    """
    return MambaSplitConv1dScanFusedFn.apply(
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size,
        activation,
        rmsnorm_weight,
        rmsnorm_eps,
        outproj_weight,
        outproj_bias,
        headdim,
    )


def mamba_split_conv1d_scan_ref(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * dstate, width)
        conv1d_bias: (dim + 2 * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
    Return:
        out: (batch, seqlen, dim)
    """
    if D.dim() == 1:
        assert headdim is not None
        (nheads,) = D.shape
    else:
        nheads, headdim = D.shape
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // 2
    assert seqlen % chunk_size == 0
    assert zxbcdt.shape == (batch, seqlen, 2 * dim + 2 * dstate + nheads)
    assert dt_bias.shape == (nheads,)
    assert A.shape == (nheads,)
    if rmsnorm_weight is not None:
        assert rmsnorm_weight.shape == (dim,)
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + dstate * 2, nheads], dim=-1)
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)  # (B, nheads, L)
    xBC = rearrange(
        causal_conv1d_fn(
            rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation
        ),
        "b d s -> b s d",
    )
    x, B, C = torch.split(xBC, [dim, dstate, dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
    out = ssd_selective_scan(
        x,
        dt.to(x.dtype),
        A,
        B,
        C,
        D=D.float(),
        z=z if rmsnorm_weight is None else None,
        dt_bias=dt_bias,
        dt_softplus=True,
    )
    out = rearrange(out, "b s h p -> b s (h p)")
    if rmsnorm_weight is not None:
        out = rmsnorm_fn(
            out,
            rmsnorm_weight,
            None,
            z=rearrange(z, "b l h p -> b l (h p)"),
            eps=rmsnorm_eps,
        )
    if outproj_weight is not None:
        out = F.linear(out, outproj_weight, outproj_bias)
    return out


def chunk_scan_old_ref(
    x, delta_tied, A_tied, B, C, delta_tied_chunk_cumsum, chunk_size
):
    """
    Arguments:
        x: (batch, seqlen, dim)
        delta_tied: (batch, nheads, seqlen)
        A_tied: (nheads,)
        B: (batch, seqlen, dstate)
        C: (batch, seqlen, dstate)
        delta_tied_chunk_cumsum: (batch, nheads, seqlen)
        chunk_size: int
    Return:
        out: (batch, seqlen, dim)
    """
    device = x.device
    batch, seqlen, dim = x.shape
    nheads = A_tied.shape[0]
    assert dim % nheads == 0
    dhead = dim // nheads
    dstate = B.shape[2]
    assert delta_tied.shape == (batch, nheads, seqlen)
    assert delta_tied_chunk_cumsum.shape == (batch, nheads, seqlen)
    assert A_tied.shape == (nheads,)
    assert B.shape == (batch, seqlen, dstate)
    assert C.shape == (batch, seqlen, dstate)
    if seqlen % chunk_size != 0:
        remainder = seqlen % chunk_size
        x = F.pad(x, (0, 0, 0, remainder))
        delta_tied = F.pad(delta_tied, (0, remainder))
        delta_tied_chunk_cumsum = F.pad(delta_tied, (0, remainder))
        B = F.pad(B, (0, 0, 0, remainder))
        C = F.pad(C, (0, 0, 0, remainder))
    nchunks = seqlen // chunk_size
    x = rearrange(x, "b (c l) (h p) -> b c l h p", c=nchunks, p=dhead)
    delta_tied = rearrange(delta_tied, "b h (c l) -> b h c l", c=nchunks)
    delta_tied_chunk_cumsum = rearrange(
        delta_tied_chunk_cumsum, "b h (c l) -> b h c l", c=nchunks
    )
    B = rearrange(B, "b (c l) n -> b c l n", c=nchunks)
    C = rearrange(C, "b (c l) n -> b c l n", c=nchunks)
    if C.is_cuda:
        scores = torch.einsum("bcln,bcsn->bcls", C, B)
    else:
        scores = bmm(C, B)
    # (batch, nheads, nchunks, chunk_size, chunk_size)
    delta_segment_sum = (
        delta_tied_chunk_cumsum[:, :, :, :, None]
        - delta_tied_chunk_cumsum[:, :, :, None, :]
    )
    # The (i, j) entry for i <= j is the sum of delta_tied[:, :, :, (i+1):(j+1)]
    # (batch, nheads, nchunks, chunk_size, chunk_size)
    decay = torch.exp(delta_segment_sum * rearrange(A_tied, "h -> h 1 1 1"))
    scores_decay = rearrange(scores, "b c l s -> b 1 c l s") * decay
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=device, dtype=bool), diagonal=0
    )
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhcls,bhcs,bcshp->bclhp", scores_decay, delta_tied, x)
    out = rearrange(out, "b c l h p -> b (c l) (h p)")[:, :seqlen]
    return out


if __name__ == "__main__":
    from flash_attn.utils.benchmark import (
        benchmark_all,
        benchmark_forward,
        pytorch_profiler,
    )

    torch.manual_seed(42)
    batch = 1
    seqlen = 8192
    chunk_size = 128
    nchunks = seqlen // chunk_size
    dim = 2048
    headdim = 64
    nheads = dim // headdim
    dstate = 64
    dtype = torch.bfloat16
    device = "cuda"
    a = torch.randn(batch, nchunks, chunk_size, dstate, dtype=dtype, device=device)
    b = torch.randn(
        batch,
        nchunks,
        chunk_size,
        dstate,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    c_ref = torch.einsum("bcln,bcsn->bcls", a, b)
    c = bmm(a, b)
    # print((c - c_ref).abs().max())
    c = bmm(a, b, causal=True)
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=device, dtype=bool), diagonal=0
    )
    c_causal_ref = c_ref.masked_fill(~causal_mask, 0)
    c = c.masked_fill(~causal_mask, 0)

    x = torch.randn(
        batch, nheads, headdim, nchunks, chunk_size, dtype=dtype, device=device
    )
    x = rearrange(x, "b h p c s -> b c s h p").detach().requires_grad_()
    dt = F.softplus(
        torch.randn(
            batch, nheads, nchunks, chunk_size, dtype=torch.float32, device=device
        )
        - 4
    ).requires_grad_()
    dt_cumsum = torch.cumsum(dt.detach(), dim=-1).requires_grad_()
    A_tied = (
        -torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))
    ).requires_grad_()
    # (batch, nheads, nchunks, chunk_size, chunk_size)
    delta_segment_sum = dt_cumsum[:, :, :, :, None] - dt_cumsum[:, :, :, None, :]
    decay = torch.exp(delta_segment_sum * rearrange(A_tied, "h -> h 1 1 1"))
    scores_decay = rearrange(c_ref, "b c l s -> b 1 c l s") * decay
    C = torch.randn(batch, nchunks, chunk_size, dstate, dtype=dtype, device=device)
    prev_states = torch.randn(
        batch, nchunks, nheads, headdim, dstate, dtype=dtype, device=device
    )

    decay = torch.randn(
        batch, nchunks, nheads, chunk_size, chunk_size, dtype=dtype, device=device
    )
    dt_chunk_cumsum = dt_cumsum[:, :, :, -1].requires_grad_()

    tflops = batch * nchunks * chunk_size * chunk_size * nheads * headdim * 2 / 10 ** 12
    optimal_usec = tflops / 312 * 10 ** 6
    gbytes = (
        2 * batch * nchunks * chunk_size * nheads * headdim * 2 / 10 ** 9
    )  # * 2 because of 1 read and 1 write
    optimal_usec_io = gbytes / (2 * 10 ** 3) * 10 ** 6
