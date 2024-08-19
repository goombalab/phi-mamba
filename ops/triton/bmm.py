# Copyright (out) 2023, Tri Dao.

"""We want triton==2.1.0 for this
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, repeat


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
    key=["M", "N", "K", "IS_CAUSAL"],
)
@triton.heuristics({"HAS_RESIDUAL": lambda args: args["res_ptr"] is not None})
@triton.jit
def bmm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    out_ptr,
    res_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    nheads,
    stride_a_batch,
    stride_a_chunk,
    stride_a_head,
    stride_am,
    stride_ak,
    stride_b_batch,
    stride_b_chunk,
    stride_b_head,
    stride_bn,
    stride_bk,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_outm,
    stride_outn,
    stride_res_batch,
    stride_res_chunk,
    stride_res_head,
    stride_resm,
    stride_resn,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // nheads
    pid_h = pid_ch - pid_c * nheads
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += pid_b * stride_a_batch + pid_c * stride_a_chunk + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * stride_b_chunk + pid_h * stride_b_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ).to(dot_dtype)
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_RESIDUAL:
        res_ptr += (
            pid_b * stride_res_batch
            + pid_c * stride_res_chunk
            + pid_h * stride_res_head
        )
        res_ptrs = res_ptr + (
            offs_m[:, None] * stride_resm + offs_n[None, :] * stride_resn
        )
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)).to(
            tl.float32
        )
        acc += res
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += (
        pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    )
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def bmm(a, b, residual=None, out=None, causal=False, output_dtype=None):
    """
    Argument:
        a: (batch, nchunks, m, k) or (batch, nchunks, nheads, m, k)
        b: (batch, nchunks, n, k) or (batch, nchunks, nheads, n, k)
        residual: (batch, nchunks, m, n) or (batch, nchunks, nheads, m, n)
    Return:
        out: (batch, nchunks, m, k) or (batch, nchunks, nheads, m, k)
    """
    # Check constraints.
    has_head = a.dim() == 5
    if not has_head:
        batch, nchunks, m, k = a.shape
        _, _, n, _ = b.shape
    else:
        batch, nchunks, nheads, m, k = a.shape
        _, _, _, n, _ = b.shape
    assert (
        b.shape == (batch, nchunks, n, k)
        if not has_head
        else (batch, nchunks, nheads, n, k)
    )
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(-2) != 1:
        b = b.contiguous()
    if residual is not None:
        assert (
            residual.shape == (batch, nchunks, m, n)
            if not has_head
            else (batch, nchunks, nheads, m, n)
        )
        if residual.stride(-1) != 1 and residual.stride(-2) != 1:
            residual = residual.contiguous()
    # Allocates output.
    if out is not None:
        assert (
            out.shape == (batch, nchunks, m, n)
            if not has_head
            else (batch, nchunks, nheads, m, n)
        )
        assert out.stride(-1) == 1 or out.stride(-2) == 1
    else:
        out_dtype = a.dtype if output_dtype is None else output_dtype
        out = torch.empty(
            (batch, nchunks, m, n) if not has_head else (batch, nchunks, nheads, m, n),
            device=a.device,
            dtype=out_dtype,
        )
    dot_dtype = (
        tl.bfloat16
        if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16
        else (
            tl.float16
            if a.dtype == torch.float16 or b.dtype == torch.float16
            else tl.float32
        )
    )
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
        batch,
        nchunks if not has_head else nchunks * nheads,
    )
    residual_strides = (
        (
            residual.stride(0),
            residual.stride(1),
            0 if not has_head else residual.stride(2),
            residual.stride(-2),
            residual.stride(-1),
        )
        if residual is not None
        else (0, 0, 0, 0, 0)
    )
    with torch.cuda.device(a.device.index):
        bmm_kernel[grid](
            a,
            b,
            out,
            residual,
            m,
            n,
            k,
            nheads if has_head else 1,
            a.stride(0),
            a.stride(1),
            0 if not has_head else a.stride(2),
            a.stride(-2),
            a.stride(-1),
            b.stride(0),
            b.stride(1),
            0 if not has_head else b.stride(2),
            b.stride(-2),
            b.stride(-1),
            out.stride(0),
            out.stride(1),
            0 if not has_head else out.stride(2),
            out.stride(-2),
            out.stride(-1),
            residual_strides[0],
            residual_strides[1],
            residual_strides[2],
            residual_strides[3],
            residual_strides[4],
            causal,
            dot_dtype,
        )
    return out
