# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for mla_prefill_ps_asm_fwd + mla_reduce_v1.

Targets the failure modes hit when wiring this kernel pair into vLLM's
chunked-context prefill path on DeepSeek R1:

  1. LSE correctness for tiles a single TG can absorb (num_splits == 1).
     Previously the reduce step skipped those tiles, leaving final_lse
     uninitialized. vLLM consumes final_lse to merge per-chunk outputs, so
     wrong LSE silently corrupted accuracy.

  2. Scratch-buffer sizing for the noncausal case where K can be much larger
     than Q per sequence (e.g. 8K cached context vs a few hundred new tokens).
     get_ps_metadata_info_v1 must size work_info and reduce_partial_map
     proportional to max_kvlen / kvlen_granularity, not just max_qlen.

The tests compare both the attention output AND final_lse against a torch
reference for causal (Q == K) and noncausal (Q < K) shapes across the
range of (batch, qlen, kvlen) that exercises both failure modes.

Run with:  pytest op_tests/test_mla_prefill_ps_chunked_context.py -v
"""

import math
import sys

import aiter
import pytest
import torch

from aiter import dtypes, per_tensor_quant
from aiter.jit.utils.chip_info import get_gfx


if get_gfx() == "gfx942":
    pytest.skip(
        "mla_prefill_ps_asm_fwd is only supported on gfx950",
        allow_module_level=True,
    )


torch.set_default_device("cuda")


_TILE_Q = 256
_TILE_KV = 128
_QK_HEAD_DIM = 192
_V_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = _QK_HEAD_DIM - _V_HEAD_DIM
_BLOCK_SIZE = 1


def _ref_masked_attention(query, key, value, scale, dtype, is_causal):
    """Torch reference: returns (out [s_q, h, v_d], lse [h, s_q])."""
    attn = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q, s_k = query.shape[0], key.shape[0]
        mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(
            diagonal=s_k - s_q
        )
        bias = torch.zeros(s_q, s_k, dtype=torch.float32, device=query.device)
        bias.masked_fill_(mask.logical_not(), float("-inf"))
        attn = attn + bias
    lse = attn.logsumexp(dim=-1)
    weights = torch.softmax(attn, dim=-1)
    out = torch.einsum("hqk,khd->qhd", weights, value.float())
    return out.to(dtype), lse


def _torch_mla_extend(
    q_bf16, kv_bf16, qo_indptr, kv_indptr, kv_indices, softmax_scale, is_causal
):
    """Per-sequence torch reference. Returns (out [total_q, h, v_d], lse [h, total_q])."""
    bs = qo_indptr.shape[0] - 1
    kv_gather = torch.index_select(kv_bf16, 0, kv_indices)
    q_split = torch.tensor_split(q_bf16, qo_indptr.tolist()[1:-1])
    kv_split = torch.tensor_split(kv_gather, kv_indptr.tolist()[1:-1])

    outs, lses = [], []
    for i in range(bs):
        q = q_split[i]
        kvc = kv_split[i]
        if q.shape[0] == 0 or kvc.shape[0] == 0:
            outs.append(
                torch.zeros(
                    (q.shape[0], q.shape[1], _V_HEAD_DIM),
                    dtype=torch.bfloat16,
                    device=q.device,
                )
            )
            lses.append(
                torch.full(
                    (q.shape[1], q.shape[0]),
                    float("-inf"),
                    dtype=torch.float32,
                    device=q.device,
                )
            )
            continue
        k = kvc
        v = kvc[..., :_V_HEAD_DIM]
        o, lse = _ref_masked_attention(
            q, k, v, softmax_scale, torch.bfloat16, is_causal=is_causal
        )
        outs.append(o)
        lses.append(lse)
    out = torch.concat(outs, dim=0)
    lse = torch.concat(lses, dim=1)
    return out, lse


def _build_ps_metadata(
    qo_indptr, kv_indptr, seq_lens_kv, num_head_kv, is_causal,
):
    """Allocate PS scratch buffers and run get_ps_metadata_v1."""
    device = qo_indptr.device
    batch_size = qo_indptr.numel() - 1
    max_qlen = int((qo_indptr[1:] - qo_indptr[:-1]).max().item())
    max_kvlen = int(seq_lens_kv.max().item())
    gqa_ratio = 1  # MHA-style: num_q_heads == num_kv_heads
    qhead_granularity = gqa_ratio
    qlen_granularity = _TILE_Q // qhead_granularity
    kvlen_granularity = max(_TILE_KV, _BLOCK_SIZE)

    sizes = aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_head_kv,
        max_qlen=max_qlen,
        qlen_granularity=qlen_granularity,
        max_kvlen=max_kvlen,
        kvlen_granularity=kvlen_granularity,
    )
    (
        (work_meta_size, work_meta_dtype),
        (work_indptr_size, work_indptr_dtype),
        (work_info_size, work_info_dtype),
        (reduce_indptr_size, reduce_indptr_dtype),
        (reduce_final_map_size, reduce_final_map_dtype),
        (reduce_partial_map_size, reduce_partial_map_dtype),
    ) = sizes

    work_metadata_ptrs = torch.empty(
        work_meta_size, dtype=work_meta_dtype, device=device
    )
    work_indptr = torch.empty(
        work_indptr_size, dtype=work_indptr_dtype, device=device
    )
    work_info = torch.empty(
        *work_info_size, dtype=work_info_dtype, device=device
    )
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
    )
    reduce_final_map = torch.empty(
        *reduce_final_map_size, dtype=reduce_final_map_dtype, device=device
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_dtype, device=device
    )

    aiter.get_ps_metadata_v1(
        qo_indptr.cpu(),
        kv_indptr.cpu(),
        seq_lens_kv.cpu(),
        gqa_ratio,
        num_head_kv,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        qhead_granularity=qhead_granularity,
        qlen_granularity=qlen_granularity,
        kvlen_granularity=kvlen_granularity,
        block_size=_BLOCK_SIZE,
        is_causal=is_causal,
    )
    return {
        "work_indptr": work_indptr,
        "work_info": work_info,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


def _run_asm_kernel_pair(
    q_bf16, kv_bf16, qo_indptr, kv_indptr, kv_indices, seq_lens_kv,
    num_head_kv, is_causal, softmax_scale,
):
    """Run mla_prefill_ps_asm_fwd + mla_reduce_v1; return (out_bf16, final_lse_fp32)."""
    device = q_bf16.device
    total_q, num_heads, _ = q_bf16.shape
    max_qlen = int((qo_indptr[1:] - qo_indptr[:-1]).max().item())

    q_quant, q_scale = per_tensor_quant(q_bf16, quant_dtype=dtypes.fp8)
    k_quant, k_scale = per_tensor_quant(kv_bf16, quant_dtype=dtypes.fp8)
    v_quant, v_scale = per_tensor_quant(
        kv_bf16[..., :_V_HEAD_DIM].contiguous(), quant_dtype=dtypes.fp8
    )

    meta = _build_ps_metadata(
        qo_indptr, kv_indptr, seq_lens_kv, num_head_kv, is_causal
    )

    output = torch.zeros(
        (total_q, num_heads, _V_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    n_partial_slots = meta["reduce_partial_map"].size(0)
    partial_out = torch.zeros(
        (n_partial_slots * _TILE_Q, num_heads, _V_HEAD_DIM),
        dtype=dtypes.fp32,
        device=device,
    )
    partial_lse = torch.full(
        (n_partial_slots * _TILE_Q, num_heads),
        float("-inf"),
        dtype=dtypes.fp32,
        device=device,
    )
    final_lse = torch.full(
        (total_q, num_heads), float("-inf"), dtype=dtypes.fp32, device=device
    )

    aiter.mla_prefill_ps_asm_fwd(
        q_quant,
        k_quant,
        v_quant,
        qo_indptr,
        kv_indptr,
        kv_indices,
        meta["work_indptr"],
        meta["work_info"],
        max_qlen,
        softmax_scale,
        is_causal,
        partial_out,
        partial_lse,
        output,
        q_scale,
        k_scale,
        v_scale,
    )
    aiter.mla_reduce_v1(
        partial_out,
        partial_lse,
        meta["reduce_indptr"],
        meta["reduce_final_map"],
        meta["reduce_partial_map"],
        _TILE_Q,
        output,
        final_lse,
    )
    return output, final_lse


def _make_inputs(seq_lens_q, seq_lens_kv, num_heads, device="cuda", seed=0):
    """Build packed-varlen tensors. KV is unpaged (block_size=1)."""
    torch.manual_seed(seed)
    batch_size = len(seq_lens_q)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(
        torch.tensor(seq_lens_q, dtype=torch.int32, device=device), dim=0
    )
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(seq_lens_kv, dtype=torch.int32, device=device), dim=0
    )
    total_q = int(qo_indptr[-1].item())
    total_kv = int(kv_indptr[-1].item())

    q_bf16 = (
        torch.randn(
            (total_q, num_heads, _QK_HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        * 0.5
    )
    kv_bf16 = (
        torch.randn(
            (total_kv, num_heads, _QK_HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        * 0.5
    )
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    seq_lens_kv_t = torch.tensor(seq_lens_kv, dtype=torch.int32, device=device)
    return q_bf16, kv_bf16, qo_indptr, kv_indptr, kv_indices, seq_lens_kv_t


def _check(name, ref, got, rtol, atol):
    diff = (ref.float() - got.float()).abs()
    finite = torch.isfinite(diff)
    max_abs = diff[finite].max().item() if finite.any() else 0.0
    denom = ref.float().abs()[finite]
    max_rel = (
        (diff[finite] / denom.clamp(min=1e-6)).max().item() if finite.any() else 0.0
    )
    ok = torch.allclose(ref.float(), got.float(), rtol=rtol, atol=atol, equal_nan=False)
    assert ok, (
        f"[{name}] mismatch: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}, "
        f"rtol={rtol}, atol={atol}\nref[:3]={ref.flatten()[:3]}, "
        f"got[:3]={got.flatten()[:3]}"
    )


# Shapes chosen to cover:
# - Q == K causal (the previously-working path)
# - Q < K noncausal (chunked-context path)
# - kv_split_per_qtile > cus_per_cluster (the sizing-bug regime)
# - num_splits == 1 tiles (the LSE-bug regime: small batches, single q-tile)
# - varlen across the batch (mixed split counts)
@pytest.mark.parametrize(
    "name,seq_lens_q,seq_lens_kv,is_causal",
    [
        # LSE-bug regime: single batch, single q-tile, num_splits==1.
        ("causal_single_unsplit", [128], [128], True),
        # Causal multi-batch baseline.
        ("causal_varlen_small", [128, 256, 384], [128, 256, 384], True),
        # Noncausal Q << K (chunked-context shape, small).
        ("noncausal_q64_k1024", [64, 64], [1024, 1024], False),
        # Noncausal large-KV regime that triggers the sizing bug.
        # max_kv_split_per_qtile = ceil(8192/128) = 64, far above cus_per_cluster.
        ("noncausal_q128_k8192", [128, 128], [8192, 8192], False),
        # Varlen noncausal: mixed K lengths, including a sequence with very
        # large K to exercise both the sizing fix and the dedup in reduce.
        ("noncausal_varlen_mixed", [96, 192, 64], [4096, 8000, 1024], False),
        # Causal large-context (matches dsv3 prefill chunk shape).
        ("causal_large", [4096], [4096], True),
        # Chunked-context shapes where MOST sequences have K=0 for this chunk
        # (the realistic vLLM case: only the few sequences whose cached context
        # extends into this chunk contribute K; the rest have K=0 and the
        # scheduler must skip their q-tiles cleanly). Decode-heavy batches
        # have q=1 per sequence with sparse K coverage.
        (
            "noncausal_sparse_k_decode_heavy",
            [1] * 16 + [1],
            [0] * 16 + [8000],
            False,
        ),
        (
            "noncausal_sparse_k_mixed",
            [1] * 8 + [500] + [1] * 8,
            [0, 0, 4096, 0, 0, 8000, 0, 0, 500, 0, 0, 0, 8000, 0, 0, 0, 0],
            False,
        ),
        # Big prefill seq alongside many zero-K decode tokens (the dsv3
        # context_3(8073)_generation_119(119) shape, scaled down).
        (
            "noncausal_one_prefill_many_zero_k",
            [7893] + [1] * 119,
            [8000] + [0] * 119,
            False,
        ),
        # All-zero K: every sequence has K=0 for this chunk. num_partial_tiles
        # should be 0 and the kernel must not OOB on empty work.
        ("noncausal_all_zero_k", [1] * 8, [0] * 8, False),
    ],
)
@pytest.mark.parametrize("num_heads", [1, 16])
def test_asm_kernel_pair_matches_torch(
    name, seq_lens_q, seq_lens_kv, is_causal, num_heads
):
    if not is_causal:
        for sq, sk in zip(seq_lens_q, seq_lens_kv):
            assert sq <= sk, "noncausal requires Q <= K per sequence"

    softmax_scale = 1.0 / math.sqrt(_QK_HEAD_DIM)
    q_bf16, kv_bf16, qo_indptr, kv_indptr, kv_indices, seq_lens_kv_t = _make_inputs(
        seq_lens_q, seq_lens_kv, num_heads
    )

    out_asm, lse_asm = _run_asm_kernel_pair(
        q_bf16,
        kv_bf16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        seq_lens_kv_t,
        num_heads,
        is_causal,
        softmax_scale,
    )
    out_ref, lse_ref = _torch_mla_extend(
        q_bf16,
        kv_bf16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        softmax_scale,
        is_causal,
    )

    # Output: fp8 path, accept the same tolerances as test_mla_prefill_ps.py.
    _check(f"{name}/output", out_ref, out_asm, rtol=5e-2, atol=5e-2)

    # LSE: this is the metric we actually care about for chunked-context
    # correctness. Reference LSE is fp32 from bf16 inputs; ASM LSE is fp32
    # accumulated in fp8 path, so allow a similar tolerance.
    lse_asm_t = lse_asm.transpose(0, 1)  # [h, total_q]
    finite_mask = torch.isfinite(lse_ref)
    if finite_mask.any():
        _check(
            f"{name}/lse",
            lse_ref[finite_mask],
            lse_asm_t[finite_mask],
            rtol=5e-2,
            atol=5e-2,
        )
    else:
        # All-zero-K case: reference LSE is all -inf. The kernel must not
        # crash; we just verify it returned without faulting (already true
        # if we got here) and that the ASM LSE is also fully -inf at the
        # positions the reference flagged.
        assert not torch.isfinite(lse_asm_t).any(), (
            f"[{name}/lse] expected all -inf but got finite values"
        )


def test_lse_is_written_for_unsplit_tile():
    """Regression test for the num_splits==1 bug.

    With a single short sequence that fits in one TG, the scheduler emits a
    single work_info entry per head. Before the fix, mla_reduce_v1 would skip
    that tile (gated on num_splits > 1) and final_lse stayed at its init value
    (we pre-fill -inf to make the bug observable).
    """
    softmax_scale = 1.0 / math.sqrt(_QK_HEAD_DIM)
    q_bf16, kv_bf16, qo_indptr, kv_indptr, kv_indices, seq_lens_kv_t = _make_inputs(
        [64], [64], num_heads=1, seed=42
    )
    _, lse_asm = _run_asm_kernel_pair(
        q_bf16,
        kv_bf16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        seq_lens_kv_t,
        num_head_kv=1,
        is_causal=True,
        softmax_scale=softmax_scale,
    )
    # Every position should have a finite LSE.
    assert torch.isfinite(lse_asm).all(), (
        "final_lse contains -inf for an unsplit tile: "
        "mla_reduce_v1 did not write it back. "
        "Check reduce.cu num_splits==1 branch."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
