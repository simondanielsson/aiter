# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone reproducer for the AITER top-p sampling GPU memory-access fault
seen on gfx950 (MI350) when serving DeepSeek-V3 in vLLM with top_p < 1.

It drives torch.ops.aiter.top_p_sampling_from_probs directly, matching the EXACT
call vLLM makes in the ROCm sampler path
(vllm/v1/sample/ops/topk_topp_sampler.py::aiter_sample, top-p-only branch):

    probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
    next_token_ids = aiter_ops.top_p_sampling_from_probs(
        probs, None, *_to_tensor_scalar_tuple(p), deterministic=True
    )

CRITICAL FIDELITY DETAIL: in vLLM `p` is a per-request TENSOR, so
_to_tensor_scalar_tuple(p) -> (p_tensor, 0). vLLM therefore passes
maybe_top_p_arr=<tensor>, top_p_val=0 -- the kernel reads top_p_arr[row_idx].
An earlier version of this repro passed a Python float, which took the scalar
path (maybe_top_p_arr=None, top_p_val=0.9) and DID NOT reproduce. This version
defaults to the tensor path (REPRO_PARR=1).

Other things a faithful repro must do, because a single static uniform tensor
will pass even on a broken kernel:
  * draw FRESH probs every iteration (a rare pathological row is the trigger;
    reusing one tensor for N iters almost never hits it), and
  * include PATHOLOGICAL rows that real serving produces but torch.rand does
    not: rows with -inf logits (vocab padding / masked tokens), an all -inf row
    (empty/dummy DP batch), and NaN rows. These make softmax emit all-zero or
    NaN prob rows, which stress the pivot-bisection convergence and the
    "u close to 1" fallback in the kernel.

Run with the caching allocator OFF so any out-of-bounds access faults AT the
sampling kernel instead of silently poisoning a neighbor:

    PYTORCH_NO_HIP_MEMORY_CACHING=1 AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3 \
        python op_tests/repro_topp_maf.py

Knobs (env):
    REPRO_DIST=peaked        prob distribution: peaked | uniform | degenerate |
                             padded | allzero | nan | mixed | all
                             (default: all)
    REPRO_PARR=1             1 = per-request top_p TENSOR (vLLM default);
                             0 = scalar top_p_val; "both" = run each
    REPRO_DETERMINISTIC=1    1 = deterministic scan (vLLM default, wave64 suspect)
                             0 = hipcub BlockScan path; "both" = run each
    REPRO_FRESH=1            1 = draw fresh probs every iter (default);
                             0 = reuse one tensor (old, weaker behavior)
    REPRO_VOCAB=129280       vocab size (default: 129280, DeepSeek-V3)
    REPRO_BATCH=256          batch size / concurrency (default: 256)
    REPRO_P=0.9              top_p value (default: 0.9)
    REPRO_ITERS=1000         sampling calls per config (default: 1000)
    REPRO_SEED=0             base RNG seed (per-iter seed = base + iter)
    REPRO_LOGIT_SCALE=8.0    std of the normal used to build peaked logits
    REPRO_PAD_FRAC=0.1       fraction of vocab set to -inf in the 'padded' dist
    REPRO_BAD_ROWS=4         # of all-inf/nan rows injected in the 'mixed' dist

If a config faults, note WHICH one: the distribution + the parr/deterministic
flags localize the mechanism (see the bisection matrix printed at the end).
Each surviving config also verifies every finite-prob row's sample is inside the
true top-p nucleus, so a SILENT wrong answer (bad index, no fault) is caught too.
"""

import os

import torch

from aiter.ops import sampling  # noqa: F401  (registers torch.ops.aiter.*)

torch.set_default_device("cuda")

NEG_INF = float("-inf")


def _env(name, default):
    return os.environ.get(name, default)


def _build_logits(dist, batch, vocab, scale, pad_frac, bad_rows, gen):
    """Return [batch, vocab] float32 logits for the given distribution.

    Returns raw logits (pre-softmax) so the repro can softmax them exactly as
    vLLM does. Distributions that can yield non-finite prob rows are flagged via
    the returned `finite_row` boolean mask (those rows are excluded from the
    nucleus correctness check but STILL sent to the kernel -- they are the whole
    point).
    """
    finite_row = torch.ones(batch, dtype=torch.bool)

    if dist == "uniform":
        # Uniform probs == constant logits; use log(rand) so softmax != flat.
        logits = torch.rand(batch, vocab, generator=gen).clamp_min(1e-9).log()
        return logits, finite_row

    if dist == "peaked":
        return torch.randn(batch, vocab, generator=gen) * scale, finite_row

    if dist == "degenerate":
        logits = torch.full((batch, vocab), -30.0)
        cols = torch.arange(batch) % vocab
        logits[torch.arange(batch), cols] = 30.0
        return logits, finite_row

    if dist == "padded":
        # Realistic: a contiguous tail of the vocab is padding -> -inf.
        logits = torch.randn(batch, vocab, generator=gen) * scale
        n_pad = max(1, int(vocab * pad_frac))
        logits[:, vocab - n_pad:] = NEG_INF
        return logits, finite_row

    if dist == "allzero":
        # Every logit -inf -> softmax is all-NaN (0/0). Empty/dummy DP batch.
        logits = torch.full((batch, vocab), NEG_INF)
        finite_row[:] = False
        return logits, finite_row

    if dist == "nan":
        logits = torch.randn(batch, vocab, generator=gen) * scale
        logits[:, 0] = float("nan")
        finite_row[:] = False
        return logits, finite_row

    if dist == "mixed":
        # Mostly healthy peaked rows, plus a few pathological rows interleaved,
        # mimicking a real batch that contains some dummy/padded requests.
        logits = torch.randn(batch, vocab, generator=gen) * scale
        n_pad = max(1, int(vocab * pad_frac))
        logits[:, vocab - n_pad:] = NEG_INF
        nbad = min(bad_rows, batch)
        # Spread the bad rows across the batch (not just at the top).
        idx = torch.linspace(0, batch - 1, steps=nbad).long()
        for r, ridx in enumerate(idx.tolist()):
            if r % 2 == 0:
                logits[ridx, :] = NEG_INF          # empty row
            else:
                logits[ridx, ridx % vocab] = float("nan")  # nan row
            finite_row[ridx] = False
        return logits, finite_row

    raise ValueError(f"unknown REPRO_DIST={dist}")


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def _topp_mask(probs, p):
    """Boolean [batch, vocab] mask of tokens in the top-p nucleus.

    Mirrors the reference construction in test_sampling.py::test_top_p_sampling.
    Only meaningful for finite rows.
    """
    eps = 1e-4
    sorted_prob, indices = torch.sort(probs, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros_like(probs, dtype=torch.int32)
    mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    return mask


def _make_p_arg(use_parr, p, batch):
    """Return the (maybe_top_p_arr, top_p_val) tuple exactly as vLLM would.

    vLLM passes a per-request tensor -> (tensor, 0). Scalar path -> (None, p).
    """
    if use_parr:
        p_tensor = torch.full((batch,), p, dtype=torch.float32)
        return _to_tensor_scalar_tuple(p_tensor)
    return _to_tensor_scalar_tuple(p)


def _run_one(dist, use_parr, deterministic, fresh, batch, vocab, p, iters,
             scale, pad_frac, bad_rows, base_seed):
    tag = (
        f"dist={dist:<10} parr={int(use_parr)} det={int(deterministic)} "
        f"fresh={int(fresh)} batch={batch} vocab={vocab} p={p}"
    )
    print(f"[run] {tag} iters={iters} ...", flush=True)

    gen = torch.Generator(device="cuda")
    static_logits = None
    static_finite = None

    for it in range(iters):
        if fresh or static_logits is None:
            gen.manual_seed(base_seed + it)
            logits, finite_row = _build_logits(
                dist, batch, vocab, scale, pad_frac, bad_rows, gen
            )
            if not fresh:
                static_logits, static_finite = logits, finite_row
        else:
            logits, finite_row = static_logits, static_finite

        probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
        maybe_arr, pval = _make_p_arg(use_parr, p, batch)

        samples = torch.ops.aiter.top_p_sampling_from_probs(
            probs, None, maybe_arr, pval, deterministic=deterministic
        )
        # Force completion so a fault surfaces here, not later.
        torch.cuda.synchronize()

        s = samples.view(-1)
        if not bool((s >= 0).all() and (s < vocab).all()):
            bad = s[(s < 0) | (s >= vocab)]
            raise AssertionError(
                f"OUT-OF-RANGE sample at iter {it}: {tag} -> "
                f"{bad[:8].tolist()} (vocab={vocab})"
            )

        # Correctness check only on finite rows (pathological rows have no
        # well-defined nucleus; we only require they do not fault / go OOB).
        if bool(finite_row.all()):
            in_nucleus = _topp_mask(probs, p)[torch.arange(batch), s]
            if not bool((in_nucleus == 1).all()):
                n_bad = int((in_nucleus != 1).sum())
                raise AssertionError(
                    f"SILENT WRONG ANSWER at iter {it}: {tag} -> "
                    f"{n_bad} samples outside the top-p nucleus"
                )
    print(f"[ok ] {tag}", flush=True)


def main():
    dist = _env("REPRO_DIST", "all")
    parr = _env("REPRO_PARR", "1")
    det = _env("REPRO_DETERMINISTIC", "1")
    fresh = bool(int(_env("REPRO_FRESH", "1")))
    vocab = int(_env("REPRO_VOCAB", "129280"))
    batch = int(_env("REPRO_BATCH", "256"))
    p = float(_env("REPRO_P", "0.9"))
    iters = int(_env("REPRO_ITERS", "1000"))
    base_seed = int(_env("REPRO_SEED", "0"))
    scale = float(_env("REPRO_LOGIT_SCALE", "8.0"))
    pad_frac = float(_env("REPRO_PAD_FRAC", "0.1"))
    bad_rows = int(_env("REPRO_BAD_ROWS", "4"))

    all_dists = [
        "peaked", "uniform", "degenerate", "padded", "allzero", "nan", "mixed",
    ]
    dists = all_dists if dist == "all" else [dist]
    parr_flags = [True, False] if parr == "both" else [bool(int(parr))]
    det_flags = [True, False] if det == "both" else [bool(int(det))]

    print(
        f"gfx top-p repro: vocab={vocab} batch={batch} p={p} iters={iters} "
        f"seed={base_seed} scale={scale} pad_frac={pad_frac} bad_rows={bad_rows}",
        flush=True,
    )
    print(
        f"distributions={dists} parr={parr_flags} deterministic={det_flags} "
        f"fresh={fresh}",
        flush=True,
    )

    for d in dists:
        for pa in parr_flags:
            for df in det_flags:
                _run_one(d, pa, df, fresh, batch, vocab, p, iters, scale,
                         pad_frac, bad_rows, base_seed)

    print("\nALL CONFIGS PASSED (no fault, no wrong answer).", flush=True)
    print(
        "If this still passes on MI350, the standalone kernel is not the whole "
        "story: the fault likely needs the in-server interaction (memory "
        "pressure / concurrent decode / DP dummy batches). Next: bisect with "
        "REPRO_DIST=mixed REPRO_PARR=both REPRO_DETERMINISTIC=both.",
        flush=True,
    )


if __name__ == "__main__":
    main()
