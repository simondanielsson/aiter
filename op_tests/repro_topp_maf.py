# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone reproducer for the AITER top-p sampling GPU memory-access fault
seen on gfx950 (MI350) when serving DeepSeek-V3 in vLLM with top_p < 1.

It drives torch.ops.aiter.top_p_sampling_from_probs directly, using the EXACT
call vLLM makes in the ROCm sampler path:

    probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
    next_token_ids = aiter_ops.top_p_sampling_from_probs(
        probs, None, maybe_top_p_arr=None, top_p_val=p, deterministic=True
    )

(see vLLM vllm/v1/sample/ops/topk_topp_sampler.py::aiter_sample, top-p-only branch)

The upstream op_tests/test_sampling.py::test_top_p_sampling does NOT reproduce
this because it feeds uniform `torch.rand` probs. Real serving probs are
softmax(logits) which are highly PEAKED (a few tokens near 1.0, a long tail
near 0). This script sweeps the distribution shape, the deterministic flag,
vocab size, batch size, and p so the trigger can be bisected.

Run with the caching allocator OFF so any out-of-bounds access faults AT the
sampling kernel instead of silently poisoning a neighbor:

    PYTORCH_NO_HIP_MEMORY_CACHING=1 AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3 \
        python op_tests/repro_topp_maf.py

Knobs (env):
    REPRO_DIST=peaked        prob distribution: peaked | uniform | degenerate | all
                             (default: all -> runs every distribution in turn)
    REPRO_DETERMINISTIC=1    1 = deterministic scan (vLLM default, wave64 suspect)
                             0 = hipcub BlockScan path; "both" = run each
    REPRO_VOCAB=129280       vocab size (default: 129280, DeepSeek-V3)
    REPRO_BATCH=256          batch size / concurrency (default: 256)
    REPRO_P=0.9              top_p value (default: 0.9)
    REPRO_ITERS=1000         sampling calls per config (default: 1000)
    REPRO_SEED=0             RNG seed
    REPRO_LOGIT_SCALE=8.0    std of the normal used to build peaked logits

If it faults, the bug is isolated to the AITER kernel with a seconds-long repro
that can be bisected (toggle deterministic, swap distribution, shrink vocab).
Each surviving config also verifies the sample is inside the true top-p set, so
a SILENT wrong answer (no fault, bad index) is caught too.
"""

import os

import torch

from aiter.ops import sampling  # noqa: F401  (registers torch.ops.aiter.*)

torch.set_default_device("cuda")


def _env(name, default):
    return os.environ.get(name, default)


def _build_probs(dist: str, batch: int, vocab: int, scale: float) -> torch.Tensor:
    """Return a [batch, vocab] float32 prob tensor for the given distribution.

    peaked:     softmax(randn * scale) -- realistic serving distribution.
    uniform:    normalized rand -- what the existing upstream test uses.
    degenerate: one token ~1.0, rest ~0 -- top-p set of size 1, pivot-bisection
                edge case and prime MAF bait.
    """
    if dist == "uniform":
        pre = torch.rand(batch, vocab)
        return (pre / pre.sum(dim=-1, keepdim=True)).float().contiguous()
    if dist == "degenerate":
        logits = torch.full((batch, vocab), -30.0)
        # One dominant token per row (varied by row so it is not column 0).
        cols = torch.arange(batch) % vocab
        logits[torch.arange(batch), cols] = 30.0
        return logits.softmax(dim=-1, dtype=torch.float32).contiguous()
    # peaked (default): softmax of scaled normal logits.
    logits = torch.randn(batch, vocab) * scale
    return logits.softmax(dim=-1, dtype=torch.float32).contiguous()


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def _topp_mask(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Boolean [batch, vocab] mask of tokens in the top-p nucleus.

    Mirrors the reference construction in test_sampling.py::test_top_p_sampling.
    """
    eps = 1e-4
    sorted_prob, indices = torch.sort(probs, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros_like(probs, dtype=torch.int32)
    mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    return mask


def _run_one(dist, deterministic, batch, vocab, p, iters, scale, seed):
    torch.manual_seed(seed)
    probs = _build_probs(dist, batch, vocab, scale)
    mask = _topp_mask(probs, p)
    tag = (
        f"dist={dist:<10} deterministic={int(deterministic)} "
        f"batch={batch} vocab={vocab} p={p}"
    )
    print(f"[run] {tag} iters={iters} ...", flush=True)

    for it in range(iters):
        samples = torch.ops.aiter.top_p_sampling_from_probs(
            probs,
            None,
            *_to_tensor_scalar_tuple(p),
            deterministic=deterministic,
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
        in_nucleus = mask[torch.arange(batch), s]
        if not bool((in_nucleus == 1).all()):
            n_bad = int((in_nucleus != 1).sum())
            raise AssertionError(
                f"SILENT WRONG ANSWER at iter {it}: {tag} -> "
                f"{n_bad} samples outside the top-p nucleus"
            )
    print(f"[ok ] {tag}", flush=True)


def main():
    dist = _env("REPRO_DIST", "all")
    det = _env("REPRO_DETERMINISTIC", "1")
    vocab = int(_env("REPRO_VOCAB", "129280"))
    batch = int(_env("REPRO_BATCH", "256"))
    p = float(_env("REPRO_P", "0.9"))
    iters = int(_env("REPRO_ITERS", "1000"))
    seed = int(_env("REPRO_SEED", "0"))
    scale = float(_env("REPRO_LOGIT_SCALE", "8.0"))

    dists = ["peaked", "uniform", "degenerate"] if dist == "all" else [dist]
    if det == "both":
        det_flags = [True, False]
    else:
        det_flags = [bool(int(det))]

    print(
        f"gfx repro: vocab={vocab} batch={batch} p={p} iters={iters} "
        f"seed={seed} logit_scale={scale}",
        flush=True,
    )
    print(f"distributions={dists} deterministic={det_flags}", flush=True)

    for d in dists:
        for df in det_flags:
            _run_one(d, df, batch, vocab, p, iters, scale, seed)

    print("\nALL CONFIGS PASSED (no fault, no wrong answer).", flush=True)


if __name__ == "__main__":
    main()
