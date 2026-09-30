"""Runnable small-matrix demo of LowRankCompensatedLinear (CPU, NumPy only)."""

import os
import tempfile

import numpy as np

from lowrank_operator import LowRankCompensatedLinear


def main():
    rng = np.random.default_rng(0)

    out_features, in_features = 4, 6
    weight = rng.normal(size=(out_features, in_features))
    samples = rng.normal(size=(8, in_features))  # calibration set
    quantile, target_rank = 0.6, 2

    op = LowRankCompensatedLinear(weight, quantile=quantile, target_rank=target_rank)
    threshold = op.calibrate(samples)
    print(f"frozen threshold (quantile={quantile}, linear interpolation): {threshold:.6f}")
    print(f"target rank: {op.target_rank}, effective rank: {op.effective_rank}")

    # Single-vector inference.
    x = rng.normal(size=in_features)
    result = op.forward(x)
    print("\n[single vector]")
    print("output         :", np.round(result.output, 6))
    print("dense reference:", np.round(result.dense_reference, 6))
    print(f"relative error : {result.relative_error:.3e}")
    print(f"kept channels  : {result.kept_channels}/{in_features} "
          f"(mask {result.kept_mask.astype(int)})")

    # Batch inference.
    batch = rng.normal(size=(3, in_features))
    batch_result = op.forward(batch)
    print("\n[batch of 3]")
    print(f"relative error : {batch_result.relative_error:.3e}")
    print(f"kept channels  : {batch_result.kept_channels}/{3 * in_features} "
          f"(per-sample {batch_result.kept_mask.sum(axis=1)})")

    # Save / reload: the frozen threshold is restored, never re-estimated.
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "operator.npz")
        op.save(path)
        loaded = LowRankCompensatedLinear.load(path)
        reloaded = loaded.forward(x)
        print("\n[save / load]")
        print(f"threshold after load: {loaded.threshold:.6f} (no recalibration)")
        print(f"outputs identical   : {np.array_equal(result.output, reloaded.output)}")


if __name__ == "__main__":
    main()
