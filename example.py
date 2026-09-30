"""Runnable small-matrix demonstration of the low-rank linear operator.

Run:  python3 example.py
Only NumPy is required; nothing is downloaded.
"""

import numpy as np

from lowrank_linear import LowRankLinear


def main() -> None:
    rng = np.random.default_rng(7)

    # Weight of shape (out_features=5, in_features=8), rank 4.
    base = rng.standard_normal((5, 4))
    mix = rng.standard_normal((4, 8))
    weight = base @ mix

    # Calibration samples: (n_samples=6, in_features=8).
    calibration = rng.standard_normal((6, 8))

    q, r = 0.5, 2
    op = LowRankLinear(weight, calibration, quantile=q, rank=r)

    print("== Calibration ==")
    print(f"weight shape      : {op.shape} (out, in)")
    print(f"numerical rank    : {op.numerical_rank}")
    print(f"requested rank    : {op.requested_rank}, effective rank: {op.rank}")
    print(f"quantile          : {q}")
    print(f"frozen |x| cut    : {op.threshold:.6f}  "
          "(np.quantile with linear interpolation; >= cut is kept)")

    # ---- Single vector inference ----
    x = rng.standard_normal(8)
    res = op.infer(x)
    print("\n== Single vector ==")
    print(f"channels kept     : {res.kept_count} / 8")
    print(f"dense reference   : {np.round(res.dense_output, 4)}")
    print(f"approximate output: {np.round(res.output, 4)}")
    print(f"relative error    : {res.relative_error:.3e}")
    print(f"absolute error    : {res.absolute_error:.3e}")

    # ---- Batch inference ----
    X = rng.standard_normal((4, 8))
    batch = op.infer(X)
    print("\n== Batch (4 samples) ==")
    print("kept per sample   :", batch.kept_count.tolist())
    print("relative errors   :", np.round(batch.relative_error, 4).tolist())

    # ---- Endpoint quantiles ----
    all_keep = LowRankLinear(weight, calibration, quantile=0.0, rank=r)
    all_drop = LowRankLinear(weight, calibration, quantile=1.0, rank=r)
    r0 = all_drop.infer(X)
    print("\n== Endpoints ==")
    print("q=0 keeps         :", all_keep.infer(X).kept_count.tolist())
    print("q=1 kept counts   :", r0.kept_count.tolist(),
          " approximation = rank-2 compensation only")

    # ---- Rank 0: compensation is exactly zero ----
    zero_rank = LowRankLinear(weight, calibration, quantile=0.75, rank=0)
    zr = zero_rank.infer(X)
    kept_dense = np.where(np.abs(X) >= zero_rank.threshold, X, 0.0) @ weight.T
    assert np.array_equal(zr.output, kept_dense)
    print("rank=0 output equals dense(kept columns only):",
          np.allclose(zr.output, kept_dense))

    # ---- Full rank: approximation == dense even at q=1 ----
    full = LowRankLinear(weight, calibration, quantile=1.0, rank=8)
    fr = full.infer(X)
    print(f"full-rank + q=1 vs dense max abs diff: "
          f"{np.max(np.abs(fr.output - fr.dense_output)):.3e}")

    # ---- Save / reload: threshold stays frozen ----
    path = "/tmp/lowrank_op.npz"
    op.save(path)
    restored = LowRankLinear.load(path)
    assert restored.threshold == op.threshold
    assert restored.rank == op.rank
    print(f"\nSaved to {path}; reloaded threshold={restored.threshold:.6f} "
          "(identical, no re-estimation)")


if __name__ == "__main__":
    main()
