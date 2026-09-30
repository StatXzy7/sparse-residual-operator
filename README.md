# sparse-residual-operator

NumPy-only, CPU/Linux linear-operator library that keeps large-magnitude
**input channels** with the original weight and compensates the filtered
channels with a truncated-SVD **low-rank** approximation. Supports single
vectors and batched matrices, calibration with a frozen threshold, and
save/reload of a calibrated operator.

## Model

For a weight `W` of shape `(out_features, in_features)` and a row input `x`:

```
y_dense = x @ W.T
```

1. **Calibration** pools the absolute values of *every element* of all
   calibration samples and takes one common threshold
   `t = quantile(|calibration|, q)`.
2. **Inference** (threshold never changes after calibration), per sample and
   per input channel `c`:
   * kept channel   (`|x_c| >= t`): use the original weight column `W[:, c]`;
   * filtered channel (`|x_c| <  t`): use the rank-`r` SVD approximation
     `W_r = U_r diag(s_r) V_r^T`.

```
y_hat = (x * kept_mask) @ W.T  +  (x * ~kept_mask) @ W_r.T
```

The two masks are complementary, so the dense and low-rank addends act on
**disjoint column blocks** — no input contribution is counted twice.

### Endpoint and tie rules (endpoint rules take priority)

* `q == 0.0` → every channel is kept (`t = 0`, mask all-True), output equals
  the dense result for any rank.
* `q == 1.0` → every channel is filtered (`t = +inf`), output is pure
  low-rank compensation.
* `0 < q < 1` → a channel with `|x_c|` **equal** to the threshold is kept
  (`>=`).

### Quantile interpolation

`numpy.quantile(values, q, method="linear")` on the pooled absolute values —
the default linear interpolation: after sorting `v_0 <= ... <= v_{N-1}`, the
position `h = q * (N - 1)` is split into `i = floor(h)` and
`f = h - i`, giving `v_i * (1 - f) + v_{i+1} * f` (ties use the averaged
order-statistic positions, as NumPy defines them).

### Rank handling

* `rank = 0` disables compensation: filtered channels contribute exactly zero.
* `rank >= numerical_rank` (up to `in_features`) is clamped to the numerical
  rank. Null-space singular directions are **never** padded with fake non-zero
  components (`sigma_k <= rank_tol * sigma_1`, default `rank_tol = 1e-10`,
  count as exact zeros).
* Full rank (or `q = 0`) reproduces the dense result up to SVD round-off.

## Outputs and errors

`LowRankLinear.infer(x)` returns an `InferenceResult` with:

* `output` — approximate output (1-D output for a 1-D vector input, 2-D for a
  batch);
* `dense_output` — dense reference `x @ W.T`;
* `relative_error`, `absolute_error` — per-sample
  `||y_hat - y_dense||_2 / ||y_dense||_2` and `||y_hat - y_dense||_2`;
* `error` — relative error, or the **absolute** error when the reference norm
  is exactly zero;
* `kept_count` — channels actually kept per sample;
* `threshold` — the frozen threshold.

### Tolerances

All arithmetic is `float64`. The only approximation is the truncated SVD:

* full-rank vs dense: per-element difference expected at round-off level,
  tests assert `<= 1e-10`;
* hand-computed decompositions and interpolation: `atol = 1e-10`
  (often `1e-12`/`1e-14` for exact checks);
* save/load is bit-exact (float64 stored via `np.savez`).

## Persistence

`op.save("path.npz")` / `LowRankLinear.load("path.npz")` stores the weight,
SVD factors, quantile and the **frozen threshold**. Loading never re-estimates
the threshold and requires no calibration data. Saves are atomic
(temp file + `os.replace`), so a failed save never corrupts an existing file.
`op.recalibrate(new_cal, quantile=..., rank=...)` returns a new operator and
leaves the original untouched on failure.

## Errors

`CalibrationError`, `DimensionMismatchError`, `InvalidParameterError`,
`NonFiniteError` (all `ValueError` subclasses) are raised for: empty
calibration set, dimension mismatches, quantile outside `[0, 1]`, negative /
non-integer rank, and any NaN/inf in weight, calibration, or inference input.

## Run

```bash
python3 example.py                     # small runnable matrix demo
python3 -m unittest discover -s tests  # 35 tests
```

Dependency: NumPy only (`numpy 2.x` tested). No models or data are
downloaded.
