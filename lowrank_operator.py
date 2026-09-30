"""Low-rank compensated linear operator with magnitude-based channel filtering.

The operator implements y = x @ W.T on CPU with NumPy.  During inference,
input channels whose absolute value is large are kept and multiplied by the
original weight columns; the remaining (filtered) channels are approximated
by a truncated-SVD low-rank model of W.  Each input channel contributes to
the output exactly once, so the two parts never double-count.

Quantile interpolation rule
---------------------------
The calibration threshold is ``numpy.quantile(|X|, q, method="linear")``
computed over the absolute values of *all* elements of *all* calibration
samples (a single scalar).  With ``n`` sorted values the virtual index is
``i = q * (n - 1)`` and the result is linearly interpolated between the
values at ``floor(i)`` and ``ceil(i)``.  This is NumPy's default method.

Endpoint and equality rules
---------------------------
* ``q == 0`` keeps every channel, ``q == 1`` filters every channel; these
  endpoint rules take priority over the threshold comparison.
* For ``0 < q < 1`` a channel is kept iff ``|x| >= threshold``; channels
  exactly equal to the threshold are therefore kept.
"""

from __future__ import annotations

import numbers
import os
import tempfile
from dataclasses import dataclass

import numpy as np

__all__ = ["LowRankCompensatedLinear", "ForwardResult"]


@dataclass
class ForwardResult:
    """Result of one forward pass.

    Attributes
    ----------
    output:
        Operator output, shape ``(out_features,)`` or ``(batch, out_features)``.
    dense_reference:
        Dense reference ``x @ W.T`` with the same shape as ``output``.
    relative_error:
        ``norm(output - dense_reference) / norm(dense_reference)`` or ``None``
        when the reference norm is zero.
    absolute_error:
        ``norm(output - dense_reference)`` (always defined).
    kept_channels:
        Total number of kept (sample, channel) pairs.
    kept_mask:
        Boolean mask of kept channels, ``(in_features,)`` or
        ``(batch, in_features)``.
    """

    output: np.ndarray
    dense_reference: np.ndarray
    relative_error: float | None
    absolute_error: float
    kept_channels: int
    kept_mask: np.ndarray


def _as_finite_float_array(value, name):
    if isinstance(value, np.ndarray) and np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued, got complex values")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array: {exc}") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values (NaN or inf)")
    return array


def _effective_rank(singular_values, shape):
    if singular_values.size == 0:
        return 0
    tolerance = np.finfo(np.float64).eps * max(shape) * singular_values[0]
    return int(np.count_nonzero(singular_values > tolerance))


class LowRankCompensatedLinear:
    """Linear operator with magnitude filtering and low-rank compensation.

    Parameters
    ----------
    weight:
        2D array of shape ``(out_features, in_features)``.
    quantile:
        Filter quantile in ``[0, 1]``.  ``0`` keeps all channels, ``1``
        filters all channels (endpoint rules take priority).
    target_rank:
        Rank of the truncated-SVD compensation model, an integer in
        ``[0, min(weight.shape)]``.  ``0`` disables compensation (filtered
        channels contribute zero).  If the numerical rank of ``weight`` is
        smaller than ``target_rank``, the compensation keeps only the truly
        non-zero singular components; no fake components are fabricated.
    """

    def __init__(self, weight, quantile, target_rank):
        W = _as_finite_float_array(weight, "weight")
        if W.ndim != 2:
            raise ValueError(
                f"weight must be a 2D matrix, got {W.ndim} dimensions"
            )
        if W.shape[0] < 1 or W.shape[1] < 1:
            raise ValueError(f"weight must be non-empty, got shape {W.shape}")

        if isinstance(quantile, bool) or not isinstance(quantile, numbers.Real):
            raise ValueError(
                f"quantile must be a real number in [0, 1], got {quantile!r}"
            )
        q = float(quantile)
        if not np.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile!r}")

        if isinstance(target_rank, bool) or not isinstance(
            target_rank, numbers.Integral
        ):
            raise ValueError(
                f"target_rank must be an integer, got {target_rank!r}"
            )
        rank = int(target_rank)
        max_rank = min(W.shape)
        if rank < 0 or rank > max_rank:
            raise ValueError(
                f"target_rank must be in [0, {max_rank}] for weight shape "
                f"{W.shape}, got {rank}"
            )

        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        k = min(rank, s.shape[0])
        self._weight = W
        self._U = U[:, :k]
        self._s = s[:k]
        self._Vt = Vt[:k, :]
        self._quantile = q
        self._target_rank = rank
        self._threshold = None
        self._delta = W - (self._U * self._s) @ self._Vt
        self._effective_rank = _effective_rank(self._s, W.shape)

    @property
    def weight(self):
        return self._weight

    @property
    def quantile(self):
        return self._quantile

    @property
    def target_rank(self):
        return self._target_rank

    @property
    def effective_rank(self):
        """Numerical rank of the compensation model (<= target_rank)."""
        return self._effective_rank

    @property
    def in_features(self):
        return self._weight.shape[1]

    @property
    def out_features(self):
        return self._weight.shape[0]

    @property
    def is_calibrated(self):
        return self._threshold is not None

    @property
    def threshold(self):
        """Frozen calibration threshold, or ``None`` before calibration."""
        return self._threshold

    def calibrate(self, samples):
        """Estimate and freeze the threshold from calibration samples.

        ``samples`` must be a 2D array ``(n_samples, in_features)`` with at
        least one row.  The threshold is one scalar: the ``quantile`` of the
        absolute values of all elements of all samples, using linear
        interpolation (see the module docstring).  On any validation error
        the previously calibrated state is left untouched.
        """
        X = _as_finite_float_array(samples, "samples")
        if X.ndim != 2:
            raise ValueError(
                f"samples must be a 2D array (n_samples, in_features), "
                f"got {X.ndim} dimensions"
            )
        if X.shape[0] < 1:
            raise ValueError("calibration set is empty (n_samples = 0)")
        if X.shape[1] != self.in_features:
            raise ValueError(
                f"dimension mismatch: samples have {X.shape[1]} features, "
                f"expected {self.in_features}"
            )
        threshold = float(
            np.quantile(np.abs(X).ravel(), self._quantile, method="linear")
        )
        self._threshold = threshold
        return threshold

    def _kept_mask(self, abs_X):
        if self._quantile == 0.0:
            return np.ones(abs_X.shape, dtype=bool)
        if self._quantile == 1.0:
            return np.zeros(abs_X.shape, dtype=bool)
        return abs_X >= self._threshold

    def forward(self, x):
        """Run the operator on a vector ``(in_features,)`` or a batch
        ``(batch, in_features)`` and return a :class:`ForwardResult`."""
        if self._threshold is None:
            raise RuntimeError(
                "operator is not calibrated; call calibrate(samples) or "
                "load a calibrated operator first"
            )
        X = _as_finite_float_array(x, "x")
        if X.ndim == 1:
            X2d = X[np.newaxis, :]
            squeeze = True
        elif X.ndim == 2:
            X2d = X
            squeeze = False
        else:
            raise ValueError(
                f"x must be a vector or a 2D batch, got {X.ndim} dimensions"
            )
        if X2d.shape[1] != self.in_features:
            raise ValueError(
                f"dimension mismatch: x has {X2d.shape[1]} features, "
                f"expected {self.in_features}"
            )
        if X2d.shape[0] < 1:
            raise ValueError("x is an empty batch (0 samples)")

        mask = self._kept_mask(np.abs(X2d))
        X_kept = X2d * mask
        # Low-rank contribution of every channel ...
        low_rank = (X2d @ (self._Vt.T * self._s)) @ self._U.T
        # ... plus the residual (W - W_lr) for kept channels only, so each
        # channel contributes exactly once: kept -> W, filtered -> W_lr.
        output = low_rank + X_kept @ self._delta.T

        dense = X2d @ self._weight.T
        difference = output - dense
        absolute_error = float(np.linalg.norm(difference))
        reference_norm = float(np.linalg.norm(dense))
        relative_error = (
            absolute_error / reference_norm if reference_norm > 0.0 else None
        )

        if squeeze:
            output = output[0]
            dense = dense[0]
            mask = mask[0]
        return ForwardResult(
            output=output,
            dense_reference=dense,
            relative_error=relative_error,
            absolute_error=absolute_error,
            kept_channels=int(mask.sum()),
            kept_mask=mask,
        )

    __call__ = forward

    def save(self, path):
        """Atomically save the operator (including the frozen threshold).

        The file is written to a temporary name first and then moved over
        ``path``; a failed save never clobbers an existing file.
        """
        directory = os.path.dirname(os.path.abspath(path))
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(
                    handle,
                    weight=self._weight,
                    U=self._U,
                    s=self._s,
                    Vt=self._Vt,
                    quantile=np.float64(self._quantile),
                    target_rank=np.int64(self._target_rank),
                    calibrated=np.bool_(self.is_calibrated),
                    threshold=np.float64(
                        self._threshold if self.is_calibrated else np.nan
                    ),
                )
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path):
        """Load a saved operator.  The stored threshold is restored as-is and
        is never re-estimated; no calibration data is needed."""
        with open(path, "rb") as handle:
            data = np.load(handle)
            keys = set(data.files)
            required = {
                "weight",
                "U",
                "s",
                "Vt",
                "quantile",
                "target_rank",
                "calibrated",
                "threshold",
            }
            missing = required - keys
            if missing:
                raise ValueError(
                    f"file {path!r} is not a valid operator archive; "
                    f"missing arrays: {sorted(missing)}"
                )
            arrays = {name: data[name] for name in required}

        W = _as_finite_float_array(arrays["weight"], "weight")
        U = _as_finite_float_array(arrays["U"], "U")
        s = _as_finite_float_array(arrays["s"], "s")
        Vt = _as_finite_float_array(arrays["Vt"], "Vt")
        if W.ndim != 2 or min(W.shape) < 1:
            raise ValueError(f"invalid weight shape {W.shape} in {path!r}")
        if s.ndim != 1 or U.shape != (W.shape[0], s.shape[0]) or Vt.shape != (
            s.shape[0],
            W.shape[1],
        ):
            raise ValueError(
                f"inconsistent factor shapes in {path!r}: U{U.shape}, "
                f"s{s.shape}, Vt{Vt.shape} for weight {W.shape}"
            )

        quantile = float(arrays["quantile"])
        if not np.isfinite(quantile) or quantile < 0.0 or quantile > 1.0:
            raise ValueError(f"invalid quantile {quantile!r} in {path!r}")
        target_rank = int(arrays["target_rank"])
        if target_rank < 0 or target_rank > min(W.shape):
            raise ValueError(f"invalid target_rank {target_rank} in {path!r}")
        if s.shape[0] != min(target_rank, min(W.shape)):
            raise ValueError(
                f"factor count {s.shape[0]} does not match target_rank "
                f"{target_rank} in {path!r}"
            )
        calibrated = bool(arrays["calibrated"])
        threshold = float(arrays["threshold"])
        if calibrated and not np.isfinite(threshold):
            raise ValueError(f"invalid threshold in {path!r}")

        operator = cls.__new__(cls)
        operator._weight = W
        operator._U = U
        operator._s = s
        operator._Vt = Vt
        operator._quantile = quantile
        operator._target_rank = target_rank
        operator._threshold = threshold if calibrated else None
        operator._delta = W - (U * s) @ Vt
        operator._effective_rank = _effective_rank(s, W.shape)
        return operator
