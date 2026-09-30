"""Core implementation of the low-rank compensated linear operator."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "LowRankLinear",
    "InferenceResult",
    "CalibrationError",
    "DimensionMismatchError",
    "InvalidParameterError",
    "NonFiniteError",
]

ArrayLike = Union[np.ndarray, "list"]

# Relative/absolute tolerances used inside the library.  The operator itself
# is exact arithmetic apart from a single truncated SVD, so the expected
# deviation of a full-rank reconstruction from the dense result is only
# floating-point round-off (roughly ``eps * sigma_1 * n`` for float64).
DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12
# Singular values at or below ``rank_tol * sigma_1`` are treated as exact
# zeros; this guards against fabricated non-zero low-rank components.
DEFAULT_RANK_TOL = 1e-10


class CalibrationError(ValueError):
    """Raised when calibration data is missing or unusable."""


class DimensionMismatchError(ValueError):
    """Raised when an input does not match the operator dimensions."""


class InvalidParameterError(ValueError):
    """Raised for out-of-range parameters (quantile, target rank, ...)."""


class NonFiniteError(ValueError):
    """Raised when data contains NaN or infinite values."""


@dataclass
class InferenceResult:
    """Container for inference outputs and diagnostics.

    Attributes
    ----------
    output:
        Approximate output, shape ``(batch, out_features)`` (batch form) or
        ``(out_features,)`` (single vector form).
    dense_output:
        Dense reference output using the original weight matrix.
    error:
        Per-vector approximation error norm:
        relative error when the reference norm is non-zero, absolute error
        when it is exactly zero.
    relative_error, absolute_error:
        Both norms are always reported for convenience; ``error`` selects
        relative unless the reference norm is exactly zero.
    kept_count:
        Number of input channels kept (using the original weight) for every
        sample.
    threshold:
        Frozen absolute-value threshold used during inference.
    """

    output: np.ndarray
    dense_output: np.ndarray
    error: np.ndarray
    relative_error: np.ndarray
    absolute_error: np.ndarray
    kept_count: np.ndarray
    threshold: float


def _as_finite_2d(array: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise DimensionMismatchError(
            f"{name} must be a 2-D array, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise NonFiniteError(f"{name} contains NaN or infinite values")
    return arr


class LowRankLinear:
    """Linear operator with quantile-gated channels and low-rank compensation.

    Parameters
    ----------
    weight:
        Weight matrix of shape ``(out_features, in_features)`` mapping a row
        vector ``x`` to ``y = x @ weight.T``.
    calibration:
        Calibration samples of shape ``(n_samples, in_features)``. Must be
        non-empty and finite.
    quantile:
        Filtering quantile ``q`` in ``[0, 1]``. One common threshold is taken
        at the ``q``-quantile of the absolute values of *all* calibration
        elements pooled together. Endpoints have priority over interpolation:
        ``q == 0`` keeps every channel, ``q == 1`` filters every channel.
        For ``0 < q < 1`` a channel is kept whenever ``|x_c| >= threshold``,
        so channels exactly equal to the threshold are always retained.
    rank:
        Target rank ``r`` for the truncated-SVD compensation, with
        ``0 <= r <= in_features``. Ranks above the numerical rank of the
        weight are clamped to it (zero singular directions are never filled
        with fabricated components). ``r == 0`` disables compensation.
    rtol, atol:
        Tolerances documented with the reported errors (not used to silently
        perturb any computation).
    rank_tol:
        Singular values ``<= rank_tol * sigma_max`` count as exact zeros.
    """

    def __init__(
        self,
        weight: ArrayLike,
        calibration: ArrayLike,
        quantile: float,
        rank: int,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        rank_tol: float = DEFAULT_RANK_TOL,
    ) -> None:
        w = _as_finite_2d(weight, "weight")
        self._out_features, self._in_features = w.shape

        q = self._check_quantile(quantile)
        r = self._check_rank(rank)
        self._requested_rank = r

        cal = _as_finite_2d(calibration, "calibration")
        if cal.shape[0] == 0:
            raise CalibrationError("calibration sample set must be non-empty")
        if cal.shape[1] != self._in_features:
            raise DimensionMismatchError(
                "calibration feature dimension "
                f"{cal.shape[1]} does not match weight input dimension "
                f"{self._in_features}"
            )

        # Truncated SVD: W_r = U_r diag(s_r) V_r^T. Only genuinely non-zero
        # singular directions are retained (no padded fake components).
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        cutoff = s[0] * rank_tol if s.size and s[0] > 0.0 else 0.0
        nonzero = int(np.count_nonzero(s > cutoff))
        self._numerical_rank = nonzero
        r_eff = min(r, nonzero)
        self._rank = r_eff
        if r_eff == 0:
            self._left = np.zeros((self._out_features, 0), dtype=np.float64)
            self._right = np.zeros((0, self._in_features), dtype=np.float64)
            self._singular_values = s.copy()
        else:
            self._left = u[:, :r_eff] * s[:r_eff]
            self._right = vt[:r_eff, :]
            self._singular_values = s.copy()
        # Sanity identity: factors must reproduce W on the retained subspace.
        self._weight = w.copy()

        # One pooled absolute-value quantile over every calibration element.
        abs_cal = np.abs(cal).reshape(-1)
        self._quantile = q
        if q == 0.0:
            self._threshold = 0.0
        elif q == 1.0:
            self._threshold = np.inf
        else:
            self._threshold = float(np.quantile(abs_cal, q, method="linear"))

        self._rtol = float(rtol)
        self._atol = float(atol)
        self._rank_tol = float(rank_tol)
        self._calibration_stats = (
            int(cal.shape[0]),
            float(abs_cal.min()),
            float(abs_cal.max()),
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def infer(self, x: ArrayLike) -> InferenceResult:
        """Run gated + low-rank compensated inference.

        Accepts a single vector of shape ``(in_features,)`` or a batch of
        shape ``(batch, in_features)``; the output shape mirrors the input.
        """
        arr = np.asarray(x, dtype=np.float64)
        single = False
        if arr.ndim == 1:
            single = True
            batch = arr[None, :]
        elif arr.ndim == 2:
            batch = arr
        else:
            raise DimensionMismatchError(
                f"input must be 1-D or 2-D, got {arr.ndim}-D"
            )
        if batch.shape[1] != self._in_features:
            raise DimensionMismatchError(
                f"input feature dimension {batch.shape[1]} does not match "
                f"operator input dimension {self._in_features}"
            )
        if batch.shape[0] == 0:
            raise DimensionMismatchError("input batch must be non-empty")
        if not np.all(np.isfinite(batch)):
            raise NonFiniteError("input contains NaN or infinite values")

        n = batch.shape[0]
        if self._quantile == 0.0:
            kept = np.ones_like(batch, dtype=bool)
        elif self._quantile == 1.0:
            kept = np.zeros_like(batch, dtype=bool)
        else:
            kept = np.abs(batch) >= self._threshold

        kept_count = kept.sum(axis=1).astype(np.int64)

        # Dense reference with the full original weight.
        dense = batch @ self._weight.T

        # Part 1: original weight restricted to kept columns.
        x_kept = np.where(kept, batch, 0.0)
        y_dense_part = x_kept @ self._weight.T

        # Part 2: low-rank compensation restricted to filtered columns.
        # Masking inside the right factor keeps the two column blocks
        # disjoint, hence no input contribution is ever counted twice.
        x_filtered = np.where(kept, 0.0, batch)
        if self._rank == 0:
            y_low_part = np.zeros_like(dense)
        else:
            y_low_part = (x_filtered @ self._right.T) @ self._left.T

        approx = y_dense_part + y_low_part

        ref_norm = np.linalg.norm(dense, axis=1)
        abs_err = np.linalg.norm(approx - dense, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(
                ref_norm > 0.0, abs_err / ref_norm, np.zeros_like(abs_err)
            )
        # Reference norm exactly zero -> report the absolute error instead.
        reported = np.where(ref_norm > 0.0, rel_err, abs_err)

        if single:
            result = InferenceResult(
                output=approx[0],
                dense_output=dense[0],
                error=reported[0],
                relative_error=rel_err[0],
                absolute_error=abs_err[0],
                kept_count=kept_count[0],
                threshold=float(self._threshold if np.isfinite(self._threshold) else self._threshold),
            )
        else:
            result = InferenceResult(
                output=approx,
                dense_output=dense,
                error=reported,
                relative_error=rel_err,
                absolute_error=abs_err,
                kept_count=kept_count,
                threshold=float(self._threshold),
            )
        return result

    def __call__(self, x: ArrayLike) -> np.ndarray:
        """Return only the approximate output (single vector or batch)."""
        return self.infer(x).output

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_quantile(q: object) -> float:
        qf = float(q)
        if not np.isfinite(qf):
            raise InvalidParameterError(
                f"quantile must be a finite number in [0, 1], got {q!r}"
            )
        if qf < 0.0 or qf > 1.0:
            raise InvalidParameterError(
                f"quantile must lie in [0, 1], got {qf}"
            )
        return qf

    @staticmethod
    def _check_rank(r: object) -> int:
        if isinstance(r, bool) or not isinstance(r, (int, np.integer)):
            raise InvalidParameterError(
                f"rank must be an integer, got {type(r).__name__}"
            )
        ri = int(r)
        if ri < 0:
            raise InvalidParameterError(f"rank must be >= 0, got {ri}")
        return ri

    # ------------------------------------------------------------------ #
    # Recalibration / properties
    # ------------------------------------------------------------------ #
    def recalibrate(
        self,
        calibration: ArrayLike,
        quantile: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> "LowRankLinear":
        """Estimate a fresh threshold (and optionally rank) from new samples.

        Returns a *new* operator; the receiver is left untouched on failure so
        a bad calibration set never destroys a working operator.
        """
        q = self._quantile if quantile is None else self._check_quantile(quantile)
        r = self._requested_rank if rank is None else self._check_rank(rank)
        return LowRankLinear(
            self._weight,
            calibration,
            q,
            r,
            rtol=self._rtol,
            atol=self._atol,
            rank_tol=self._rank_tol,
        )

    @property
    def weight(self) -> np.ndarray:
        return self._weight.copy()

    @property
    def shape(self) -> Tuple[int, int]:
        return self._out_features, self._in_features

    @property
    def quantile(self) -> float:
        return self._quantile

    @property
    def threshold(self) -> float:
        """Frozen calibration threshold (0 for q=0, +inf for q=1)."""
        return self._threshold

    @property
    def rank(self) -> int:
        """Effective rank actually used by the SVD compensation."""
        return self._rank

    @property
    def requested_rank(self) -> int:
        return self._requested_rank

    @property
    def numerical_rank(self) -> int:
        return self._numerical_rank

    @property
    def singular_values(self) -> np.ndarray:
        return self._singular_values.copy()

    @property
    def rtol(self) -> float:
        return self._rtol

    @property
    def atol(self) -> float:
        return self._atol

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save the calibrated operator to ``path`` (``.npz`` archive).

        The write is atomic (temp file + ``os.replace``), so an existing
        operator file is never left half-written or overwritten on failure.
        """
        directory = os.path.dirname(os.path.abspath(path)) or "."
        payload = {
            "version": np.array(1, dtype=np.int64),
            "weight": self._weight,
            "left": self._left,
            "right": self._right,
            "singular_values": self._singular_values,
            "quantile": np.array(self._quantile, dtype=np.float64),
            "threshold": np.array(self._threshold, dtype=np.float64),
            "rank": np.array(self._rank, dtype=np.int64),
            "requested_rank": np.array(self._requested_rank, dtype=np.int64),
            "numerical_rank": np.array(self._numerical_rank, dtype=np.int64),
            "rtol": np.array(self._rtol, dtype=np.float64),
            "atol": np.array(self._atol, dtype=np.float64),
            "rank_tol": np.array(self._rank_tol, dtype=np.float64),
        }
        fd, tmp_path = tempfile.mkstemp(
            prefix=".lowrank-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, **payload)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "LowRankLinear":
        """Load a calibrated operator.

        The frozen threshold is restored exactly; it is never re-estimated
        and no calibration data is required.
        """
        data = np.load(path, allow_pickle=False)
        try:
            if int(data["version"]) != 1:
                raise InvalidParameterError(
                    f"unsupported operator file version {int(data['version'])}"
                )
            obj = cls.__new__(cls)
            w = np.asarray(data["weight"], dtype=np.float64)
            obj._out_features, obj._in_features = w.shape
            obj._weight = w
            obj._left = np.asarray(data["left"], dtype=np.float64)
            obj._right = np.asarray(data["right"], dtype=np.float64)
            obj._singular_values = np.asarray(
                data["singular_values"], dtype=np.float64
            )
            obj._quantile = float(data["quantile"])
            obj._threshold = float(data["threshold"])
            obj._rank = int(data["rank"])
            obj._requested_rank = int(data["requested_rank"])
            obj._numerical_rank = int(data["numerical_rank"])
            obj._rtol = float(data["rtol"])
            obj._atol = float(data["atol"])
            obj._rank_tol = float(data["rank_tol"])
            obj._calibration_stats = None
        finally:
            data.close()
        return obj
