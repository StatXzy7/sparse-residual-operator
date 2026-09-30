"""Low-rank compensated linear operator (NumPy only, CPU).

A dense linear map ``y = x @ W.T`` is approximated by:

* keeping the original weight for a subset of *input channels* selected on a
  per-sample basis with a calibration-frozen absolute-value threshold;
* approximating the contribution of the remaining (filtered) channels with a
  truncated-SVD low-rank factorisation of the weight matrix.

The two parts add disjoint column blocks of ``W``, so no input contribution is
ever counted twice.
"""

from .core import (
    LowRankLinear,
    InferenceResult,
    CalibrationError,
    DimensionMismatchError,
    InvalidParameterError,
    NonFiniteError,
)

__all__ = [
    "LowRankLinear",
    "InferenceResult",
    "CalibrationError",
    "DimensionMismatchError",
    "InvalidParameterError",
    "NonFiniteError",
]
