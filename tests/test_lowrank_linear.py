"""Unittest suite for the low-rank compensated linear operator.

Run from the repository root:
    python3 -m unittest discover -s tests -v

Tolerances (float64 throughout):
* full-rank reconstruction vs dense is pure SVD round-off:
  |approx - dense|_inf <= 1e-10 (per element)
* relative errors compared against hand-computed values: atol 1e-10
* save/load numerical consistency: exact (np.savez stores float64 bit patterns)
"""

import os
import tempfile
import unittest

import numpy as np

from lowrank_linear import (
    CalibrationError,
    DimensionMismatchError,
    InvalidParameterError,
    LowRankLinear,
    NonFiniteError,
)

RTOL = 1e-10
ATOL = 1e-10


def make_operator(q=0.5, r=2, seed=0, m=5, n=8, true_rank=4, ncal=6):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((m, true_rank)) @ rng.standard_normal((true_rank, n))
    cal = rng.standard_normal((ncal, n))
    return LowRankLinear(w, cal, quantile=q, rank=r), w, cal


class NormalOperationTests(unittest.TestCase):
    def test_shapes_single_and_batch(self):
        op, w, _ = make_operator()
        x = np.linspace(-1.0, 1.0, w.shape[1])
        res_v = op.infer(x)
        res_b = op.infer(x[None, :])
        self.assertEqual(res_v.output.shape, (w.shape[0],))
        self.assertEqual(res_v.dense_output.shape, (w.shape[0],))
        self.assertEqual(np.ndim(res_v.error), 0)
        self.assertEqual(res_b.output.shape, (1, w.shape[0]))
        np.testing.assert_allclose(res_v.output, res_b.output[0], atol=ATOL)
        np.testing.assert_allclose(res_v.dense_output, x @ w.T, atol=ATOL)

    def test_dense_reference_matches_manual_matmul(self):
        op, w, _ = make_operator(q=0.3, r=1)
        rng = np.random.default_rng(99)
        x = rng.standard_normal((3, w.shape[1]))
        res = op.infer(x)
        np.testing.assert_allclose(res.dense_output, x @ w.T, atol=ATOL)
        self.assertTrue(np.all(res.kept_count >= 0))
        self.assertTrue(np.all(res.kept_count <= w.shape[1]))

    def test_two_parts_are_disjoint_no_double_count(self):
        op, w, _ = make_operator(q=0.6, r=2)
        rng = np.random.default_rng(5)
        x = rng.standard_normal((2, w.shape[1]))
        res = op.infer(x)
        # Kept and filtered channel index sets are complementary masks, so
        # the dense and low-rank addends never share an input contribution.
        kept_mask = np.abs(x) >= op.threshold
        u, sv, vt = np.linalg.svd(w, full_matrices=False)
        wr = (u[:, :2] * sv[:2]) @ vt[:2, :]
        expected = (np.where(kept_mask, x, 0.0) @ w.T
                    + np.where(~kept_mask, x, 0.0) @ wr.T)
        np.testing.assert_allclose(res.output, expected, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(res.output)))

    def test_manual_decomposition(self):
        """Hand-compute gated-dense + SVD compensation and compare."""
        rng = np.random.default_rng(11)
        w = rng.standard_normal((4, 6))
        cal = rng.standard_normal((5, 6))
        q, r = 0.5, 3
        op = LowRankLinear(w, cal, quantile=q, rank=r)
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        wr = (u[:, :r] * s[:r]) @ vt[:r, :]
        x = rng.standard_normal((3, 6))
        kept = np.abs(x) >= op.threshold
        expected = (np.where(kept, x, 0.0) @ w.T
                    + np.where(~kept, x, 0.0) @ wr.T)
        np.testing.assert_allclose(op(x), expected, atol=1e-12)


class ThresholdTests(unittest.TestCase):
    def test_threshold_is_pooled_quantile_linear(self):
        rng = np.random.default_rng(3)
        n = 7
        w = rng.standard_normal((3, n))
        cal = rng.standard_normal((10, n))
        q = 0.63
        op = LowRankLinear(w, cal, quantile=q, rank=2)
        expected = float(np.quantile(np.abs(cal).reshape(-1), q, method="linear"))
        self.assertAlmostEqual(op.threshold, expected, places=14)

    def test_linear_interpolation_explicit(self):
        # Calibration absolute values are exactly [0, 1, 2, 3].
        w = np.eye(4)
        cal = np.array([[0.0, 1.0, 2.0, 3.0]])
        # q=0.5 -> midpoint of 1 and 2 -> 1.5 under linear interpolation.
        op = LowRankLinear(w, cal, quantile=0.5, rank=1)
        self.assertAlmostEqual(op.threshold, 1.5, places=14)
        # q=0.25 -> halfway between 0 and 1 -> 0.75.
        op25 = LowRankLinear(w, cal, quantile=0.25, rank=1)
        self.assertAlmostEqual(op25.threshold, 0.75, places=14)

    def test_threshold_frozen_at_inference(self):
        op, w, cal = make_operator(q=0.5, r=2)
        frozen = op.threshold
        # Extreme inference samples must not move the threshold.
        x = np.full((3, w.shape[1]), 1e6)
        res = op.infer(x)
        self.assertEqual(op.threshold, frozen)
        self.assertTrue(np.all(res.kept_count == w.shape[1]))
        x_small = np.full((2, w.shape[1]), 1e-9)
        res2 = op.infer(x_small)
        self.assertEqual(op.threshold, frozen)
        self.assertTrue(np.all(res2.kept_count == 0))

    def test_tie_at_threshold_is_kept(self):
        # Build a calibration set whose median absolute value is exactly 1.
        w = np.array([[1.0, 2.0, 3.0, 4.0]])
        # Pooled abs values [0.5, 1, 1, 2] -> linear median is exactly 1.
        cal = np.array([[0.5, -1.0, 1.0, 2.0]])
        op = LowRankLinear(w, cal, quantile=0.5, rank=1)
        self.assertAlmostEqual(op.threshold, 1.0, places=14)
        # An input element exactly at the threshold must be retained (>=).
        x = np.array([1.0, -1.0, 0.2, 1.0])
        res = op.infer(x)
        self.assertEqual(int(res.kept_count), 3)

    def test_endpoint_zero_keeps_everything(self):
        op, w, _ = make_operator(q=0.0, r=2)
        self.assertEqual(op.threshold, 0.0)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, w.shape[1]))
        res = op.infer(x)
        self.assertTrue(np.all(res.kept_count == w.shape[1]))
        # Even exact zeros are kept at q=0 (mask is all-True), and output is dense.
        np.testing.assert_allclose(res.output, res.dense_output, atol=ATOL)

    def test_endpoint_one_filters_everything(self):
        op, w, _ = make_operator(q=1.0, r=2)
        self.assertTrue(np.isinf(op.threshold))
        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, w.shape[1]))
        res = op.infer(x)
        self.assertTrue(np.all(res.kept_count == 0))


class RankTests(unittest.TestCase):
    def test_rank_zero_compensation_disabled(self):
        op, w, _ = make_operator(q=0.5, r=0)
        self.assertEqual(op.rank, 0)
        rng = np.random.default_rng(4)
        x = rng.standard_normal((3, w.shape[1]))
        res = op.infer(x)
        kept = np.abs(x) >= op.threshold
        expected = np.where(kept, x, 0.0) @ w.T
        np.testing.assert_array_equal(res.output, expected)

    def test_full_rank_matches_dense_even_when_all_filtered(self):
        rng = np.random.default_rng(8)
        w = rng.standard_normal((6, 5))  # full rank min(6,5)=5
        cal = rng.standard_normal((4, 5))
        op = LowRankLinear(w, cal, quantile=1.0, rank=5)
        x = rng.standard_normal((7, 5))
        res = op.infer(x)
        np.testing.assert_allclose(res.output, res.dense_output, atol=1e-10)
        self.assertTrue(np.all(res.relative_error <= RTOL))

    def test_all_kept_matches_dense_any_rank(self):
        op, w, _ = make_operator(q=0.0, r=1)
        rng = np.random.default_rng(6)
        x = rng.standard_normal((3, w.shape[1]))
        res = op.infer(x)
        np.testing.assert_allclose(res.output, res.dense_output, atol=ATOL)

    def test_rank_deficient_no_fake_components(self):
        rng = np.random.default_rng(2)
        true_rank = 3
        w = (rng.standard_normal((6, true_rank))
             @ rng.standard_normal((true_rank, 8)))
        cal = rng.standard_normal((4, 8))
        # Requesting a rank above the true rank must be clamped, not padded.
        op = LowRankLinear(w, cal, quantile=0.5, rank=8)
        self.assertEqual(op.requested_rank, 8)
        self.assertEqual(op.rank, true_rank)
        self.assertEqual(op.numerical_rank, true_rank)
        # SVD must expose exact (near-zero) null-space singular values.
        sv = op.singular_values
        self.assertTrue(np.all(sv[true_rank:] <= sv[0] * 1e-10 + 1e-12))

    def test_explicit_zero_weight_rank_zero(self):
        w = np.zeros((3, 4))
        cal = np.ones((2, 4))
        op = LowRankLinear(w, cal, quantile=0.5, rank=4)
        self.assertEqual(op.rank, 0)
        self.assertEqual(op.numerical_rank, 0)
        res = op.infer(np.ones((2, 4)))
        np.testing.assert_array_equal(res.output, np.zeros((2, 3)))


class ErrorReportingTests(unittest.TestCase):
    def test_relative_error_when_reference_nonzero(self):
        op, w, _ = make_operator(q=0.5, r=1)
        rng = np.random.default_rng(13)
        x = rng.standard_normal((2, w.shape[1]))
        res = op.infer(x)
        rel = np.linalg.norm(res.dense_output - res.output, axis=1) \
            / np.linalg.norm(res.dense_output, axis=1)
        np.testing.assert_allclose(res.error, rel, atol=ATOL)
        np.testing.assert_allclose(res.relative_error, rel, atol=ATOL)

    def test_absolute_error_when_reference_zero(self):
        # Zero input vector -> dense reference norm is exactly zero.
        op, w, _ = make_operator(q=0.5, r=2)
        x = np.zeros(w.shape[1])
        res = op.infer(x)
        self.assertEqual(np.linalg.norm(res.dense_output), 0.0)
        self.assertAlmostEqual(float(res.error), float(res.absolute_error),
                               places=14)
        self.assertEqual(float(res.relative_error), 0.0)

    def test_abs_error_nonzero_reference_zero_with_rank0(self):
        # With a zero vector both outputs are zero; ensure field consistency.
        op, w, _ = make_operator(q=1.0, r=0)
        res = op.infer(np.zeros(w.shape[1]))
        self.assertEqual(float(res.absolute_error), 0.0)
        self.assertEqual(float(res.error), 0.0)


class SaveLoadTests(unittest.TestCase):
    def setUp(self):
        self.op, self.w, _ = make_operator(q=0.55, r=3)
        self.rng = np.random.default_rng(21)
        self.x = self.rng.standard_normal((4, self.w.shape[1]))
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "op.npz")

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        os.rmdir(self.tmpdir)

    def test_save_load_threshold_frozen_and_outputs_equal(self):
        before = self.op.infer(self.x)
        self.op.save(self.path)
        loaded = LowRankLinear.load(self.path)
        self.assertEqual(loaded.threshold, self.op.threshold)
        self.assertEqual(loaded.quantile, self.op.quantile)
        self.assertEqual(loaded.rank, self.op.rank)
        self.assertEqual(loaded.numerical_rank, self.op.numerical_rank)
        after = loaded.infer(self.x)
        np.testing.assert_array_equal(after.output, before.output)
        np.testing.assert_array_equal(after.kept_count, before.kept_count)

    def test_load_does_not_re_estimate(self):
        # Even with different ambient data, loading requires no calibration
        # and reproduces the identical frozen cut.
        self.op.save(self.path)
        loaded = LowRankLinear.load(self.path)
        self.assertEqual(loaded.threshold, self.op.threshold)

    def test_save_failure_does_not_overwrite_existing(self):
        self.op.save(self.path)
        with open(self.path, "rb") as fh:
            original_bytes = fh.read()
        bad = LowRankLinear(self.w, np.ones((1, self.w.shape[1])), 0.5, 1)
        with self.assertRaises((FileNotFoundError, OSError)):
            bad.save(os.path.join(self.tmpdir, "no_such_dir", "op.npz"))
        with open(self.path, "rb") as fh:
            self.assertEqual(fh.read(), original_bytes)

    def test_recalibrate_returns_new_without_destroying_old(self):
        rng = np.random.default_rng(31)
        new_cal = rng.standard_normal((9, self.w.shape[1])) * 10.0
        old_threshold = self.op.threshold
        new_op = self.op.recalibrate(new_cal, quantile=0.5, rank=2)
        self.assertEqual(self.op.threshold, old_threshold)
        self.assertNotEqual(new_op.threshold, old_threshold)
        self.assertEqual(new_op.rank, 2)

    def test_recalibrate_failure_leaves_operator_intact(self):
        bad_cal = np.zeros((0, self.w.shape[1]))
        with self.assertRaises(CalibrationError):
            self.op.recalibrate(bad_cal)
        self.assertTrue(np.isfinite(self.op.threshold))

    def test_roundtrip_endpoint_thresholds(self):
        for q in (0.0, 1.0):
            op = LowRankLinear(self.w, np.ones((1, self.w.shape[1])), q, 1)
            p = os.path.join(self.tmpdir, f"op_{q}.npz")
            op.save(p)
            loaded = LowRankLinear.load(p)
            self.assertEqual(loaded.quantile, q)
            self.assertEqual(np.isinf(loaded.threshold), q == 1.0)
            os.unlink(p)


class InvalidInputTests(unittest.TestCase):
    def setUp(self):
        self.w = np.eye(4)
        self.cal = np.ones((2, 4))

    def test_empty_calibration(self):
        with self.assertRaises(CalibrationError):
            LowRankLinear(self.w, np.zeros((0, 4)), 0.5, 1)

    def test_calibration_dimension_mismatch(self):
        with self.assertRaises(DimensionMismatchError):
            LowRankLinear(self.w, np.ones((2, 5)), 0.5, 1)

    def test_inference_dimension_mismatch(self):
        op = LowRankLinear(self.w, self.cal, 0.5, 1)
        with self.assertRaises(DimensionMismatchError):
            op.infer(np.ones(3))
        with self.assertRaises(DimensionMismatchError):
            op.infer(np.ones((2, 3)))
        with self.assertRaises(DimensionMismatchError):
            op.infer(np.ones((2, 0, 4)))

    def test_weight_wrong_ndim(self):
        with self.assertRaises(DimensionMismatchError):
            LowRankLinear(np.ones((2, 2, 2)), np.ones((1, 2)), 0.5, 1)

    def test_quantile_out_of_range(self):
        for bad in (-0.01, 1.01, -1.0, 2.0):
            with self.assertRaises(InvalidParameterError):
                LowRankLinear(self.w, self.cal, bad, 1)

    def test_quantile_not_finite(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(InvalidParameterError):
                LowRankLinear(self.w, self.cal, bad, 1)

    def test_rank_negative_or_wrong_type(self):
        with self.assertRaises(InvalidParameterError):
            LowRankLinear(self.w, self.cal, 0.5, -1)
        with self.assertRaises(InvalidParameterError):
            LowRankLinear(self.w, self.cal, 0.5, 1.5)

    def test_non_finite_weight(self):
        with self.assertRaises(NonFiniteError):
            LowRankLinear(np.full((4, 4), np.nan), self.cal, 0.5, 1)
        with self.assertRaises(NonFiniteError):
            LowRankLinear(np.full((4, 4), np.inf), self.cal, 0.5, 1)

    def test_non_finite_calibration(self):
        with self.assertRaises(NonFiniteError):
            LowRankLinear(self.w, np.full((2, 4), np.inf), 0.5, 1)

    def test_non_finite_inference(self):
        op = LowRankLinear(self.w, self.cal, 0.5, 1)
        with self.assertRaises(NonFiniteError):
            op.infer(np.array([1.0, np.nan, 0.0, 0.0]))
        with self.assertRaises(NonFiniteError):
            op.infer(np.array([[1.0, 2.0, 3.0, np.inf]]))

    def test_failed_construction_does_not_register_anything(self):
        # A bad operator must simply not exist; prior valid op is untouched.
        good = LowRankLinear(self.w, self.cal, 0.5, 1)
        before = good.threshold
        with self.assertRaises(CalibrationError):
            LowRankLinear(self.w, np.zeros((0, 4)), 0.5, 1)
        self.assertEqual(good.threshold, before)


if __name__ == "__main__":
    unittest.main()
