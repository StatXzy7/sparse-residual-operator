"""Unittest suite for LowRankCompensatedLinear.

All computations use float64.  Algebraic identities (exact same operation
order) are asserted with np.testing.assert_allclose at rtol=1e-12,
atol=1e-12; identities that go through the SVD reconstruction use
rtol=1e-10, atol=1e-10.
"""

import os
import tempfile
import unittest

import numpy as np

from lowrank_operator import LowRankCompensatedLinear

ALGEBRA_RTOL = 1e-12
ALGEBRA_ATOL = 1e-12
SVD_RTOL = 1e-10
SVD_ATOL = 1e-10


def make_operator(quantile=0.5, target_rank=2, seed=1, shape=(4, 6)):
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=shape)
    op = LowRankCompensatedLinear(weight, quantile=quantile, target_rank=target_rank)
    samples = rng.normal(size=(10, shape[1]))
    op.calibrate(samples)
    return op, weight, samples


def low_rank_matrix(weight, rank):
    U, s, Vt = np.linalg.svd(weight, full_matrices=False)
    return (U[:, :rank] * s[:rank]) @ Vt[:rank, :]


class TestCombinedOutput(unittest.TestCase):
    def test_single_vector_matches_manual_combination(self):
        op, weight, _ = make_operator()
        rng = np.random.default_rng(7)
        x = rng.normal(size=op.in_features)
        result = op.forward(x)

        mask = np.abs(x) >= op.threshold
        W_lr = low_rank_matrix(weight, op.target_rank)
        expected = weight[:, mask] @ x[mask] + W_lr[:, ~mask] @ x[~mask]
        np.testing.assert_allclose(
            result.output, expected, rtol=ALGEBRA_RTOL, atol=ALGEBRA_ATOL
        )
        np.testing.assert_allclose(
            result.dense_reference, weight @ x, rtol=ALGEBRA_RTOL, atol=ALGEBRA_ATOL
        )
        self.assertEqual(result.kept_channels, int(mask.sum()))
        np.testing.assert_array_equal(result.kept_mask, mask)

    def test_no_double_counting_per_channel(self):
        # Each channel contributes exactly once: kept via W, filtered via W_lr.
        op, weight, _ = make_operator()
        rng = np.random.default_rng(11)
        x = rng.normal(size=op.in_features)
        result = op.forward(x)
        mask = result.kept_mask
        W_lr = low_rank_matrix(weight, op.target_rank)
        # Sum of the two disjoint channel groups must equal the output.
        kept_part = weight @ (x * mask)
        filtered_part = W_lr @ (x * ~mask)
        np.testing.assert_allclose(
            kept_part + filtered_part,
            result.output,
            rtol=ALGEBRA_RTOL,
            atol=ALGEBRA_ATOL,
        )
        self.assertTrue(np.any(mask) and np.any(~mask))

    def test_batch_matches_rowwise_single(self):
        op, _, _ = make_operator()
        rng = np.random.default_rng(3)
        batch = rng.normal(size=(5, op.in_features))
        batch_result = op.forward(batch)
        self.assertEqual(batch_result.output.shape, (5, op.out_features))
        for row, out_row, mask_row in zip(
            batch, batch_result.output, batch_result.kept_mask
        ):
            single = op.forward(row)
            np.testing.assert_allclose(
                out_row, single.output, rtol=ALGEBRA_RTOL, atol=ALGEBRA_ATOL
            )
            np.testing.assert_array_equal(mask_row, single.kept_mask)
        self.assertEqual(
            batch_result.kept_channels, int(batch_result.kept_mask.sum())
        )


class TestThresholdFreezing(unittest.TestCase):
    def test_threshold_is_frozen_across_inputs(self):
        op, _, samples = make_operator()
        frozen = op.threshold
        expected = float(np.quantile(np.abs(samples), 0.5, method="linear"))
        self.assertAlmostEqual(frozen, expected, places=15)
        rng = np.random.default_rng(5)
        for _ in range(4):
            op.forward(rng.normal(size=(3, op.in_features)) * 100.0)
            self.assertEqual(op.threshold, frozen)

    def test_equality_with_threshold_is_kept(self):
        # |values| = [0, 1, 2, 3]; q = 0.5 -> linear interpolation gives 1.5.
        weight = np.arange(8, dtype=np.float64).reshape(2, 4) + 1.0
        op = LowRankCompensatedLinear(weight, quantile=0.5, target_rank=1)
        op.calibrate(np.array([[0.0, 1.0, 2.0, 3.0]]))
        self.assertAlmostEqual(op.threshold, 1.5, places=15)
        x = np.array([1.5, -1.5, 1.49, 0.0])
        result = op.forward(x)
        np.testing.assert_array_equal(
            result.kept_mask, np.array([True, True, False, False])
        )

    def test_failed_calibrate_preserves_previous_state(self):
        op, _, _ = make_operator()
        frozen = op.threshold
        with self.assertRaises(ValueError):
            op.calibrate(np.empty((0, op.in_features)))
        with self.assertRaises(ValueError):
            op.calibrate(np.full((2, op.in_features), np.nan))
        self.assertEqual(op.threshold, frozen)
        self.assertTrue(op.is_calibrated)


class TestEndpointsAndRanks(unittest.TestCase):
    def test_quantile_zero_keeps_everything(self):
        op, weight, _ = make_operator(quantile=0.0, target_rank=1)
        rng = np.random.default_rng(13)
        x = rng.normal(size=op.in_features) * 1e-6  # tiny values still kept
        result = op.forward(x)
        self.assertEqual(result.kept_channels, op.in_features)
        np.testing.assert_allclose(
            result.output, weight @ x, rtol=ALGEBRA_RTOL, atol=ALGEBRA_ATOL
        )
        self.assertLess(result.absolute_error, 1e-12)

    def test_quantile_one_filters_everything(self):
        op, weight, _ = make_operator(quantile=1.0, target_rank=2)
        rng = np.random.default_rng(17)
        x = rng.normal(size=op.in_features) * 1e6  # huge values still filtered
        result = op.forward(x)
        self.assertEqual(result.kept_channels, 0)
        W_lr = low_rank_matrix(weight, 2)
        np.testing.assert_allclose(
            result.output, W_lr @ x, rtol=ALGEBRA_RTOL, atol=ALGEBRA_ATOL
        )

    def test_zero_rank_disables_compensation(self):
        op, weight, _ = make_operator(quantile=0.5, target_rank=0)
        self.assertEqual(op.effective_rank, 0)
        rng = np.random.default_rng(19)
        x = rng.normal(size=op.in_features)
        result = op.forward(x)
        mask = result.kept_mask
        np.testing.assert_allclose(
            result.output,
            weight[:, mask] @ x[mask],
            rtol=ALGEBRA_RTOL,
            atol=ALGEBRA_ATOL,
        )

    def test_full_rank_matches_dense(self):
        op, weight, _ = make_operator(quantile=0.7, target_rank=4)
        rng = np.random.default_rng(23)
        batch = rng.normal(size=(6, op.in_features))
        result = op.forward(batch)
        np.testing.assert_allclose(
            result.output,
            batch @ weight.T,
            rtol=SVD_RTOL,
            atol=SVD_ATOL,
        )
        self.assertLess(result.relative_error, 1e-10)

    def test_rank_deficient_weight_allows_larger_target_rank(self):
        rng = np.random.default_rng(29)
        a = rng.normal(size=(4, 1))
        b = rng.normal(size=(1, 5))
        weight = a @ b  # exact rank 1
        op = LowRankCompensatedLinear(weight, quantile=0.5, target_rank=3)
        self.assertEqual(op.target_rank, 3)
        # No fabricated components: only 1 truly non-zero singular direction.
        self.assertEqual(op.effective_rank, 1)
        op.calibrate(rng.normal(size=(4, 5)))
        x = rng.normal(size=5)
        result = op.forward(x)
        # Rank-1 weight is reproduced exactly by the compensation model.
        np.testing.assert_allclose(
            result.output, weight @ x, rtol=SVD_RTOL, atol=SVD_ATOL
        )
        self.assertLess(result.relative_error, 1e-10)


class TestErrorMetrics(unittest.TestCase):
    def test_relative_error_definition(self):
        op, weight, _ = make_operator()
        rng = np.random.default_rng(31)
        x = rng.normal(size=op.in_features)
        result = op.forward(x)
        dense = weight @ x
        expected = np.linalg.norm(result.output - dense) / np.linalg.norm(dense)
        self.assertAlmostEqual(result.relative_error, expected, places=15)

    def test_zero_reference_norm_reports_absolute_error(self):
        weight = np.zeros((3, 4))
        op = LowRankCompensatedLinear(weight, quantile=0.5, target_rank=2)
        op.calibrate(np.ones((2, 4)))
        result = op.forward(np.ones(4))
        self.assertIsNone(result.relative_error)
        self.assertEqual(result.absolute_error, 0.0)
        np.testing.assert_array_equal(result.dense_reference, np.zeros(3))


class TestSaveLoad(unittest.TestCase):
    def test_save_load_roundtrip(self):
        op, _, _ = make_operator()
        rng = np.random.default_rng(37)
        x = rng.normal(size=op.in_features)
        batch = rng.normal(size=(4, op.in_features))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "op.npz")
            op.save(path)
            loaded = LowRankCompensatedLinear.load(path)
            self.assertTrue(loaded.is_calibrated)
            self.assertEqual(loaded.threshold, op.threshold)  # not re-estimated
            self.assertEqual(loaded.quantile, op.quantile)
            self.assertEqual(loaded.target_rank, op.target_rank)
            self.assertEqual(loaded.effective_rank, op.effective_rank)
            np.testing.assert_array_equal(
                loaded.forward(x).output, op.forward(x).output
            )
            np.testing.assert_array_equal(
                loaded.forward(batch).output, op.forward(batch).output
            )

    def test_loaded_operator_uses_frozen_threshold(self):
        op, _, _ = make_operator()
        rng = np.random.default_rng(41)
        x = rng.normal(size=op.in_features)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "op.npz")
            op.save(path)
            loaded = LowRankCompensatedLinear.load(path)
            # Inference must use the stored threshold, not a recomputed one.
            expected_mask = np.abs(x) >= op.threshold
            np.testing.assert_array_equal(loaded.forward(x).kept_mask, expected_mask)

    def test_failed_save_does_not_clobber_existing_file(self):
        op, _, _ = make_operator()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "op.npz")
            op.save(path)
            with self.assertRaises(OSError):
                op.save(os.path.join(directory, "missing-dir", "op.npz"))
            loaded = LowRankCompensatedLinear.load(path)  # original intact
            self.assertEqual(loaded.threshold, op.threshold)

    def test_load_rejects_corrupt_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.npz")
            np.savez(path, weight=np.eye(2))
            with self.assertRaises(ValueError):
                LowRankCompensatedLinear.load(path)
            with self.assertRaises(FileNotFoundError):
                LowRankCompensatedLinear.load(os.path.join(directory, "nope.npz"))


class TestInvalidInputs(unittest.TestCase):
    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            LowRankCompensatedLinear(np.ones(3), quantile=0.5, target_rank=1)
        with self.assertRaises(ValueError):
            LowRankCompensatedLinear(np.ones((0, 3)), quantile=0.5, target_rank=1)
        with self.assertRaises(ValueError):
            LowRankCompensatedLinear(
                np.array([[np.nan, 1.0], [0.0, 1.0]]), quantile=0.5, target_rank=1
            )
        with self.assertRaises(ValueError):
            LowRankCompensatedLinear(
                np.array([[np.inf, 1.0], [0.0, 1.0]]), quantile=0.5, target_rank=1
            )
        for bad_quantile in (-0.1, 1.1, np.nan, np.inf, "0.5"):
            with self.assertRaises(ValueError, msg=f"quantile={bad_quantile}"):
                LowRankCompensatedLinear(np.ones((2, 2)), bad_quantile, 1)
        for bad_rank in (-1, 3, 1.5, "2"):
            with self.assertRaises(ValueError, msg=f"rank={bad_rank}"):
                LowRankCompensatedLinear(np.ones((2, 2)), 0.5, bad_rank)

    def test_calibrate_validation(self):
        op = LowRankCompensatedLinear(np.ones((3, 4)), quantile=0.5, target_rank=1)
        with self.assertRaises(ValueError):
            op.calibrate(np.empty((0, 4)))  # empty calibration set
        with self.assertRaises(ValueError):
            op.calibrate(np.ones((2, 5)))  # dimension mismatch
        with self.assertRaises(ValueError):
            op.calibrate(np.ones(4))  # not 2D
        with self.assertRaises(ValueError):
            op.calibrate(np.full((2, 4), np.inf))  # non-finite
        self.assertFalse(op.is_calibrated)

    def test_forward_validation(self):
        op, _, _ = make_operator()
        with self.assertRaises(ValueError):
            op.forward(np.ones(op.in_features + 1))  # dimension mismatch
        with self.assertRaises(ValueError):
            op.forward(np.full(op.in_features, np.nan))  # non-finite
        with self.assertRaises(ValueError):
            op.forward(np.empty((0, op.in_features)))  # empty batch
        with self.assertRaises(ValueError):
            op.forward(np.ones((2, 2, 2)))  # 3D input

    def test_forward_before_calibrate_raises(self):
        op = LowRankCompensatedLinear(np.ones((2, 3)), quantile=0.5, target_rank=1)
        with self.assertRaises(RuntimeError):
            op.forward(np.ones(3))


if __name__ == "__main__":
    unittest.main()
