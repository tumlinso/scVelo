"""CPU-only checks for probe export, gate application, and fixture recipes."""
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from scipy import sparse

import harness


class HarnessTest(unittest.TestCase):
    def test_scvelo_recipe_exports_shared_csr_and_dense_fields(self):
        root = Path(__file__).resolve().parents[2]
        fixture = root / harness.SCVELO_DATASETS["scvelo-pancreas-100"]
        data, W, left_sp, right_sp, left, right, expected, recipe = harness.prepare_scvelo(root, fixture)
        self.assertEqual(W.dtype, np.float32)
        self.assertEqual(left.dtype, np.float32)
        np.testing.assert_allclose(expected["mean_left"], W @ left, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(expected["mean_right"], W @ right, rtol=1e-5, atol=1e-6)
        self.assertLessEqual(recipe["adjusted_reference_max_abs_error"]["second_left"], 1e-6)
        self.assertLessEqual(recipe["adjusted_reference_max_abs_error"]["cross_second"], 1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = {"fixture_sha256": harness.sha256(fixture)}
            harness.write_inputs(out, W, left_sp, right_sp, left, right, manifest)
            cols, rows = W.shape[1], W.shape[0]
            exported_left = np.fromfile(out / "left.f32", dtype="<f4").reshape(cols, left.shape[1])
            np.testing.assert_array_equal(exported_left, left)
            with np.load(out / "reference_sparse.npz") as saved:
                saved_w = sparse.csr_matrix((saved["w_data"], saved["w_indices"], saved["w_indptr"]),
                                            shape=tuple(saved["w_shape"]))
                np.testing.assert_array_equal(saved_w.toarray(), W.toarray())
            self.assertEqual(manifest["shape"], {"sources": 100, "destinations": rows,
                                                   "features": 200, "edges": W.nnz})

    def test_dentate_fixture_recomputes_current_upstream_moments(self):
        root = Path(__file__).resolve().parents[2]
        fixture = root / harness.SCVELO_DATASETS["scvelo-dentategyrus-100"]
        _, W, _, _, left, right, expected, recipe = harness.prepare_scvelo(root, fixture)
        self.assertEqual(left.shape, (100, 200))
        self.assertEqual(right.shape, (100, 200))
        self.assertEqual(W.dtype, np.float32)
        self.assertLessEqual(recipe["adjusted_reference_max_abs_error"]["second_left"], 1e-6)
        self.assertLessEqual(recipe["adjusted_reference_max_abs_error"]["cross_second"], 1e-6)
        self.assertTrue(np.isfinite(expected["second_right"]).all())

    def test_deterministic_model_residual_is_read_from_fitted_instance(self):
        root = Path(__file__).resolve().parents[2]
        fixture = root / harness.SCVELO_DATASETS["scvelo-pancreas-100"]
        data, _, _, _, _, _, expected, _ = harness.prepare_scvelo(root, fixture)
        result = harness.downstream(data, expected, expected)
        self.assertTrue(result["all_pass"], result)
        self.assertTrue(result["fields"]["residual"]["pass"])

    def test_cellrank_recipe_uses_exact_helper_and_preserves_fp64_velocity(self):
        root = Path("/home/tumlinson/accelerated-ports/CellRank")
        fixture = root / harness.CELLRANK_DATASET
        data, W, _, _, left, right, expected, recipe = harness.prepare_cellrank(root, fixture)
        self.assertEqual(W.dtype, np.float32)
        self.assertEqual(left.dtype, np.float32)
        self.assertEqual(right.dtype, np.float32)
        self.assertEqual(recipe["velocity_dtype_preserved"], "float64")
        self.assertTrue(recipe["fp64_velocity_native_followup"])
        self.assertEqual(data.layers["velocity"].dtype, np.float64)
        helper, _ = harness.import_cellrank_moments(root)
        mean, variance = helper._knn_moments(W, left)
        np.testing.assert_array_equal(expected["mean_left"], mean)
        np.testing.assert_array_equal(expected["variance_left"], variance)
        self.assertEqual(expected["second_left"].shape, left.shape)

    def test_centered_gate_uses_expression_scale_near_zero(self):
        expected = {
            "mean_left": np.array([[1.0]], np.float32),
            "mean_right": np.array([[2.0]], np.float32),
            "second_left": np.array([[1.00001]], np.float32),
            "cross_second": np.array([[2.00001]], np.float32),
            "second_right": np.array([[4.00001]], np.float32),
        }
        expected["variance_left"] = expected["second_left"] - expected["mean_left"] ** 2
        expected["variance_right"] = expected["second_right"] - expected["mean_right"] ** 2
        expected["covariance"] = expected["cross_second"] - expected["mean_left"] * expected["mean_right"]
        expected["affine_second_left"] = 2 * expected["second_left"] - expected["mean_left"]
        expected["affine_cross_second"] = 2 * expected["cross_second"] - expected["mean_right"]
        actual = {key: value.copy() for key, value in expected.items()}
        actual["variance_left"] += np.float32(1e-6)
        self.assertTrue(harness.compare(actual, expected, "analytic")["all_pass"])
        actual["variance_left"] += np.float32(1e-3)
        self.assertFalse(harness.compare(actual, expected, "analytic")["all_pass"])

    def test_bad_output_shape_is_rejected_before_reshape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mean_left.f32"
            path.write_bytes(b"\0\0\0\0")
            with self.assertRaisesRegex(RuntimeError, "Invalid mean_left output size"):
                harness.read_outputs(Path(tmp), (2, 3))

    def test_streamed_output_comparison_checks_each_field(self):
        shape = (2, 3)
        expected = {key: np.zeros(shape, np.float32) for key in harness.FIELDS}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for key in harness.FIELDS:
                expected[key].tofile(root / f"{key}.f32")
            result = harness.compare_native_files(root, expected, shape, "streamed")
            self.assertTrue(result["all_pass"])
            np.ones(shape, np.float32).tofile(root / "mean_left.f32")
            result = harness.compare_native_files(root, expected, shape, "streamed")
            self.assertFalse(result["all_pass"])
            self.assertFalse(result["fields"]["mean_left"]["pass"])

    def test_native_metrics_nested_schema_and_counter_failures(self):
        shape = (4, 3, 2, 6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, size in (("row_offsets.u32", 5 * 4), ("column_indices.u32", 6 * 4),
                               ("weights.f32", 6 * 4), ("left.f32", 4 * 2 * 4),
                               ("right.f32", 4 * 2 * 4)):
                (root / name).write_bytes(bytes(size))
            metrics = {
                "schema_version": 1,
                "dimensions": dict(zip(("sources", "destinations", "features", "edges"), shape)),
                "runs": {"warmups": 5, "repeats": 20},
                "native_policy": {"relation": "FP32", "input": "FP32", "multiply": "FP32",
                                  "accumulation": "FP32", "output": "FP32", "tensor_cores": False},
                "gpu": {"name": "test"}, "resident_ms": [1.0] * 20,
                "resident_wall_ms": [2.0] * 20, "preparation_ms": 1.0,
                "allocation_ms": 2.0, "input_read_ms": 3.0, "h2d_ms": 4.0,
                "d2h_ms": 5.0, "output_write_ms": 6.0, "process_total_ms": 100.0,
                "memory_bytes": {"tracked_device_allocations": 9, "host_input_files": 132,
                                 "host_output_files": 240, "native_process_peak_rss": 400},
                "preparation_report": {"topology_preparations": 1, "value_refreshes": 1,
                                       "accepted_forward_launches": 130},
            }
            args = SimpleNamespace(warmups=5, repeats=20)
            valid = harness.validate_native_metrics(metrics, args, root, shape)
            self.assertEqual(valid["memory_bytes"]["native_process_peak_rss_bytes"], 400)
            bad_counter = {**metrics, "preparation_report": {**metrics["preparation_report"],
                                                               "accepted_forward_launches": 129}}
            with self.assertRaisesRegex(RuntimeError, "accepted forward launches"):
                harness.validate_native_metrics(bad_counter, args, root, shape)

    def test_provenance_records_scoped_source_and_binary_digests(self):
        scvelo_root = Path(__file__).resolve().parents[2]
        cellrank_root = Path("/home/tumlinson/accelerated-ports/CellRank")
        cellerator_root = Path("/home/tumlinson/Cellerator")
        with tempfile.TemporaryDirectory() as tmp:
            native = Path(tmp) / "native-exe"
            native.write_bytes(b"test binary digest")
            build_info = Path(tmp) / "build-info.json"
            build_info.write_text('{"schema_version": 1, "toolkit": "fixture"}')
            args = SimpleNamespace(scvelo_root=scvelo_root, cellrank_root=cellrank_root,
                                   cellerator_root=cellerator_root, native=str(native),
                                   build_info=build_info)
            record = harness.provenance(args)
            self.assertEqual(record["native_executable"]["sha256"], harness.sha256(native))
            self.assertIn("source_content_sha256", record["repositories"]["cellerator"])
            self.assertIn("probes/native_moments/harness.py",
                          record["repositories"]["scvelo"]["scoped_file_sha256"])
            self.assertIsNotNone(record["build_info"])

    def test_downstream_failure_invalidates_overall_acceptance(self):
        passing = {"correctness": {"all_pass": True}, "downstream": {"all_pass": True}}
        self.assertTrue(harness.acceptance_passes({"errors": [], "datasets": {"x": passing}}))
        failing = {"correctness": {"all_pass": True}, "downstream": {"all_pass": False}}
        self.assertFalse(harness.acceptance_passes({"errors": [], "datasets": {"x": failing}}))

    def test_nonfinite_pattern_requires_infinity_sign_to_match(self):
        expected = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
        same = np.array([np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
        flipped = np.array([np.nan, -np.inf, np.inf, 2.0], dtype=np.float32)
        self.assertTrue(harness.same_nonfinite_pattern(same, expected))
        self.assertFalse(harness.same_nonfinite_pattern(flipped, expected))

    def test_synthetic_graph_has_independent_unique_nonself_rows_and_is_seeded(self):
        def generate(seed):
            rng = np.random.default_rng(seed)
            return harness.independent_csr_relation(rng, n=23, degree=5)

        first = generate(271828)
        second = generate(271828)
        np.testing.assert_array_equal(first.indptr, second.indptr)
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_array_equal(first.data, second.data)
        self.assertEqual(np.diff(first.indptr).tolist(), [5] * 23)
        patterns = []
        for row in range(first.shape[0]):
            cols = first.indices[first.indptr[row]:first.indptr[row + 1]]
            self.assertNotIn(row, cols)
            self.assertEqual(np.unique(cols).size, 5)
            patterns.append(tuple(sorted((cols - row) % first.shape[0])))
        # A shared offset pattern translated around rows would have one unique
        # relative-neighbor tuple; independent row sampling produces many.
        self.assertGreater(len(set(patterns)), 1)

    def test_fixed_acceptance_does_not_allow_clipping(self):
        self.assertFalse(harness.GATES["centered_and_adjusted"]["clip_results"])
        self.assertEqual(harness.GATES["raw_moments"], {"rtol": 1e-5, "atol": 1e-6})


if __name__ == "__main__":
    unittest.main()
