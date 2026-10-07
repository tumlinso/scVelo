"""CPU-only contract tests for the resident scientific consumer harness."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
from scipy import sparse

import harness
import resident_moments as resident


class FakeBuffer:
    def __init__(self, shape):
        self.value = np.empty(shape, dtype=np.float32)

    @property
    def nbytes(self):
        return self.value.nbytes


class FakeRelation:
    generation = 1
    prepared_bytes = 0

    def __init__(self, dense):
        self.dense = dense


class FakeEvent:
    def __init__(self):
        self.time = time.perf_counter()


class FakeBackend:
    name = "fake-cpu"

    def __init__(self, *_args):
        self.package = SimpleNamespace(__file__=__file__, __version__="test")
        self.loaded_extensions = []
        self.device = 0
        self.buffers = []
        self.download_calls = []
        self.prepare_calls = 0
        self.apply_calls = 0
        self.relation = None

    def allocate(self, shape):
        result = FakeBuffer(shape)
        self.buffers.append(result)
        return result

    @staticmethod
    def nbytes(value):
        return value.nbytes

    @staticmethod
    def upload(buffer, array):
        buffer.value[...] = array

    def create_relation(self, indptr, indices, weights, source_count, feature_width):
        self.prepare_calls += 1
        destinations = len(indptr) - 1
        dense = np.zeros((destinations, source_count), dtype=np.float32)
        for row in range(destinations):
            for edge in range(int(indptr[row]), int(indptr[row + 1])):
                dense[row, int(indices[edge])] = weights.value[edge]
        self.relation = FakeRelation(dense)
        return self.relation

    def apply(self, relation, value, output):
        self.apply_calls += 1
        output.value[...] = relation.dense @ value.value

    @staticmethod
    def multiply(a, b, out):
        np.multiply(a.value, b.value, out=out.value)

    @staticmethod
    def axpby(alpha, a, beta, b, out):
        np.multiply(a.value, np.float32(alpha), out=out.value)
        out.value[...] += np.float32(beta) * b.value

    @staticmethod
    def event():
        return FakeEvent()

    @staticmethod
    def elapsed_ms(start, stop):
        return (stop.time - start.time) * 1000

    @staticmethod
    def synchronize():
        return None

    def download(self, buffer):
        self.download_calls.append(buffer)
        return buffer.value.copy()

    def memory_report(self):
        return {"known_native_buffer_bytes": sum(x.nbytes for x in self.buffers),
                "prepared_relation_bytes": self.relation.prepared_bytes}

    def generation(self):
        return self.relation.generation

    @staticmethod
    def close():
        return None


def analytic_inputs():
    _, W, _, _, left, right, _, _ = resident._analytic_case()
    return W, left, right


def fake_receipt(source_root):
    return {"source_root": str(source_root), "source_head": "cpu-test-head",
            "_validated": {"source_head_matches_build": True,
                           "head_mismatch_rule": "CPU test receipt",
                           "source_content_sha256": "cpu-test-source",
                           "extension_path": "/tmp/fake-native.so",
                           "extension_sha256": "fake-extension"}}


class ResidentHarnessTest(unittest.TestCase):
    def test_ten_fields_match_independent_sparse_oracle_on_rectangular_tail_case(self):
        W, left, right = analytic_inputs()
        backend = FakeBackend()
        resident_state = resident.prepare_resident(backend, W, left, right)
        self.assertEqual(resident_state["csr_u64_indptr"].dtype, np.uint64)
        self.assertEqual(resident_state["csr_u64_indices"].dtype, np.uint64)
        np.testing.assert_array_equal(resident_state["csr_u64_indptr"], W.indptr.astype(np.uint64))
        np.testing.assert_array_equal(resident_state["csr_u64_indices"], W.indices.astype(np.uint64))
        timing = resident.time_compositions(backend, resident_state, warmups=1, repeats=2)
        outputs, _, nbytes = resident.materialize(backend, resident_state["buffers"], harness.FIELDS)
        expected = harness.sparse_composition(W, sparse.csr_matrix(left), sparse.csr_matrix(right))
        result = resident.compare_analytic(outputs, expected, "analytic")
        self.assertTrue(result["all_pass"], result)
        self.assertEqual(outputs["mean_left"].shape, (4, 131))
        self.assertEqual(timing["generation"], resident_state["generation"])
        self.assertTrue(timing["relation_reused"])
        self.assertEqual(backend.prepare_calls, 1)
        self.assertEqual(backend.apply_calls, 15)  # five applies per warmup/iteration
        self.assertEqual(timing["operation_counts"],
                         {"multiply": 18, "relation_apply": 15, "axpby": 15})
        self.assertEqual(nbytes, 10 * 4 * 131 * np.dtype(np.float32).itemsize)
        memory = backend.memory_report()
        expected_known = (5 * 5 + 13 * 4) * 131 * 4 + W.nnz * 4
        self.assertEqual(memory["known_native_buffer_bytes"], expected_known)
        self.assertEqual(memory["prepared_relation_bytes"], 0)
        self.assertEqual(timing["event_stats"]["count"], 2)
        perturbed = {key: value.copy() for key, value in outputs.items()}
        perturbed["mean_left"][3, 0] = np.float32(2e-6)  # analytic empty-row zero
        self.assertFalse(resident.compare_analytic(perturbed, expected, "analytic")["all_pass"])

    def test_selective_materialization_has_exact_bytes_and_no_extra_downloads(self):
        W, left, right = analytic_inputs()
        backend = FakeBackend()
        resident_state = resident.prepare_resident(backend, W, left, right)
        resident.compose_once(backend, resident_state["relation"], resident_state["buffers"])
        means, _, mean_bytes = resident.materialize(backend, resident_state["buffers"], resident.MEAN_OUTPUTS)
        self.assertEqual(set(means), set(resident.MEAN_OUTPUTS))
        self.assertEqual(mean_bytes, 2 * 4 * 131 * 4)
        expected = harness.sparse_composition(W, sparse.csr_matrix(left), sparse.csr_matrix(right))
        self.assertTrue(resident.compare_analytic(means, expected, "analytic", resident.MEAN_OUTPUTS)["all_pass"])
        self.assertEqual(len(backend.download_calls), 2)
        all_fields, _, all_bytes = resident.materialize(backend, resident_state["buffers"], harness.FIELDS)
        self.assertEqual(set(all_fields), set(harness.FIELDS))
        self.assertEqual(all_bytes, 10 * 4 * 131 * 4)
        self.assertEqual(len(backend.download_calls), 12)

    def test_invalid_materialization_and_input_shapes_fail_before_native_work(self):
        W, left, right = analytic_inputs()
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, "dimensions"):
            resident.prepare_resident(backend, W, left[:-1], right[:-1])
        self.assertEqual(backend.prepare_calls, 0)
        with self.assertRaisesRegex(ValueError, "unknown requested"):
            resident.materialize(backend, {}, ("not_a_moment",))
        self.assertEqual(backend.download_calls, [])

    def test_build_receipt_binds_scoped_source_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            source = root / "binding.cc"
            source.write_text("stable source\n")
            package = Path(tmp) / "stage"
            package.mkdir()
            extension = package / "_native.so"
            extension.write_bytes(b"binary")
            source_hash = harness.sha256(source)
            content_hash = hashlib.sha256(f"binding.cc:{source_hash}\n".encode()).hexdigest()
            receipt = {"schema_version": 1, "source_root": str(root), "source_head": "abc",
                       "source_file_sha256": {"binding.cc": source_hash},
                       "source_content_sha256": content_hash, "package_root": str(package),
                       "extension_path": str(extension), "extension_sha256": harness.sha256(extension)}
            path = Path(tmp) / "receipt.json"
            path.write_text(json.dumps(receipt))
            loaded = resident.load_build_receipt(path, root)
            self.assertEqual(loaded["_validated"]["source_content_sha256"], content_hash)
            source.write_text("changed source\n")
            with self.assertRaisesRegex(RuntimeError, "source hash differs"):
                resident.load_build_receipt(path, root)
            source.write_text("stable source\n")
            extension.write_bytes(b"changed binary")
            with self.assertRaisesRegex(RuntimeError, "extension path/hash"):
                resident.load_build_receipt(path, root)

    def test_cpp_baseline_path_is_bound_to_receipt_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "ceNativeNeighborhoodMoments"
            binary.write_bytes(b"native baseline")
            receipt = {"native_executable": {"path": str(binary),
                                             "sha256": harness.sha256(binary)}}
            identity = resident.validate_cpp_executable(receipt, binary)
            self.assertEqual(identity["path"], str(binary.resolve()))
            self.assertEqual(identity["sha256"], harness.sha256(binary))
            with self.assertRaisesRegex(RuntimeError, "differs from build receipt"):
                resident.validate_cpp_executable(receipt, Path(tmp) / "other")
            binary.write_bytes(b"replaced native baseline")
            with self.assertRaisesRegex(RuntimeError, "path/hash"):
                resident.validate_cpp_executable(receipt, binary)

    def test_shared_typed_export_and_cpp_files_compare_all_ten_oracle_fields(self):
        W, left, right = analytic_inputs()
        expected = harness.sparse_composition(W, sparse.csr_matrix(left),
                                              sparse.csr_matrix(right))
        manifest = {"manifest_sha256": "fixture-manifest", "shape": {
            "sources": 5, "destinations": 4, "features": 131, "edges": W.nnz},
            "array_sha256": {"indptr_u64": harness.array_sha256(W.indptr.astype(np.uint64)),
                             "indices_u64": harness.array_sha256(W.indices.astype(np.uint64)),
                             "weights": harness.array_sha256(W.data.astype(np.float32)),
                             "left": harness.array_sha256(left),
                             "right": harness.array_sha256(right)}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            typed = resident.export_shared_finalized_inputs(root / "typed", W, left, right, manifest)
            self.assertEqual(typed["matched_input_manifest_sha256"], "fixture-manifest")
            np.testing.assert_array_equal(np.fromfile(root / "typed" / "indptr.u64", dtype="<u8"), W.indptr)
            np.testing.assert_array_equal(np.fromfile(root / "typed" / "indices.u64", dtype="<u8"), W.indices)
            self.assertEqual(np.fromfile(root / "typed" / "left.f32", dtype="<f4").size, left.size)
            output = root / "cpp-output"
            output.mkdir()
            for name in harness.FIELDS:
                np.asarray(expected[name], dtype="<f4").tofile(output / f"{name}.f32")
            result = resident.compare_cpp_output_files(output, expected, (4, 131), "analytic")
            self.assertTrue(result["all_pass"], result)
            self.assertEqual(set(result["fields"]), set(harness.FIELDS))
            # The analytic gate applies to native files too, including empty rows.
            changed = expected["mean_left"].copy()
            changed[3, 0] = np.float32(2e-6)
            changed.tofile(output / "mean_left.f32")
            failed = resident.compare_cpp_output_files(output, expected, (4, 131), "analytic")
            self.assertFalse(failed["fields"]["mean_left"]["pass"])

    def test_cpp_parity_runner_exports_exact_case_and_uses_shared_oracle(self):
        W, left, right = analytic_inputs()
        left_sp, right_sp = sparse.csr_matrix(left), sparse.csr_matrix(right)
        expected = harness.sparse_composition(W, left_sp, right_sp)
        manifest = {"manifest_sha256": "same-case", "shape": {
            "sources": 5, "destinations": 4, "features": 131, "edges": W.nnz},
            "array_sha256": {"left": harness.array_sha256(left),
                             "right": harness.array_sha256(right)}}

        def fake_invoke(args, input_dir, result_dir, shape):
            self.assertEqual((args.warmups, args.repeats), (0, 1))
            self.assertEqual(shape, (5, 4, 131, W.nnz))
            result_dir.mkdir(parents=True, exist_ok=True)
            for name in harness.FIELDS:
                np.asarray(expected[name], dtype="<f4").tofile(result_dir / f"{name}.f32")
            np.testing.assert_array_equal(np.fromfile(input_dir / "left.f32", dtype="<f4").reshape(left.shape), left)
            np.testing.assert_array_equal(np.fromfile(input_dir / "right.f32", dtype="<f4").reshape(right.shape), right)
            return {"gpu": {"name": "test-device", "uuid": "cpu-fake"}, "process_total_ms": 1.0}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(resident.harness, "invoke_native", fake_invoke):
            typed = resident.export_shared_finalized_inputs(Path(tmp) / "case" / "shared-finalized-inputs",
                                                             W, left, right, manifest)
            result = resident.run_cpp_baseline(Path(tmp) / "native", {"path": "verified", "sha256": "hash"},
                                               Path(tmp) / "case", W, left_sp, right_sp,
                                               left, right, expected, "analytic", manifest,
                                               "same-case", typed)
            self.assertTrue(result["correctness"]["all_pass"])
            self.assertEqual(result["matched_input_manifest_sha256"], "same-case")
            self.assertEqual(result["matched_input_array_sha256"], manifest["array_sha256"])
            self.assertEqual(result["device_identity"]["uuid"], "cpu-fake")
            self.assertEqual(result["shared_typed_input_manifest"]["files"]["indptr.u64"]["bytes"],
                             W.indptr.size * np.dtype(np.uint64).itemsize)

    def test_cold_path_estimate_uses_staged_source_inputs_and_labels_its_scope(self):
        recipe = {"source_read_ms": 10, "graph_and_sparse_field_preparation_ms": 5,
                  "dense_input_materialization_ms": 2,
                  "second_moments_ms": 100}
        self.assertEqual(resident.source_preparation_ms(recipe), 17)
        synthetic = {"generation_stages_ms": {"graph": 3, "fields": 4, "densify": 5}}
        self.assertEqual(resident.source_preparation_ms(synthetic), 12)

    def test_analytic_case_tuple_runs_through_backend_orchestration(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), backend="direct", native_executable=None,
                                   warmups=0, repeats=1)
            with mock.patch.object(resident, "_backend_class", return_value=FakeBackend):
                result = resident._run_one_case("analytic", None, resident._analytic_case(),
                                                 args, fake_receipt(Path(tmp)), None)
            provider = result["providers"]["direct"]
            self.assertTrue(provider["correctness"]["all_pass"], provider["correctness"])
            self.assertEqual(result["manifest"]["shape"], {
                "sources": 5, "destinations": 4, "features": 131, "edges": 8})
            self.assertEqual(result["typed_inputs"]["matched_input_manifest_sha256"],
                             result["manifest"]["manifest_sha256"])

    def test_synthetic_tuple_and_deferred_oracle_feed_scipy_and_fake_backend(self):
        W = sparse.csr_matrix(np.array([[0.5, 0.5, 0, 0, 0, 0],
                                        [0, 0.25, 0.25, 0.5, 0, 0],
                                        [0, 0, 0, 0.25, 0.25, 0.5]], dtype=np.float32))
        left = np.arange(42, dtype=np.float32).reshape(6, 7) / 10
        right = np.flip(left, axis=0).copy() - np.float32(1)
        left_sp, right_sp = sparse.csr_matrix(left), sparse.csr_matrix(right)
        generation = {"graph_generation_and_normalization_ms": 0.1,
                      "sparse_feature_source_generation_ms": 0.2,
                      "sparse_to_dense_materialization_ms": 0.3}
        synthetic_call = {}

        def fake_synthetic(seed, n, features, degree, density):
            synthetic_call.update(seed=seed, n=n, features=features,
                                  degree=degree, density=density)
            return W, left_sp, left, right_sp, right, generation

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), backend="direct", native_executable=None,
                                   warmups=1, repeats=2)
            with mock.patch.object(resident.harness, "synthetic", fake_synthetic), \
                    mock.patch.object(resident, "_backend_class", return_value=FakeBackend):
                raw = resident._load_case("compute", args)
                self.assertIsNone(raw[0])
                self.assertEqual(raw[1].shape, W.shape)
                self.assertIs(raw[2], left_sp)
                self.assertIs(raw[3], right_sp)
                np.testing.assert_array_equal(raw[4], left)
                np.testing.assert_array_equal(raw[5], right)
                self.assertIsNone(raw[6])  # no eager, duplicate dense reference
                result = resident._run_synthetic("compute", args, fake_receipt(Path(tmp)))
            provider = result["providers"]["direct"]
            self.assertTrue(provider["correctness"]["all_pass"], provider["correctness"])
            self.assertEqual(result["manifest"]["shape"], {
                "sources": 6, "destinations": 3, "features": 7, "edges": W.nnz})
            self.assertEqual(result["manifest"]["array_sha256"]["left"],
                             harness.array_sha256(left))
            self.assertEqual(result["manifest"]["array_sha256"]["right"],
                             harness.array_sha256(right))
            self.assertEqual(synthetic_call["n"], 8192)
            self.assertEqual(result["scipy_reference"]["repeats"], 2)

    def test_oracle_import_does_not_import_binding_or_torch_modules(self):
        self.assertNotIn("cellerator", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
