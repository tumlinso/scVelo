#!/usr/bin/env python3
"""Scientific consumer probe for Cellerator's resident Python bindings.

No Cellerator, CUDA, or Torch import occurs until a selected GPU backend is
constructed. CPU tests inject a small backend implementing the same operation
surface.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse

import harness


RAW_OUTPUTS = ("mean_left", "mean_right", "second_left", "cross_second", "second_right")
MEAN_OUTPUTS = ("mean_left", "mean_right")
SCVELO_SOURCE_FILES = (
    "probes/native_moments/harness.py", "probes/native_moments/test_harness.py",
    "probes/native_moments/resident_moments.py",
    "probes/native_moments/test_resident_moments.py",
    "docs/native-moments-probe.md", "scvelo/preprocessing/moments.py",
    "scvelo/preprocessing/neighbors.py", "scvelo/preprocessing/utils.py",
    "scvelo/tools/_steady_state_model.py",
)
CELLRANK_SOURCE_FILES = ("src/cellrank/kernels/utils/_moments.py",)


def _source_hashes(root: Path, entries: dict[str, str]) -> tuple[dict[str, str], str]:
    actual = {}
    for relative, expected in sorted(entries.items()):
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f"Invalid or missing Cellerator source receipt path: {relative}")
        digest = harness.sha256(path)
        if digest != expected:
            raise RuntimeError(f"Cellerator source hash differs from build receipt: {relative}")
        actual[relative] = digest
    content = hashlib.sha256("".join(f"{key}:{value}\n" for key, value in sorted(actual.items())).encode()).hexdigest()
    return actual, content


def load_build_receipt(path: Path, cellerator_root: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") != 1:
        raise RuntimeError("Unsupported Cellerator build receipt schema")
    if Path(receipt.get("source_root", "")).resolve() != cellerator_root.resolve():
        raise RuntimeError("Build receipt source_root differs from --cellerator-root")
    source_files = receipt.get("source_file_sha256")
    if not isinstance(source_files, dict) or not source_files:
        raise RuntimeError("Build receipt must contain source_file_sha256")
    hashes, content_hash = _source_hashes(cellerator_root, source_files)
    if receipt.get("source_content_sha256") != content_hash:
        raise RuntimeError("Build receipt source_content_sha256 does not match scoped source files")
    extension = Path(receipt.get("extension_path", "")).resolve()
    if not extension.is_file() or harness.sha256(extension) != receipt.get("extension_sha256"):
        raise RuntimeError("Build receipt extension path/hash is missing or mismatched")
    package_root = Path(receipt.get("package_root", "")).resolve()
    if not package_root.is_dir():
        raise RuntimeError("Build receipt package_root does not exist")
    receipt["_validated"] = {
        "source_file_sha256": hashes, "source_content_sha256": content_hash,
        "extension_path": str(extension), "extension_sha256": receipt["extension_sha256"],
        "package_root": str(package_root),
        "current_source_head": harness.revision(cellerator_root),
        "source_head_matches_build": harness.revision(cellerator_root) == receipt.get("source_head"),
        "head_mismatch_rule": "accepted only because every receipt-scoped source hash and aggregate content digest match",
    }
    return receipt


def validate_cpp_executable(receipt: dict[str, Any], requested_path: Path) -> dict[str, str]:
    """Require the optional C++ baseline to match its build receipt exactly."""
    entry = receipt.get("native_executable")
    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
        raise RuntimeError("Build receipt lacks native_executable.path/sha256")
    resolved = requested_path.resolve()
    receipt_path = Path(entry["path"]).resolve()
    if resolved != receipt_path:
        raise RuntimeError("--native-executable path differs from build receipt")
    if not resolved.is_file() or harness.sha256(resolved) != entry["sha256"]:
        raise RuntimeError("C++ native executable path/hash is missing or mismatched")
    return {"path": str(resolved), "sha256": entry["sha256"]}


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def import_binding(backend: str, receipt: dict[str, Any]):
    """Import only from the receipt's installed package and verify the binary."""
    package_root = Path(receipt["_validated"]["package_root"])
    sys.path.insert(0, str(package_root))
    package = importlib.import_module("cellerator")
    package_file = Path(package.__file__).resolve()
    if not _inside(package_file, package_root / "cellerator"):
        raise RuntimeError(f"Imported cellerator package is outside receipt package_root: {package_file}")
    cuda = importlib.import_module("cellerator.cuda")
    native = importlib.import_module("cellerator._native")
    if Path(native.__file__).resolve() != Path(receipt["_validated"]["extension_path"]).resolve():
        raise RuntimeError("cellerator._native path differs from the build receipt")
    if not getattr(cuda, "resident_cuda_available", False):
        raise RuntimeError("Imported Cellerator build does not expose resident CUDA operations")
    torch_cuda = None
    if backend == "torch":
        torch_cuda = importlib.import_module("cellerator.torch.cuda")
    extension_path = Path(receipt["_validated"]["extension_path"]).resolve()
    loaded = []
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename:
            candidate = Path(filename).resolve()
            if candidate == extension_path:
                loaded.append(name)
    if not loaded or harness.sha256(extension_path) != receipt["_validated"]["extension_sha256"]:
        raise RuntimeError("Imported Cellerator extension does not match the build receipt")
    if backend == "direct":
        return cuda, package, loaded
    return (torch_cuda, cuda, package, loaded)


class DirectBackend:
    """Adapter for `cellerator.cuda`; imports lazily and never selects a GPU."""

    name = "direct"

    def __init__(self, receipt: dict[str, Any]):
        self.cuda, self.package, self.loaded_extensions = import_binding("direct", receipt)
        # Controller admission maps its single visible device to ordinal zero.
        self.stream = self.cuda.Stream(0)
        self.device = int(self.stream.device)
        self.buffers = []
        self.relation = None

    def allocate(self, shape):
        value = self.cuda.Buffer(tuple(shape), self.stream)
        self.buffers.append(value)
        return value

    @staticmethod
    def nbytes(value):
        return int(value.nbytes)

    def upload(self, buffer, array):
        buffer.upload(np.ascontiguousarray(array, dtype=np.float32))

    def create_relation(self, indptr, indices, weights, source_count, feature_width):
        self.relation = self.cuda.PreparedCsr(indptr, indices, weights, source_count,
                                              feature_width, self.stream)
        return self.relation

    def apply(self, relation, value, output):
        relation.apply_into(value, output)

    def multiply(self, a, b, out):
        self.cuda.multiply_into(a, b, out, self.stream)

    def axpby(self, alpha, a, beta, b, out):
        self.cuda.axpby_into(alpha, a, beta, b, out, self.stream)

    def event(self):
        return self.cuda.Event.record(self.stream)

    @staticmethod
    def elapsed_ms(start, stop):
        return float(start.elapsed_ms(stop))

    def synchronize(self):
        self.stream.synchronize()

    def download(self, buffer):
        return np.asarray(buffer.download(), dtype=np.float32)

    def memory_report(self):
        return {"known_native_buffer_bytes": sum(self.nbytes(x) for x in self.buffers),
                "prepared_relation_bytes": int(self.relation.prepared_bytes),
                "prepared_relation_bytes_scope": "reported by native PreparedCsr; separately scoped"}

    def generation(self):
        return int(self.relation.generation)

    def close(self):
        if self.relation is not None:
            self.relation.close()


class _TorchBuffer:
    __slots__ = ("tensor", "native")

    def __init__(self, tensor, native):
        self.tensor, self.native = tensor, native


class TorchBackend:
    """Torch supplies storage/transfers; every arithmetic operation is native."""

    name = "torch"

    def __init__(self, receipt: dict[str, Any]):
        self.tcuda, self.cuda, self.package, self.loaded_extensions = import_binding("torch", receipt)
        import torch
        self.torch = torch
        self.device = torch.cuda.current_device()
        self.stream = self.tcuda.current_stream(self.device)
        self.buffers = []
        self.relation = None

    def allocate(self, shape):
        tensor = self.torch.empty(tuple(shape), dtype=self.torch.float32,
                                  device=self.device)
        value = _TorchBuffer(tensor, self.tcuda.borrow(tensor, self.stream))
        self.buffers.append(value)
        return value

    @staticmethod
    def nbytes(value):
        return int(value.tensor.numel() * value.tensor.element_size())

    def upload(self, buffer, array):
        host = self.torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        buffer.tensor.copy_(host, non_blocking=False)
        self.synchronize()

    def create_relation(self, indptr, indices, weights, source_count, feature_width):
        self.relation = self.tcuda.PreparedCsr(indptr, indices, weights.tensor,
                                              source_count, feature_width,
                                              stream=self.stream)
        return self.relation

    def apply(self, relation, value, output):
        relation.apply_into(value.tensor, output.tensor)

    def multiply(self, a, b, out):
        self.tcuda.multiply_into(a.tensor, b.tensor, out.tensor, stream=self.stream)

    def axpby(self, alpha, a, beta, b, out):
        self.tcuda.axpby_into(alpha, a.tensor, beta, b.tensor, out.tensor,
                              stream=self.stream)

    def event(self):
        return self.cuda.Event.record(self.stream)

    @staticmethod
    def elapsed_ms(start, stop):
        return float(start.elapsed_ms(stop))

    def synchronize(self):
        self.stream.synchronize()

    def download(self, buffer):
        return np.asarray(buffer.tensor.detach().cpu().numpy(), dtype=np.float32)

    def memory_report(self):
        device = self.torch.device("cuda", self.device)
        return {"known_native_buffer_bytes": sum(self.nbytes(x) for x in self.buffers),
                "prepared_relation_bytes": int(self.relation.native.prepared_bytes),
                "prepared_relation_bytes_scope": "reported by native PreparedCsr; separately scoped",
                "torch_cuda_memory_allocated_bytes_process_wide": int(self.torch.cuda.memory_allocated(device)),
                "torch_cuda_memory_reserved_bytes_process_wide": int(self.torch.cuda.memory_reserved(device)),
                "torch_memory_scope": "process-wide allocator counters; may include caching/unrelated live tensors"}

    def generation(self):
        return int(self.relation.generation)

    def close(self):
        if self.relation is not None:
            self.relation.close()


def _validate_inputs(W, left, right):
    if not sparse.issparse(W) or len(W.shape) != 2:
        raise TypeError("W must be a two-dimensional SciPy sparse matrix")
    if left.dtype != np.float32 or right.dtype != np.float32:
        raise TypeError("feature inputs must be FP32")
    if left.ndim != 2 or right.shape != left.shape or W.shape[1] != left.shape[0]:
        raise ValueError("relation and input dimensions do not agree")
    if left.shape[1] == 0 or W.shape[0] == 0:
        raise ValueError("destination and feature dimensions must be nonzero")
    csr = W.tocsr(copy=True)
    if csr.data.dtype != np.float32:
        raise TypeError("relation weights must be FP32")
    if not np.isfinite(csr.data).all() or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("inputs must be finite")
    return csr, np.ascontiguousarray(left), np.ascontiguousarray(right)


def allocate_workspaces(backend, source_count: int, destination_count: int, feature_width: int):
    matrix = (destination_count, feature_width)
    source = (source_count, feature_width)
    names = ("left", "right", "product_left2", "product_cross", "product_right2",
             *RAW_OUTPUTS, "mean_left2", "mean_product", "mean_right2",
             "variance_left", "covariance", "variance_right", "affine_second_left",
             "affine_cross_second")
    source_names = {"left", "right", "product_left2", "product_cross", "product_right2"}
    result = {name: backend.allocate(source if name in source_names else matrix)
              for name in names}
    return result


def compose_once(backend, relation, buffers):
    """Produce five raw relation outputs and five derived outputs in place."""
    b = buffers
    backend.multiply(b["left"], b["left"], b["product_left2"])
    backend.multiply(b["left"], b["right"], b["product_cross"])
    backend.multiply(b["right"], b["right"], b["product_right2"])
    backend.apply(relation, b["left"], b["mean_left"])
    backend.apply(relation, b["right"], b["mean_right"])
    backend.apply(relation, b["product_left2"], b["second_left"])
    backend.apply(relation, b["product_cross"], b["cross_second"])
    backend.apply(relation, b["product_right2"], b["second_right"])
    backend.multiply(b["mean_left"], b["mean_left"], b["mean_left2"])
    backend.multiply(b["mean_left"], b["mean_right"], b["mean_product"])
    backend.multiply(b["mean_right"], b["mean_right"], b["mean_right2"])
    backend.axpby(1.0, b["second_left"], -1.0, b["mean_left2"], b["variance_left"])
    backend.axpby(1.0, b["second_right"], -1.0, b["mean_right2"], b["variance_right"])
    backend.axpby(1.0, b["cross_second"], -1.0, b["mean_product"], b["covariance"])
    backend.axpby(2.0, b["second_left"], -1.0, b["mean_left"], b["affine_second_left"])
    backend.axpby(2.0, b["cross_second"], -1.0, b["mean_right"], b["affine_cross_second"])


def materialize(backend, buffers, selected):
    unknown = set(selected) - set(harness.FIELDS)
    if unknown:
        raise ValueError(f"unknown requested output fields: {sorted(unknown)}")
    outputs, elapsed_bytes = {}, 0
    start = time.perf_counter()
    for name in selected:
        outputs[name] = backend.download(buffers[name])
        elapsed_bytes += outputs[name].nbytes
    return outputs, (time.perf_counter() - start) * 1000, elapsed_bytes


def prepare_resident(backend, W, left, right):
    whole_started = time.perf_counter()
    validation_started = time.perf_counter()
    csr, left, right = _validate_inputs(W, left, right)
    validation_ms = (time.perf_counter() - validation_started) * 1000
    sources, features = left.shape
    destinations = csr.shape[0]
    started = time.perf_counter()
    buffers = allocate_workspaces(backend, sources, destinations, features)
    weights = backend.allocate((csr.nnz,))
    buffers["weights"] = weights
    allocation_ms = (time.perf_counter() - started) * 1000
    upload_started = time.perf_counter()
    backend.upload(buffers["left"], left)
    backend.upload(buffers["right"], right)
    backend.upload(weights, np.ascontiguousarray(csr.data, dtype=np.float32))
    h2d_ms = (time.perf_counter() - upload_started) * 1000
    prepare_started = time.perf_counter()
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.uint64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.uint64)
    relation = backend.create_relation(indptr, indices, weights, sources, features)
    backend.synchronize()
    preparation_ms = (time.perf_counter() - prepare_started) * 1000
    return {"buffers": buffers, "relation": relation, "csr": csr,
            "shape": {"sources": sources, "destinations": destinations,
                      "features": features, "edges": int(csr.nnz)},
            "csr_u64_indptr": indptr, "csr_u64_indices": indices,
            "input_validation_ms": validation_ms,
            "allocation_ms": allocation_ms, "h2d_ms": h2d_ms,
            "preparation_ms": preparation_ms, "generation": backend.generation(),
            "input_to_resident_ready_wall_ms": (time.perf_counter() - whole_started) * 1000}


def time_compositions(backend, resident, warmups=5, repeats=20):
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats positive")
    for _ in range(warmups):
        compose_once(backend, resident["relation"], resident["buffers"])
    backend.synchronize()
    events, walls = [], []
    for _ in range(repeats):
        start_event = backend.event()
        wall_start = time.perf_counter()
        compose_once(backend, resident["relation"], resident["buffers"])
        stop_event = backend.event()
        event_ms = backend.elapsed_ms(start_event, stop_event)
        wall_ms = (time.perf_counter() - wall_start) * 1000
        events.append(event_ms)
        walls.append(wall_ms)
    generation_after = backend.generation()
    if generation_after != resident["generation"]:
        raise RuntimeError("prepared relation generation changed during repeated applications")
    return {"warmups": warmups, "repeats": repeats, "event_ms": events,
            "wall_ms": walls, "event_stats": _timing_stats(events),
            "wall_stats": _timing_stats(walls), "generation": generation_after,
            "relation_reused": True, "relation_preparation_count": 1,
            "operation_counts": {"multiply": 6 * (warmups + repeats),
                                 "relation_apply": 5 * (warmups + repeats),
                                 "axpby": 5 * (warmups + repeats)}}


def _timing_stats(values):
    return {"count": len(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values),
            "stddev_ms": statistics.pstdev(values),
            "cv": statistics.pstdev(values) / statistics.mean(values) if statistics.mean(values) else None}


def scipy_benchmark(W, left_sp, right_sp, warmups=5, repeats=20):
    samples = []
    expected = None
    for i in range(warmups + repeats):
        # Release the prior dense ten-field result before constructing the next
        # one; the final result is retained for provider comparisons.
        if expected is not None:
            del expected
        start = time.perf_counter()
        expected = harness.sparse_composition(W, left_sp, right_sp)
        elapsed = (time.perf_counter() - start) * 1000
        if i >= warmups:
            samples.append(elapsed)
    return {"expected": expected, "warmups": warmups, "repeats": repeats,
            "samples_ms": samples, "stats": _timing_stats(samples)}


def _analytic_case():
    # Rectangular graph with a truly empty destination row; signed, zero, and
    # constant feature columns plus an odd width exercise tail handling.
    rows = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    cols = np.array([0, 2, 4, 0, 3, 1, 3, 4], dtype=np.int64)
    weights = np.array([0.25, 0.5, 0.25, 0.75, 0.25, 0.2, 0.3, 0.5], dtype=np.float32)
    W = sparse.csr_matrix((weights, (rows, cols)), shape=(4, 5), dtype=np.float32)
    left_head = np.array([[-2, 0, 7, 1, 0, -1, 3], [4, 0, 7, 2, 0, 2, -5],
                          [-1, 0, 7, -3, 0, 0, 8], [3, 0, 7, 5, 0, -4, 1],
                          [0, 0, 7, -2, 0, 3, -2]], dtype=np.float32)
    right_head = np.array([[1, 0, -2, 2, 0, 3, 1], [-3, 0, -2, -1, 0, 0, 4],
                           [2, 0, -2, 4, 0, -5, -1], [0, 0, -2, 1, 0, 1, 2],
                           [-4, 0, -2, -3, 0, 2, 6]], dtype=np.float32)
    left, right = np.zeros((5, 131), np.float32), np.zeros((5, 131), np.float32)
    left[:, :7], right[:, :7] = left_head, right_head
    left_sp, right_sp = sparse.csr_matrix(left), sparse.csr_matrix(right)
    return None, W, left_sp, right_sp, left, right, harness.sparse_composition(W, left_sp, right_sp), {
        "name": "analytic_rectangular_empty_row_signed_zero_constant_odd_width_v1"}


def compare_analytic(actual, expected, dataset, selected=harness.FIELDS):
    fields = {}
    for name in selected:
        a = np.asarray(actual[name], dtype=np.float32)
        e = np.asarray(expected[name], dtype=np.float32)
        finite = bool(np.isfinite(a).all() and np.isfinite(e).all())
        error = np.abs(a.astype(np.float64) - e.astype(np.float64))
        fields[name] = {"pass": bool(finite and np.allclose(a, e, rtol=1e-6, atol=1e-6)),
                        "max_abs_error": float(error.max(initial=0)),
                        "near_zero_count": int(np.count_nonzero(np.abs(e) <= 1e-6))}
    return {"dataset": dataset, "all_pass": all(x["pass"] for x in fields.values()), "fields": fields,
            "gate": {"rtol": 1e-6, "atol": 1e-6}}


def _load_case(case_name, args):
    if case_name == "analytic":
        return _analytic_case()
    if case_name == "real":
        raise ValueError("real case must be resolved per fixture")
    specs = {
        "compute": ("synthetic-compute-8192x2048", 8192, 2048, 0.4, 20261006),
        "transfer": ("synthetic-transfer-32768x2048", 32768, 2048, 0.25, 20261007),
    }
    dataset_name, n, features, density, seed = specs[case_name]
    W, left_sp, left, right_sp, right, generation = harness.synthetic(seed, n, features, 32, density)
    # Defer the dense ten-field reference until the shared SciPy timing loop;
    # computing it here would overlap two multi-gigabyte oracle results.
    return None, W, left_sp, right_sp, left, right, None, {
        "dataset": dataset_name, "name": "independent_uniform_unique_nonself_neighbors_v1", "seed": seed,
        "density": density, "degree": 32, "self_loop_policy": "excluded",
        "generation_stages_ms": generation}


def _real_cases(args):
    for name, relative in harness.SCVELO_DATASETS.items():
        source = (args.scvelo_root / relative).resolve()
        yield name, source, harness.prepare_scvelo(args.scvelo_root, source)
    source = (args.cellrank_root / harness.CELLRANK_DATASET).resolve()
    yield "cellrank-200", source, harness.prepare_cellrank(args.cellrank_root, source)


def _manifest(case_name, source, W, left_sp, right_sp, left, right, recipe, graph_policy, receipt):
    csr = W.tocsr(copy=True)
    return {"schema_version": 1, "case": case_name,
            "source_fixture": str(source) if source else None,
            "source_fixture_sha256": harness.sha256(source) if source else None,
            "recipe": recipe, "shape": {"sources": int(W.shape[1]), "destinations": int(W.shape[0]),
                                         "features": int(left.shape[1]), "edges": int(csr.nnz)},
            "graph_policy": graph_policy,
            "csr_order": "preserved input CSR edge order",
            "csr_indptr_dtype": "uint64", "csr_indices_dtype": "uint64",
            "array_sha256": {"left": harness.array_sha256(left), "right": harness.array_sha256(right),
                             "weights": harness.array_sha256(csr.data.astype(np.float32)),
                             "indptr_u64": harness.array_sha256(csr.indptr.astype(np.uint64)),
                             "indices_u64": harness.array_sha256(csr.indices.astype(np.uint64)),
                             "sparse_left_data": harness.array_sha256(left_sp.data.astype(np.float32)),
                             "sparse_left_indices": harness.array_sha256(left_sp.indices.astype(np.uint64)),
                             "sparse_left_indptr": harness.array_sha256(left_sp.indptr.astype(np.uint64)),
                             "sparse_right_data": harness.array_sha256(right_sp.data.astype(np.float32)),
                             "sparse_right_indices": harness.array_sha256(right_sp.indices.astype(np.uint64)),
                             "sparse_right_indptr": harness.array_sha256(right_sp.indptr.astype(np.uint64))},
            "cellerator_source_head_at_build": receipt.get("source_head"),
            "cellerator_source_head_current": harness.revision(Path(receipt["source_root"])),
            "cellerator_source_head_matches_build": receipt["_validated"]["source_head_matches_build"],
            "cellerator_head_match_policy": receipt["_validated"]["head_mismatch_rule"],
            "cellerator_source_content_sha256": receipt["_validated"]["source_content_sha256"],
            "cellerator_extension_path": receipt["_validated"]["extension_path"],
            "cellerator_extension_sha256": receipt["_validated"]["extension_sha256"]}


def _scvelo_cellrank_provenance(args):
    return {
        "scvelo": {"root": str(args.scvelo_root),
                   **harness.scoped_source_identity(args.scvelo_root, SCVELO_SOURCE_FILES)},
        "cellrank": {"root": str(args.cellrank_root),
                     **harness.scoped_source_identity(args.cellrank_root, CELLRANK_SOURCE_FILES)},
    }


def run_backend_case(backend, W, left, right, expected, data, output_dir,
                     warmups, repeats, benchmark_reference=None,
                     input_manifest_sha256=None, input_array_sha256=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    resident = prepare_resident(backend, W, left, right)
    native_timing = time_compositions(backend, resident, warmups, repeats)
    means, means_ms, means_bytes = materialize(backend, resident["buffers"], MEAN_OUTPUTS)
    case_name = output_dir.parent.name
    if case_name == "analytic":
        means_correctness = compare_analytic(means, expected, case_name, MEAN_OUTPUTS)["fields"]
    else:
        means_correctness = {name: harness.compare_field(name, means[name], expected)
                             for name in MEAN_OUTPUTS}
    downstream = harness.downstream(data, means, expected) if data is not None else None
    del means
    all_values, all_ms, all_bytes = materialize(backend, resident["buffers"], harness.FIELDS)
    gate = compare_analytic(all_values, expected, case_name) if case_name == "analytic" else harness.compare(all_values, expected, case_name)
    del all_values
    memory = backend.memory_report()
    record = {
        "backend": backend.name, "native_composition": native_timing,
        "materialization": {
            "means_only": {"fields": list(MEAN_OUTPUTS), "correctness": means_correctness,
                           "all_pass": all(x["pass"] for x in means_correctness.values()),
                           "host_wall_ms": means_ms,
                           "materialized_host_bytes": means_bytes},
            "all_outputs": {"fields": list(harness.FIELDS), "host_wall_ms": all_ms,
                            "materialized_host_bytes": all_bytes},
            "note": "Both transfers were measured after composition; materialization timing is outside resident GPU-event samples."},
        "correctness": gate, "downstream": downstream,
        "binding_identity": {"package_path": str(Path(backend.package.__file__).resolve()),
                             "extension_module_names": backend.loaded_extensions,
                             "package_version": getattr(backend.package, "__version__", "unknown"),
                             "device_ordinal": int(backend.device),
                             "torch_version": getattr(backend.torch, "__version__", None)
                             if backend.name == "torch" else None},
        "matched_input_manifest_sha256": input_manifest_sha256,
        "matched_input_array_sha256": input_array_sha256,
        "allocation_ms": resident["allocation_ms"], "h2d_ms": resident["h2d_ms"],
        "input_validation_ms": resident["input_validation_ms"],
        "relation_preparation_ms": resident["preparation_ms"],
        "input_to_resident_ready_wall_ms": resident["input_to_resident_ready_wall_ms"],
        "cold_cost_scope": "one prepare_resident wall interval from input validation through allocations, uploads, and prepared-relation readiness; stage timings are nested diagnostics and must not be added to this interval",
        "prepared_shape": resident["shape"],
        "memory": memory,
        "comparison_reference": benchmark_reference,
    }
    if benchmark_reference is not None:
        cpu_median = benchmark_reference["stats"]["median_ms"]
        event_median = native_timing["event_stats"]["median_ms"]
        wall_median = native_timing["wall_stats"]["median_ms"]
        record["resident_only_ratios"] = {
            "scipy_median_over_native_event_median": cpu_median / event_median if event_median else None,
            "scipy_median_over_native_host_wall_median": cpu_median / wall_median if wall_median else None,
            "scope": "matched sparse SciPy composition over resident native composition only; excludes setup, materialization, and end-to-end project costs",
        }
    (output_dir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def export_shared_finalized_inputs(directory: Path, W, left, right, manifest):
    """Write typed uint64 CSR and dense FP32 inputs shared across providers."""
    directory.mkdir(parents=True, exist_ok=True)
    csr = W.tocsr(copy=True)
    arrays = {
        "indptr.u64": np.ascontiguousarray(csr.indptr, dtype="<u8"),
        "indices.u64": np.ascontiguousarray(csr.indices, dtype="<u8"),
        "weights.f32": np.ascontiguousarray(csr.data, dtype="<f4"),
        "left.f32": np.ascontiguousarray(left, dtype="<f4"),
        "right.f32": np.ascontiguousarray(right, dtype="<f4"),
    }
    files = {}
    for name, value in arrays.items():
        path = directory / name
        value.tofile(path)
        files[name] = {"sha256": harness.sha256(path), "bytes": path.stat().st_size}
    record = {"schema_version": 1, "shape": manifest["shape"],
              "csr_edge_order": "finalized source CSR edge order preserved",
              "matched_input_manifest_sha256": manifest["manifest_sha256"],
              "logical_array_sha256": manifest["array_sha256"],
              "files": files}
    (directory / "manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def compare_cpp_output_files(directory: Path, expected, shape, dataset):
    """Stream C++ output fields and apply the same analytic/scientific gates."""
    fields = {}
    for name in harness.FIELDS:
        path = directory / f"{name}.f32"
        wanted_bytes = shape[0] * shape[1] * 4
        if not path.exists() or path.stat().st_size != wanted_bytes:
            fields[name] = {"pass": False, "reason": f"wrong output byte count; wanted {wanted_bytes}"}
            continue
        actual = np.fromfile(path, dtype="<f4").reshape(shape)
        if dataset == "analytic":
            result = compare_analytic({name: actual}, expected, dataset, (name,))["fields"][name]
        else:
            result = harness.compare_field(name, actual, expected)
        fields[name] = result
        del actual
        harness.gc.collect()
    return {"dataset": dataset, "all_pass": all(value["pass"] for value in fields.values()),
            "fields": fields}


def run_cpp_baseline(binary: Path, receipt_identity, case_dir: Path, W, left_sp, right_sp,
                     left, right, expected, dataset, manifest, manifest_digest,
                     typed_manifest):
    """Run the old native executable against the exact case arrays and shared oracle."""
    output_dir = case_dir / "cpp-baseline"
    input_dir, result_dir = output_dir / "inputs", output_dir / "outputs"
    cpp_manifest = dict(manifest)
    cpp_manifest["manifest_sha256"] = manifest_digest
    harness.write_inputs(input_dir, W, left_sp, right_sp, left, right, cpp_manifest)
    shape = (int(W.shape[1]), int(W.shape[0]), int(left.shape[1]), int(W.nnz))
    metrics = harness.invoke_native(SimpleNamespace(native=str(binary), warmups=0, repeats=1),
                                    input_dir, result_dir, shape)
    correctness = compare_cpp_output_files(result_dir, expected,
                                           (shape[1], shape[2]), dataset)
    return {"backend": "native-cpp", "executable": receipt_identity,
            "matched_input_manifest_sha256": manifest_digest,
            "matched_input_array_sha256": manifest["array_sha256"],
            "shared_typed_input_manifest": typed_manifest,
            "correctness": correctness,
            "native_metrics": metrics,
            "device_identity": metrics.get("gpu"),
            "timing_scope": "parity-only invocation: zero warmups and one measured composition plus the separately timed diagnostic; process_total_ms includes startup, input read, setup, diagnostic, composition, materialization, and output writing and is not an end-to-end project benchmark"}


def source_preparation_ms(recipe):
    """Return measured source loading/graph/densification stages for cold estimates."""
    if "source_read_ms" in recipe:
        keys = ("source_read_ms", "graph_and_sparse_field_preparation_ms",
                "dense_input_materialization_ms")
        return sum(float(recipe.get(key, 0.0)) for key in keys)
    stages = recipe.get("generation_stages_ms") or {}
    return sum(float(value) for value in stages.values()) if stages else 0.0


def _backend_class(name):
    return DirectBackend if name == "direct" else TorchBackend


def _run_one_case(case_name, source, prepared, args, receipt, scipy_ref):
    case_started = time.perf_counter()
    data, W, left_sp, right_sp, left, right, expected, recipe = prepared
    if not isinstance(W, sparse.spmatrix):
        W = sparse.csr_matrix(W)
    graph_policy = recipe.get("graph_topology") or (
        {key: value for key, value in recipe.items()
         if key in ("name", "seed", "density", "degree", "self_loop_policy")}
        if "seed" in recipe else
        {"recipe": recipe.get("name"), "policy": "upstream-prepared CSR preserved"})
    manifest = _manifest(case_name, source, W, left_sp, right_sp, left, right,
                         recipe, graph_policy, receipt)
    case_dir = args.output / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = case_dir / "input-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["manifest_sha256"] = manifest_digest
    typed_manifest = export_shared_finalized_inputs(case_dir / "shared-finalized-inputs",
                                                     W, left, right, manifest)
    results = {}
    cpp_result = None
    if args.native_executable is not None:
        cpp_result = run_cpp_baseline(
            args.native_executable, receipt["_validated"]["native_executable"],
            case_dir, W, left_sp, right_sp, left, right, expected, case_name,
            manifest, manifest_digest, typed_manifest)
    for backend_name in ("direct", "torch") if args.backend == "both" else (args.backend,):
        backend_init_started = time.perf_counter()
        backend = _backend_class(backend_name)(receipt)
        backend_init_ms = (time.perf_counter() - backend_init_started) * 1000
        try:
            provider_result = run_backend_case(
                backend, W, left, right, expected, data, case_dir / backend_name,
                args.warmups if case_name.startswith("synthetic-") else 0,
                args.repeats if case_name.startswith("synthetic-") else 1,
                scipy_ref, manifest_digest, manifest["array_sha256"])
            provider_result["backend_init_ms"] = backend_init_ms
            if cpp_result is not None:
                provider_result["gpu_identity_from_cpp_same_controller_device"] = cpp_result["device_identity"]
            one_resident_ms = provider_result["native_composition"]["wall_stats"]["median_ms"]
            cold_components = {
                "source_loading_graph_preparation_and_dense_materialization_ms": source_preparation_ms(recipe),
                "backend_init_ms": backend_init_ms,
                "input_to_resident_ready_wall_ms": provider_result["input_to_resident_ready_wall_ms"],
                "one_resident_composition_host_wall_ms": one_resident_ms,
                "means_only_materialization_ms": provider_result["materialization"]["means_only"]["host_wall_ms"],
            }
            provider_result["reconstructed_cold_path_estimate_ms"] = sum(cold_components.values())
            provider_result["reconstructed_cold_path_components_ms"] = cold_components
            provider_result["reconstructed_cold_path_definition"] = (
                "sum of measured source read/graph preparation/densification, backend initialization, "
                "the single whole input-to-resident-ready wall interval (which includes validation, "
                "allocation, H2D, and prepared-relation readiness), one host-wall "
                "composition, and means-only materialization; excludes oracle computation, input "
                "export/provenance, warmups, downstream fit, and all-output correctness readback")
            results[backend_name] = provider_result
        finally:
            backend.close()
    return {"case": case_name, "manifest": manifest, "input_preparation_stages_ms": recipe,
            "typed_inputs": typed_manifest,
            "providers": results,
            "cpp_baseline": cpp_result,
            "scipy_reference": scipy_ref,
            "case_runner_wall_ms": (time.perf_counter() - case_started) * 1000,
            "case_runner_wall_scope": "includes C++ parity, providers, output comparisons, and downstream fit; excludes fixture preparation and the shared SciPy baseline"}


def _run_synthetic(case_name, args, receipt):
    prepared = list(_load_case(case_name, args))
    _, W, left_sp, right_sp, _, _, _, _ = prepared
    # One matched SciPy baseline is shared across direct and Torch backends.
    reference = scipy_benchmark(W, left_sp, right_sp, args.warmups, args.repeats)
    prepared[6] = reference.pop("expected")
    dataset_name = prepared[7]["dataset"]
    record = _run_one_case(dataset_name, None, tuple(prepared), args, receipt, reference)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("direct", "torch", "both"), required=True)
    parser.add_argument("--case", choices=("real", "analytic", "compute", "transfer"), required=True)
    parser.add_argument("--output", "--output-root", "--outputroot",
                        dest="output", type=Path, required=True)
    parser.add_argument("--cellerator-root", type=Path, required=True)
    parser.add_argument("--scvelo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cellrank-root", type=Path, required=True)
    parser.add_argument("--build-info", type=Path, required=True)
    parser.add_argument("--native-executable", type=Path,
                        help="Optional receipt-verified ceNativeNeighborhoodMoments parity baseline")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.cellerator_root = args.cellerator_root.resolve()
    args.scvelo_root = args.scvelo_root.resolve()
    args.cellrank_root = args.cellrank_root.resolve()
    if not _inside(Path(__file__).resolve(), args.scvelo_root) \
            or not _inside(Path(harness.__file__).resolve(), args.scvelo_root):
        parser.error("resident harness and imported scientific oracle must be inside --scvelo-root")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/moments-bindings-20261007/environment/numba-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/moments-bindings-20261007/environment/mplconfig")
    receipt = load_build_receipt(args.build_info, args.cellerator_root)
    if args.native_executable is not None:
        try:
            args.native_executable = args.native_executable.resolve()
            receipt["_validated"]["native_executable"] = validate_cpp_executable(
                receipt, args.native_executable)
        except RuntimeError as exc:
            parser.error(str(exc))
    protected_roots = (args.cellerator_root, args.scvelo_root, args.cellrank_root,
                       Path(receipt["_validated"]["package_root"]))
    for source_root in protected_roots:
        if args.output == source_root or source_root in args.output.parents:
            parser.error("--output must be outside all source worktrees and the staged package")
    if args.case in ("compute", "transfer") and (args.warmups != 5 or args.repeats != 20):
        parser.error("synthetic performance cases are fixed to five warmups and twenty measured repetitions")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"schema_version": 1, "build_info": receipt,
                  "repositories": _scvelo_cellrank_provenance(args),
                  "versions": {"python": sys.version, "numpy": np.__version__,
                               "scipy": importlib.import_module("scipy").__version__},
                  "backend_selection": args.backend,
                  "native_executable": receipt.get("_validated", {}).get("native_executable"),
                  "acceptance": harness.GATES}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (args.output / "acceptance.json").write_text(json.dumps(harness.GATES, indent=2, sort_keys=True) + "\n")
    results, errors = {}, []
    try:
        if args.case == "real":
            for name, source, prepared in _real_cases(args):
                results[name] = _run_one_case(name, source, prepared, args, receipt, None)
        elif args.case == "analytic":
            results["analytic"] = _run_one_case("analytic", None, _analytic_case(), args, receipt, None)
        else:
            results[args.case] = _run_synthetic(args.case, args, receipt)
    except Exception as exc:
        errors.append({"case": args.case, "error": str(exc)})
    summary = {"schema_version": 1, "case": args.case, "backend": args.backend,
               "results": results, "errors": errors,
               "all_pass": bool(results) and not errors and all(
                   provider.get("correctness", {}).get("all_pass") is True
                   and provider.get("materialization", {}).get("means_only", {}).get("all_pass") is True
                   and (provider.get("downstream") is None or provider["downstream"].get("all_pass") is True)
                   for item in results.values() for provider in item["providers"].values())
               and all(item.get("cpp_baseline") is None
                       or item["cpp_baseline"].get("correctness", {}).get("all_pass") is True
                       for item in results.values())}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
