#!/usr/bin/env python3
"""Export upstream neighborhood-moment inputs and compare the native probe.

All generated files live beneath --output. Source H5AD files are opened into
memory and never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import gc

import anndata as ad
import numpy as np
import scipy
from scipy import sparse


GATES = {
    "schema_version": 1,
    "declared_before_computation": True,
    "analytic": {"rtol": 1e-6, "atol": 1e-6},
    "raw_moments": {"rtol": 1e-5, "atol": 1e-6},
    "centered_and_adjusted": {
        "absolute_error": "1e-6 + 1e-5 * sum(abs(expression terms))",
        "clip_results": False,
    },
    "downstream": {"rtol": 1e-4, "atol": 1e-5, "identical_gene_mask": True},
    "all_compared_values_finite": True,
    "datasets": ["scvelo-pancreas-100", "scvelo-dentategyrus-100", "cellrank-200"],
}
SCVELO_DATASETS = {
    "scvelo-pancreas-100": "tests/_data/pancreas_100obs_preprocessed.h5ad",
    "scvelo-dentategyrus-100": "tests/_data/dentategyrus_100obs_preprocessed.h5ad",
}
CELLRANK_DATASET = "tests/_ground_truth_adatas/adata_200.h5ad"
FIELDS = (
    "mean_left", "mean_right", "second_left", "cross_second", "second_right",
    "variance_left", "variance_right", "covariance", "affine_second_left",
    "affine_cross_second",
)
SYNTHETIC_GRAPH_TOPOLOGY = {
    "name": "independent_uniform_unique_nonself_neighbors_v1",
    "degree": 32,
    "self_loop_policy": "excluded",
    "row_sampling": "independent uniform sample without replacement per row",
    "csr_edge_order": "ascending column index within each row",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scoped_source_identity(root: Path, relative_paths):
    files = {}
    for relative in relative_paths:
        path = root / relative
        if path.is_file():
            files[relative] = sha256(path)
    paths = list(files)
    status = subprocess.run(["git", "-C", str(root), "status", "--short", "--", *paths],
                            text=True, capture_output=True, check=False).stdout
    diff = subprocess.run(["git", "-C", str(root), "diff", "--binary", "HEAD", "--", *paths],
                          capture_output=True, check=False).stdout
    return {"head": revision(root), "scoped_file_sha256": files,
            "scoped_status": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_content_sha256": hashlib.sha256(
                "".join(f"{k}:{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()}


def provenance(args):
    cellerator_files = (
        "include/Cellerator/compute/operation/native_numeric/device_linear.hh",
        "src/compute/operation/native_numeric/device_linear.cu",
        "src/compute/operation/native_numeric/CMakeLists.txt",
        "include/Cellerator/compute/operation/prepared_relation.hh",
        "include/Cellerator/compute/operation/relation_semantics.hh",
        "src/compute/operation/prepared_relation.cu",
        "src/compute/candidate/sparse/project.cu",
        "examples/native_neighborhood_moments/main.cc",
        "examples/native_neighborhood_moments/CMakeLists.txt",
        "examples/semantic_spine_v1/CMakeLists.txt",
        "tests/native_numeric/device_arithmetic.cc",
    )
    scvelo_files = (
        "probes/native_moments/harness.py", "probes/native_moments/test_harness.py",
        "docs/native-moments-probe.md", "scvelo/preprocessing/moments.py",
        "scvelo/preprocessing/neighbors.py", "scvelo/preprocessing/utils.py",
        "scvelo/tools/_steady_state_model.py",
    )
    cellrank_files = ("src/cellrank/kernels/utils/_moments.py",)
    result = {
        "schema_version": 1,
        "synthetic_graph_generator": SYNTHETIC_GRAPH_TOPOLOGY,
        "repositories": {
            "cellerator": scoped_source_identity(args.cellerator_root, cellerator_files),
            "scvelo": scoped_source_identity(args.scvelo_root, scvelo_files),
            "cellrank": scoped_source_identity(args.cellrank_root, cellrank_files),
        },
        "native_executable": {"path": args.native, "sha256": sha256(Path(args.native))},
        "build_info": None,
    }
    if args.build_info:
        result["build_info"] = json.loads(args.build_info.read_text())
        build = result["build_info"]
        binary_digest = result["native_executable"]["sha256"]
        if build.get("native_binary_sha256") and build["native_binary_sha256"] != binary_digest:
            raise RuntimeError("Build receipt native binary hash does not match --native")
        if build.get("source_head") and build["source_head"] != result["repositories"]["cellerator"]["head"]:
            raise RuntimeError("Build receipt Cellerator HEAD does not match --cellerator-root")
        if build.get("source_root") and Path(build["source_root"]).resolve() != args.cellerator_root:
            raise RuntimeError("Build receipt Cellerator source root does not match --cellerator-root")
        mismatched = [name for name, digest in build.get("source_file_sha256", {}).items()
                      if name in result["repositories"]["cellerator"]["scoped_file_sha256"]
                      and digest != result["repositories"]["cellerator"]["scoped_file_sha256"][name]]
        if mismatched:
            raise RuntimeError(f"Build receipt source hashes differ from live scoped files: {mismatched}")
    return result


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def import_cellrank_moments(root: Path):
    path = root / "src/cellrank/kernels/utils/_moments.py"
    spec = importlib.util.spec_from_file_location("probe_cellrank_moments", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CellRank helper at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def sparse_composition(W, left_sp, right_sp, mean_left=None, mean_right=None):
    mean_left = np.asarray((W @ left_sp).toarray(), dtype=np.float32) if mean_left is None else mean_left
    mean_right = np.asarray((W @ right_sp).toarray(), dtype=np.float32) if mean_right is None else mean_right
    second_left = np.asarray((W @ left_sp.multiply(left_sp)).toarray(), dtype=np.float32)
    cross_second = np.asarray((W @ left_sp.multiply(right_sp)).toarray(), dtype=np.float32)
    second_right = np.asarray((W @ right_sp.multiply(right_sp)).toarray(), dtype=np.float32)
    return {
        "mean_left": np.asarray(mean_left, dtype=np.float32),
        "mean_right": np.asarray(mean_right, dtype=np.float32),
        "second_left": second_left,
        "cross_second": cross_second,
        "second_right": second_right,
        "variance_left": second_left - mean_left * mean_left,
        "variance_right": second_right - mean_right * mean_right,
        "covariance": cross_second - mean_left * mean_right,
        "affine_second_left": 2 * second_left - mean_left,
        "affine_cross_second": 2 * cross_second - mean_right,
    }


def prepare_scvelo(root: Path, fixture: Path):
    sys.path.insert(0, str(root))
    import scvelo as scv
    from scvelo.preprocessing.neighbors import get_connectivities
    from scvelo.preprocessing.moments import second_order_moments, second_order_moments_u

    stage = time.perf_counter()
    data = ad.read_h5ad(fixture)
    source_read_ms = (time.perf_counter() - stage) * 1000
    # Use the normal upstream wrapper on an in-memory AnnData copy; this also
    # ensures the exported fields reflect any required upstream normalization.
    stage = time.perf_counter()
    scv.pp.moments(data, n_neighbors=None, use_highly_variable=False)
    first_moments_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    W = get_connectivities(data, recurse_neighbors=False).tocsr().astype(np.float32)
    left_sp = sparse.csr_matrix(data.layers["spliced"], dtype=np.float32)
    right_sp = sparse.csr_matrix(data.layers["unspliced"], dtype=np.float32)
    graph_and_sparse_fields_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    left = np.asarray(left_sp.toarray(), dtype=np.float32, order="C")
    right = np.asarray(right_sp.toarray(), dtype=np.float32, order="C")
    dense_materialization_ms = (time.perf_counter() - stage) * 1000
    mean_left = np.asarray(data.layers["Ms"], dtype=np.float32)
    mean_right = np.asarray(data.layers["Mu"], dtype=np.float32)
    expected = sparse_composition(W, left_sp, right_sp, mean_left, mean_right)
    # Compare raw second moments with scVelo's current public implementations.
    stage = time.perf_counter()
    mss, mus = second_order_moments(data, adjusted=False)
    muu = second_order_moments_u(data)
    second_moments_ms = (time.perf_counter() - stage) * 1000
    expected.update({"second_left": np.asarray(mss, dtype=np.float32),
                     "cross_second": np.asarray(mus, dtype=np.float32),
                     "second_right": np.asarray(muu, dtype=np.float32)})
    recipe = {"name": "scvelo.pp.moments + get_connectivities(recurse_neighbors=False)",
              "scvelo_version": getattr(scv, "__version__", "unknown"),
              "source_read_ms": source_read_ms,
              "upstream_first_moments_ms": first_moments_ms,
              "second_moments_ms": second_moments_ms,
              "graph_and_sparse_field_preparation_ms": graph_and_sparse_fields_ms,
              "dense_input_materialization_ms": dense_materialization_ms,
              "layer_dtypes_after_reference_preparation": {
                  key: str(data.layers[key].dtype) for key in ("spliced", "unspliced")}}
    adjusted_left, adjusted_cross = second_order_moments(data, adjusted=True)
    expected["variance_left"] = expected["second_left"] - mean_left * mean_left
    expected["variance_right"] = expected["second_right"] - mean_right * mean_right
    expected["covariance"] = expected["cross_second"] - mean_left * mean_right
    expected["affine_second_left"] = 2 * expected["second_left"] - mean_left
    expected["affine_cross_second"] = 2 * expected["cross_second"] - mean_right
    recipe["adjusted_reference_max_abs_error"] = {
        "second_left": float(np.abs(np.asarray(adjusted_left) - expected["affine_second_left"]).max(initial=0)),
        "cross_second": float(np.abs(np.asarray(adjusted_cross) - expected["affine_cross_second"]).max(initial=0)),
    }
    expected["mean_left"], expected["mean_right"] = mean_left, mean_right
    return data, W, left_sp, right_sp, left, right, expected, recipe


def prepare_cellrank(root: Path, fixture: Path):
    helper, helper_path = import_cellrank_moments(root)
    stage = time.perf_counter()
    data = ad.read_h5ad(fixture)
    source_read_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    W = helper._row_normalize_connectivities(data.obsp["connectivities"])
    left_sp = sparse.csr_matrix(data.layers["spliced"], dtype=np.float32)
    right_sp = sparse.csr_matrix(data.layers["unspliced"], dtype=np.float32)
    graph_and_sparse_fields_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    left = np.asarray(left_sp.toarray(), dtype=np.float32, order="C")
    right = np.asarray(right_sp.toarray(), dtype=np.float32, order="C")
    dense_materialization_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    mean_left, var_left = helper._knn_moments(W, left)
    mean_right, var_right = helper._knn_moments(W, right)
    cellrank_helper_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    expected = sparse_composition(W, left_sp, right_sp, mean_left, mean_right)
    second_moments_ms = (time.perf_counter() - stage) * 1000
    expected["variance_left"] = np.asarray(var_left, dtype=np.float32)
    expected["variance_right"] = np.asarray(var_right, dtype=np.float32)
    velocity_dtype = str(data.layers["velocity"].dtype) if "velocity" in data.layers else "absent"
    recipe = {
        "name": "CellRank _row_normalize_connectivities + _knn_moments",
        "source_read_ms": source_read_ms,
        "graph_and_sparse_field_preparation_ms": graph_and_sparse_fields_ms,
        "cellrank_mean_variance_helper_ms": cellrank_helper_ms,
        "sparse_second_moments_ms": second_moments_ms,
        "dense_input_materialization_ms": dense_materialization_ms,
        "helper_path": str(helper_path), "helper_sha256": sha256(helper_path),
        "layer_dtypes": {key: str(data.layers[key].dtype) for key in ("spliced", "unspliced")},
        "velocity_dtype_preserved": velocity_dtype,
        "fp64_velocity_native_followup": velocity_dtype == "float64",
    }
    return data, W.tocsr(), left_sp, right_sp, left, right, expected, recipe


def write_inputs(directory: Path, W, left_sp, right_sp, left, right, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    csr = W.tocsr(copy=True)
    np.savez_compressed(
        directory / "reference_sparse.npz",
        w_data=csr.data.astype(np.float32), w_indices=csr.indices.astype(np.uint32),
        w_indptr=csr.indptr.astype(np.uint32), w_shape=np.asarray(csr.shape, np.uint32),
        left_data=left_sp.data.astype(np.float32), left_indices=left_sp.indices.astype(np.uint32),
        left_indptr=left_sp.indptr.astype(np.uint32), right_data=right_sp.data.astype(np.float32),
        right_indices=right_sp.indices.astype(np.uint32), right_indptr=right_sp.indptr.astype(np.uint32),
        feature_shape=np.asarray(left.shape, np.uint32),
    )
    arrays = {
        "row_offsets.u32": csr.indptr.astype("<u4", copy=False),
        "column_indices.u32": csr.indices.astype("<u4", copy=False),
        "weights.f32": csr.data.astype("<f4", copy=False),
        "left.f32": np.ascontiguousarray(left, dtype="<f4"),
        "right.f32": np.ascontiguousarray(right, dtype="<f4"),
    }
    for name, value in arrays.items():
        value.tofile(directory / name)
    manifest.update({
        "shape": {"sources": int(left.shape[0]), "destinations": int(csr.shape[0]),
                  "features": int(left.shape[1]), "edges": int(csr.nnz)},
        "csr_order": "source CSR edge order preserved; native and SciPy paths consume the same arrays",
        "input_files": {name: {"sha256": sha256(directory / name),
                                "bytes": (directory / name).stat().st_size} for name in arrays},
        "array_sha256": {"left": array_sha256(left), "right": array_sha256(right),
                         "weights": array_sha256(csr.data.astype(np.float32))},
        "sparse_reference_sha256": sha256(directory / "reference_sparse.npz"),
    })
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_native_metrics(metrics, args, input_dir: Path, shape):
    required = ("schema_version", "dimensions", "runs", "native_policy", "gpu",
                "resident_ms", "resident_wall_ms", "preparation_ms", "allocation_ms",
                "input_read_ms", "h2d_ms", "d2h_ms", "output_write_ms",
                "process_total_ms", "memory_bytes", "preparation_report")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"metrics.json lacks required contract fields: {missing}")
    if metrics["schema_version"] != 1:
        raise RuntimeError(f"Unsupported metrics schema_version: {metrics['schema_version']}")
    if len(metrics["resident_ms"]) != args.repeats or len(metrics["resident_wall_ms"]) != args.repeats:
        raise RuntimeError("Native resident timing sample counts must equal --repeats")
    dimensions = metrics["dimensions"]
    for key, value in zip(("sources", "destinations", "features", "edges"), shape):
        if dimensions.get(key) != value:
            raise RuntimeError(f"Native metrics {key}={dimensions.get(key)} but requested {value}")
    if metrics["runs"].get("warmups") != args.warmups or metrics["runs"].get("repeats") != args.repeats:
        raise RuntimeError("Native metrics warmup/repeat counts do not match the invocation")
    policy = metrics["native_policy"]
    for key in ("relation", "input", "multiply", "accumulation", "output"):
        if policy.get(key) != "FP32":
            raise RuntimeError(f"Native arithmetic policy {key} must be FP32, got {policy.get(key)}")
    if policy.get("tensor_cores") is not False:
        raise RuntimeError("Native baseline must run with Tensor Cores disabled")
    memory = metrics["memory_bytes"]
    prep = metrics["preparation_report"]
    for key in ("tracked_device_allocations", "host_input_files", "host_output_files",
                "native_process_peak_rss"):
        if not isinstance(memory.get(key), int) or memory[key] < 0:
            raise RuntimeError(f"Native metrics memory_bytes.{key} must be a nonnegative integer")
    metrics["memory_bytes"]["native_process_peak_rss_bytes"] = memory["native_process_peak_rss"]
    input_bytes = sum(path.stat().st_size for path in input_dir.iterdir() if path.is_file()
                      and path.name.endswith((".u32", ".f32")))
    output_bytes = shape[1] * shape[2] * 4 * len(FIELDS)
    if memory["host_input_files"] != input_bytes or memory["host_output_files"] != output_bytes:
        raise RuntimeError("Native host input/output byte counts do not match exported raw arrays")
    if prep.get("topology_preparations") != 1 or prep.get("value_refreshes") != 1:
        raise RuntimeError("Native probe must prepare topology and publish values exactly once")
    expected_launches = 5 * (args.warmups + args.repeats + 1)  # one separately timed stage diagnostic
    if prep.get("accepted_forward_launches") != expected_launches:
        raise RuntimeError(f"Native accepted forward launches={prep.get('accepted_forward_launches')}; expected {expected_launches}")
    return metrics


def invoke_native(args, input_dir: Path, result_dir: Path, shape):
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [args.native, "--input", str(input_dir), "--output", str(result_dir),
               "--sources", str(shape[0]), "--destinations", str(shape[1]),
               "--features", str(shape[2]), "--edges", str(shape[3]),
               "--warmups", str(args.warmups), "--repeats", str(args.repeats)]
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/moments-probe-20261006/numba")
    env.setdefault("MPLCONFIGDIR", "/tmp/moments-probe-20261006/mpl")
    start = time.perf_counter()
    proc = subprocess.run(command, text=True, capture_output=True, env=env)
    subprocess_wall_ms = (time.perf_counter() - start) * 1000
    (result_dir / "native.stdout.txt").write_text(proc.stdout)
    (result_dir / "native.stderr.txt").write_text(proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"Native probe exited {proc.returncode}: {proc.stderr[-4000:]}")
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError("Native probe completed without metrics.json")
    metrics = validate_native_metrics(json.loads(metrics_path.read_text()), args, input_dir, shape)
    metrics["python_subprocess_wall_ms"] = subprocess_wall_ms
    return metrics


def read_outputs(directory: Path, shape):
    rows, cols = shape
    outputs = {}
    for name in FIELDS:
        path = directory / f"{name}.f32"
        expected_bytes = rows * cols * 4
        if not path.exists() or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Invalid {name} output size; expected {expected_bytes} bytes")
        outputs[name] = np.fromfile(path, dtype="<f4").reshape(rows, cols)
    return outputs


def compare_field(name, actual, expected):
    centered = {"variance_left", "variance_right", "covariance",
                "affine_second_left", "affine_cross_second"}
    a = np.asarray(actual, dtype=np.float32)
    e = np.asarray(expected[name], dtype=np.float32)
    finite = bool(np.isfinite(a).all() and np.isfinite(e).all())
    delta = np.abs(a.astype(np.float64) - e.astype(np.float64))
    if not finite:
        return {"pass": False, "reason": "non-finite values"}
    if name in centered:
        if name == "variance_left": terms = (expected["second_left"], expected["mean_left"] ** 2)
        elif name == "variance_right": terms = (expected["second_right"], expected["mean_right"] ** 2)
        elif name == "covariance": terms = (expected["cross_second"], expected["mean_left"] * expected["mean_right"])
        elif name == "affine_second_left": terms = (2 * expected["second_left"], expected["mean_left"])
        else: terms = (2 * expected["cross_second"], expected["mean_right"])
        bound = 1e-6 + 1e-5 * sum(np.abs(np.asarray(term, dtype=np.float64)) for term in terms)
        return {"pass": bool(np.all(delta <= bound)),
                "max_abs_error": float(delta.max(initial=0)),
                "max_allowed_error": float(bound.max(initial=0)),
                "near_zero_count": int(np.count_nonzero(np.abs(e) <= 1e-6))}
    nonzero = np.abs(e) > 1e-6
    return {"pass": bool(np.allclose(a, e, rtol=1e-5, atol=1e-6)),
            "max_abs_error": float(delta.max(initial=0)),
            "max_relative_error_nonzero": float((delta[nonzero] / np.abs(e[nonzero])).max(initial=0)),
            "near_zero_count": int(np.count_nonzero(np.abs(e) <= 1e-6))}


def compare(actual, expected, dataset):
    fields = {name: compare_field(name, actual[name], expected) for name in FIELDS}
    return {"dataset": dataset, "all_pass": all(item["pass"] for item in fields.values()), "fields": fields}


def compare_native_files(directory: Path, expected, shape, dataset):
    """Compare one output at a time to bound memory in the large case."""
    fields = {}
    for name in FIELDS:
        path = directory / f"{name}.f32"
        wanted_bytes = shape[0] * shape[1] * 4
        if not path.exists() or path.stat().st_size != wanted_bytes:
            fields[name] = {"pass": False, "reason": f"wrong output byte count; wanted {wanted_bytes}"}
            continue
        actual = np.fromfile(path, dtype="<f4").reshape(shape)
        fields[name] = compare_field(name, actual, expected)
        del actual
        gc.collect()
    return {"dataset": dataset, "all_pass": all(item["pass"] for item in fields.values()), "fields": fields}


def downstream(data, native, expected):
    from scvelo.tools._steady_state_model import SteadyStateModel

    def fit(ms, mu):
        copy = data.copy()
        copy.layers["Ms"] = np.asarray(ms, dtype=np.float32)
        copy.layers["Mu"] = np.asarray(mu, dtype=np.float32)
        model = SteadyStateModel(copy, use_highly_variable=False, r2_adjusted=False,
                                 fit_offset=False, perc=(5, 95))
        model.fit()
        # fit() stores the deterministic residual on the model instance.
        # get_velocity() currently reads state_dict().residual, which remains
        # None because SteadyStateModel.fit() has a TODO to populate that field.
        return model.state_dict(), np.asarray(model.residual)

    ref, ref_velocity = fit(expected["mean_left"], expected["mean_right"])
    got, got_velocity = fit(native["mean_left"], native["mean_right"])
    fields = {}
    for key in ("gamma", "offset", "r2"):
        a, e = np.asarray(got[key]), np.asarray(ref[key])
        both_finite = np.isfinite(a) & np.isfinite(e)
        same_nonfinite = np.array_equal(np.isnan(a), np.isnan(e)) and np.array_equal(np.isinf(a), np.isinf(e))
        fields[key] = {"pass": bool(same_nonfinite and np.allclose(a[both_finite], e[both_finite], rtol=1e-4, atol=1e-5)),
                       "nonfinite_reference_count": int(np.count_nonzero(~np.isfinite(e))),
                       "max_abs_error_finite": float(np.abs(a[both_finite] - e[both_finite]).max(initial=0))}
    a, e = np.asarray(got_velocity), np.asarray(ref_velocity)
    fields["residual"] = {"pass": bool(np.isfinite(a).all() and np.isfinite(e).all() and np.allclose(a, e, rtol=1e-4, atol=1e-5)),
                          "max_abs_error": float(np.abs(a - e).max(initial=0))}
    fields["velocity_genes"] = {"pass": bool(np.array_equal(got["velocity_genes"], ref["velocity_genes"])),
                                "mismatches": int(np.count_nonzero(got["velocity_genes"] != ref["velocity_genes"]))}
    return {"all_pass": all(item["pass"] for item in fields.values()), "fields": fields}


def run_real_dataset(args, name, source, prepare):
    dest = args.output / name
    started = time.perf_counter()
    fixture_hash = sha256(source)
    data, W, left_sp, right_sp, left, right, expected, recipe = prepare(source)
    preprocess_ms = (time.perf_counter() - started) * 1000
    manifest = {
        "schema_version": 1, "dataset": name, "fixture": str(source),
        "fixture_sha256": fixture_hash, "recipe": recipe,
        "provenance_sha256": args.provenance_sha256,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "anndata": ad.__version__},
        "platform": platform.platform(),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "reference_loading_and_preprocess_ms": preprocess_ms,
    }
    export_start = time.perf_counter()
    write_inputs(dest / "input", W, left_sp, right_sp, left, right, manifest)
    manifest["input_export_and_provenance_write_ms"] = (time.perf_counter() - export_start) * 1000
    (dest / "input" / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    metrics = invoke_native(args, dest / "input", dest / "native",
                            (left.shape[0], W.shape[0], left.shape[1], W.nnz))
    actual = read_outputs(dest / "native", (W.shape[0], left.shape[1]))
    result = {"correctness": compare(actual, expected, name),
              "downstream": downstream(data, actual, expected), "native_metrics": metrics}
    (dest / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def analytic_self_test(native):
    proc = subprocess.run([native, "--self-test"], text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"native --self-test failed: {proc.stderr or proc.stdout}")
    return {"pass": True, "stdout": proc.stdout.strip()}


def independent_csr_relation(rng, n, degree):
    if n <= degree or degree <= 0:
        raise ValueError("independent non-self graph requires 0 < degree < n")
    row_ids = np.arange(n, dtype=np.int32)
    rows = np.repeat(row_ids, degree)
    # Sample compact column IDs from N-1 choices, then skip the diagonal for
    # each row. Reject only within-row duplicates; each row is sampled anew.
    compact = rng.integers(0, n - 1, size=(n, degree), dtype=np.int32)
    while True:
        order = np.argsort(compact, axis=1)
        sorted_ids = np.take_along_axis(compact, order, axis=1)
        duplicates = sorted_ids[:, 1:] == sorted_ids[:, :-1]
        duplicate_rows, duplicate_positions = np.nonzero(duplicates)
        if duplicate_rows.size == 0:
            break
        replace_positions = order[duplicate_rows, duplicate_positions + 1]
        compact[duplicate_rows, replace_positions] = rng.integers(
            0, n - 1, size=duplicate_rows.size, dtype=np.int32
        )
    cols = (compact + (compact >= row_ids[:, None])).reshape(-1)
    weights = rng.random(n * degree, dtype=np.float32)
    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float32)
    W.sort_indices()
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    W = (sparse.diags(1.0 / row_sum) @ W).tocsr().astype(np.float32)
    W.sort_indices()
    return W


def synthetic(seed, n, features, degree, density):
    rng = np.random.default_rng(seed)
    stage = time.perf_counter()
    W = independent_csr_relation(rng, n, degree)
    graph_ms = (time.perf_counter() - stage) * 1000

    def make_field():
        mask = rng.random((n, features)) < density
        dense = (rng.normal(size=(n, features)).astype(np.float32) * mask)
        result = sparse.csr_matrix(dense)
        del dense, mask
        return result

    stage = time.perf_counter()
    left_sp, right_sp = make_field(), make_field()
    sparse_fields_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    left, right = left_sp.toarray(), right_sp.toarray()
    dense_ms = (time.perf_counter() - stage) * 1000
    return W, left_sp, left, right_sp, right, {
        "graph_generation_and_normalization_ms": graph_ms,
        "sparse_feature_source_generation_ms": sparse_fields_ms,
        "sparse_to_dense_materialization_ms": dense_ms,
    }


def run_benchmark(args, spec):
    name, n, features, density, seed = spec
    dest = args.output / name
    dest.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    W, left_sp, left, right_sp, right, generation = synthetic(seed, n, features, 32, density)
    generated_ms = (time.perf_counter() - start) * 1000
    export_start = time.perf_counter()
    write_inputs(dest / "input", W, left_sp, right_sp, left, right,
                 {"schema_version": 1, "dataset": name, "synthetic": True,
                  "seed": seed, "degree_requested": 32, "density": density,
                  "graph_topology": SYNTHETIC_GRAPH_TOPOLOGY,
                  "provenance_sha256": args.provenance_sha256})
    export_ms = (time.perf_counter() - export_start) * 1000
    scipy_warmups, scipy_repeats = [], []
    expected = None
    for i in range(args.warmups + args.repeats):
        lap = time.perf_counter()
        expected = sparse_composition(W, left_sp, right_sp)
        elapsed = (time.perf_counter() - lap) * 1000
        (scipy_warmups if i < args.warmups else scipy_repeats).append(elapsed)
        if i >= args.warmups:
            gc.collect()
    scipy_ms = float(np.median(scipy_repeats)) if scipy_repeats else 0.0
    scipy_single_start = time.perf_counter()
    expected = sparse_composition(W, left_sp, right_sp)
    scipy_single_ms = (time.perf_counter() - scipy_single_start) * 1000
    metrics = invoke_native(args, dest / "input", dest / "native", (n, n, features, W.nnz))
    native_event_median = float(np.median(metrics["resident_ms"]))
    native_wall_median = float(np.median(metrics["resident_wall_ms"]))
    one_iteration_components = {
        "input_read_ms": metrics["input_read_ms"],
        "device_allocation_ms": metrics["allocation_ms"],
        "relation_preparation_ms": metrics["preparation_ms"],
        "host_to_device_ms": metrics["h2d_ms"],
        "resident_composition_event_median_ms": native_event_median,
        "device_to_host_ms": metrics["d2h_ms"],
        "output_write_ms": metrics["output_write_ms"],
    }
    result = {
        "correctness": compare_native_files(dest / "native", expected, (n, features), name), "native_metrics": metrics,
        "scipy_composition_median_ms": scipy_ms, "scipy_resident_ms": scipy_repeats,
        "scipy_warmup_ms": scipy_warmups, "scipy_single_composition_ms": scipy_single_ms,
        "synthetic_generation_stages_ms": generation,
        "artifact_export_and_provenance_write_ms": export_ms,
        "synthetic_generation_end_to_end_ms": generated_ms,
        "harness_process_wall_ms_including_native_warmups_and_repeats": (time.perf_counter() - start) * 1000,
        "python_process_peak_rss_bytes_cumulative": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "native_single_iteration_estimate_ms": float(sum(one_iteration_components.values())),
        "native_single_iteration_estimate_components_ms": one_iteration_components,
        "native_single_iteration_estimate_definition": "input read + allocation + relation preparation + H2D + median resident composition event time + D2H + output write; excludes stage diagnostics and Python launch overhead",
        "eligible_timing_ratios": {
            "scipy_median_over_native_resident_event_median": scipy_ms / native_event_median if native_event_median else None,
            "scipy_median_over_native_resident_host_wall_median": scipy_ms / native_wall_median if native_wall_median else None,
            "scope": "complete repeated composition only; not end-to-end speedup",
        },
        "host_dense_input_bytes": int(left.nbytes + right.nbytes),
        "host_sparse_input_bytes": int(sum(a.nbytes for x in (left_sp, right_sp)
                                                   for a in (x.data, x.indices, x.indptr))),
        "degree_after_duplicate_coalescing": int(W.nnz // n),
        "feature_density_observed": {"left": float(left_sp.nnz / (n * features)),
                                      "right": float(right_sp.nnz / (n * features))},
    }
    (dest / "benchmark.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def acceptance_passes(summary):
    if summary.get("errors"):
        return False
    if summary.get("analytic") is not None and not summary["analytic"].get("pass", False):
        return False
    entries = [*summary.get("datasets", {}).values(), *summary.get("benchmarks", {}).values()]
    return all(entry["correctness"]["all_pass"]
               and ("downstream" not in entry or entry["downstream"]["all_pass"])
               for entry in entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scvelo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cellrank-root", type=Path, default=Path("/home/tumlinson/accelerated-ports/CellRank"))
    parser.add_argument("--cellerator-root", type=Path, default=Path("/home/tumlinson/Cellerator"))
    parser.add_argument("--build-info", type=Path,
                        help="Optional JSON receipt with CMake options and CUDA toolchain versions")
    parser.add_argument("--mode", choices=("correctness", "benchmarks", "full"), default="correctness")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--skip-analytic", action="store_true")
    parser.add_argument("--only", action="append", help="Dataset/benchmark key; repeatable")
    args = parser.parse_args()
    args.native = str(Path(args.native).resolve())
    args.output = args.output.resolve()
    args.scvelo_root = args.scvelo_root.resolve()
    args.cellrank_root = args.cellrank_root.resolve()
    args.cellerator_root = args.cellerator_root.resolve()
    args.build_info = args.build_info.resolve() if args.build_info else None
    args.output.mkdir(parents=True, exist_ok=True)
    # This file is committed before any computations are launched.
    (args.output / "acceptance.json").write_text(json.dumps(GATES, indent=2, sort_keys=True) + "\n")
    provenance_start = time.perf_counter()
    args.provenance = provenance(args)
    args.provenance_sha256 = hashlib.sha256(
        json.dumps(args.provenance, sort_keys=True).encode()).hexdigest()
    args.provenance_collection_ms = (time.perf_counter() - provenance_start) * 1000
    provenance_write_start = time.perf_counter()
    (args.output / "provenance.json").write_text(json.dumps(args.provenance, indent=2, sort_keys=True) + "\n")
    summary = {"acceptance": "acceptance.json", "datasets": {}, "benchmarks": {}, "errors": []}
    summary["setup_cost"] = {"provenance_collection_ms": args.provenance_collection_ms,
                             "provenance_write_ms": (time.perf_counter() - provenance_write_start) * 1000,
                             "build_info_path": str(args.build_info) if args.build_info else None}
    if not args.skip_analytic:
        try:
            summary["analytic"] = analytic_self_test(args.native)
        except Exception as exc:
            summary["errors"].append({"stage": "analytic", "error": str(exc)})
    if args.mode in ("correctness", "full"):
        cases = [(key, args.scvelo_root / rel, lambda path: prepare_scvelo(args.scvelo_root, path))
                 for key, rel in SCVELO_DATASETS.items()]
        cases.append(("cellrank-200", args.cellrank_root / CELLRANK_DATASET,
                      lambda path: prepare_cellrank(args.cellrank_root, path)))
        for name, source, prepare in cases:
            if args.only and name not in args.only:
                continue
            try:
                summary["datasets"][name] = run_real_dataset(args, name, source, prepare)
            except Exception as exc:
                summary["errors"].append({"dataset": name, "error": str(exc),
                                          "traceback": traceback.format_exc()})
    if args.mode in ("benchmarks", "full"):
        specs = (("synthetic-compute-8192x2048", 8192, 2048, .4, 20261006),
                 ("synthetic-transfer-32768x2048", 32768, 2048, .25, 20261007))
        for spec in specs:
            if args.only and spec[0] not in args.only:
                continue
            try:
                summary["benchmarks"][spec[0]] = run_benchmark(args, spec)
            except Exception as exc:
                summary["errors"].append({"benchmark": spec[0], "error": str(exc),
                                          "traceback": traceback.format_exc()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if acceptance_passes(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
