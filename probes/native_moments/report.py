#!/usr/bin/env python3
"""Aggregate preserved correctness and final benchmark artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


FIXTURES = ("scvelo-pancreas-100", "scvelo-dentategyrus-100", "cellrank-200")
BENCHMARKS = ("synthetic-compute-8192x2048", "synthetic-transfer-32768x2048")
RAW = ("mean_left", "mean_right", "second_left", "cross_second", "second_right")
CENTERED = ("variance_left", "variance_right", "covariance", "affine_second_left", "affine_cross_second")


def read(path, missing):
    if not path.is_file():
        missing.append(f"missing {path}")
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        missing.append(f"cannot read {path}: {exc}")
        return None


def finite_numbers(values):
    return isinstance(values, list) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values)


def timing_stats(values):
    return {"count": len(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values),
            "stddev_ms": statistics.pstdev(values),
            "coefficient_of_variation": statistics.pstdev(values) / statistics.mean(values)
            if statistics.mean(values) else None}


def fixture_report(root, name, summary, missing):
    saved = summary.get("datasets", {}).get(name)
    comparison = read(root / name / "comparison.json", missing)
    manifest = read(root / name / "input/manifest.json", missing)
    metrics = read(root / name / "native/metrics.json", missing)
    if saved is None:
        missing.append(f"correctness summary lacks dataset {name}")
    if not all((comparison, manifest, metrics, saved)):
        return {"dataset": name, "status": "incomplete"}
    correctness = comparison["correctness"]
    downstream = comparison.get("downstream", {})
    fields = correctness.get("fields", {})
    field_report = {key: fields.get(key, {"pass": False, "missing": True}) for key in (*RAW, *CENTERED)}
    required_pass = correctness.get("all_pass") is True and all(x.get("pass") is True for x in field_report.values())
    required_pass = required_pass and saved.get("correctness", {}).get("all_pass") is True
    downstream_pass = downstream.get("all_pass") is True
    return {
        "dataset": name, "status": "pass" if required_pass and downstream_pass else "fail",
        "fixture": manifest.get("fixture"), "fixture_sha256": manifest.get("fixture_sha256"),
        "shape": manifest.get("shape"), "recipe": manifest.get("recipe"),
        "raw_and_centered_fields": field_report,
        "downstream": downstream,
        "native_policy": metrics.get("native_policy"), "gpu": metrics.get("gpu"),
        "native_process_peak_rss_bytes": metrics.get("memory_bytes", {}).get("native_process_peak_rss"),
        "example_owned_device_allocation_bytes": metrics.get("memory_bytes", {}).get("tracked_device_allocations"),
        "example_allocation_scope": "resident vectors and external relation-values buffer allocated by the example; excludes prepared-relation internal allocations",
        "prepared_relation_memory_report": {
            "shared_structural_bytes": metrics.get("preparation_report", {}).get("shared_structural_bytes"),
            "instance_value_bytes": metrics.get("preparation_report", {}).get("instance_value_bytes"),
            "scope": "Cellerator preparation report counters; reported separately and not added to example-owned bytes",
        },
        "provenance_sha256": manifest.get("provenance_sha256"),
        "input_manifest_file_sha256": hashlib.sha256((root / name / "input/manifest.json").read_bytes()).hexdigest(),
        "cellrank_velocity_dtype_preserved": manifest.get("recipe", {}).get("velocity_dtype_preserved")
        if name == "cellrank-200" else None,
    }


def benchmark_report(root, name, final_summary, missing):
    saved = final_summary.get("benchmarks", {}).get(name)
    bench = read(root / name / "benchmark.json", missing)
    manifest = read(root / name / "input/manifest.json", missing)
    if saved is None:
        missing.append(f"final summary lacks benchmark {name}")
    if not bench or not manifest or saved is None:
        return {"benchmark": name, "status": "incomplete"}
    metrics = bench.get("native_metrics", {})
    native_events = metrics.get("resident_ms", [])
    native_walls = metrics.get("resident_wall_ms", [])
    cpu = bench.get("scipy_resident_ms", [])
    expected_topology = manifest.get("graph_topology", {})
    complete = (metrics.get("runs", {}).get("warmups") == 5
                and metrics.get("runs", {}).get("repeats") == 20
                and len(native_events) == len(native_walls) == len(cpu) == 20
                and finite_numbers(native_events) and finite_numbers(native_walls) and finite_numbers(cpu)
                and saved.get("correctness", {}).get("all_pass") is bench.get("correctness", {}).get("all_pass")
                and expected_topology.get("name") == "independent_uniform_unique_nonself_neighbors_v1"
                and expected_topology.get("self_loop_policy") == "excluded")
    if not complete:
        missing.append(f"final benchmark {name} is incomplete or has an unexpected graph/timing policy")
    return {
        "benchmark": name, "status": "pass" if complete and bench.get("correctness", {}).get("all_pass") else "fail",
        "shape": manifest.get("shape"), "seed": manifest.get("seed"),
        "feature_density_requested": manifest.get("density"),
        "feature_density_observed": bench.get("feature_density_observed"),
        "graph_topology": expected_topology,
        "native_policy": metrics.get("native_policy"), "gpu": metrics.get("gpu"),
        "correctness": bench.get("correctness"),
        "scipy_sparse_composition": timing_stats(cpu) if finite_numbers(cpu) and cpu else None,
        "native_resident_event_composition": timing_stats(native_events) if finite_numbers(native_events) and native_events else None,
        "native_resident_host_wall_composition": timing_stats(native_walls) if finite_numbers(native_walls) and native_walls else None,
        "eligible_resident_only_ratios": bench.get("eligible_timing_ratios"),
        "cost_stages_ms": {
            "synthetic_generation": bench.get("synthetic_generation_stages_ms"),
            "artifact_export_and_manifest": bench.get("artifact_export_and_provenance_write_ms"),
            "native_single_iteration_estimate": bench.get("native_single_iteration_estimate_components_ms"),
            "native_single_iteration_estimate_total": bench.get("native_single_iteration_estimate_ms"),
            "native_multirepeat_process_total": metrics.get("process_total_ms"),
            "native_timing_scope": metrics.get("timing_scope"),
            "diagnostic_stage_ms": metrics.get("diagnostic_stage_ms"),
        },
        "memory_bytes": {"example_owned_device_allocation_bytes": metrics.get("memory_bytes", {}).get("tracked_device_allocations"),
                         "example_allocation_scope": "resident vectors and external relation-values buffer; excludes prepared-relation internal allocations",
                         "prepared_relation_shared_structural_bytes": metrics.get("preparation_report", {}).get("shared_structural_bytes"),
                         "prepared_relation_instance_value_bytes": metrics.get("preparation_report", {}).get("instance_value_bytes"),
                         "prepared_relation_counter_scope": "reported separately; not included in example-owned allocation counter",
                         "native_process_peak_rss": metrics.get("memory_bytes", {}).get("native_process_peak_rss"),
                         "python_process_peak_rss_cumulative": bench.get("python_process_peak_rss_bytes_cumulative"),
                         "host_dense_inputs": bench.get("host_dense_input_bytes"),
                         "host_sparse_inputs": bench.get("host_sparse_input_bytes")},
        "process_total_scope_note": "Includes native warmups, measured repetitions, and a diagnostic iteration; it is not a one-shot measurement.",
    }


def old_structured_attempt(root):
    candidates = (root.parent / "structured-circulant-attempt", root / "structured-circulant-attempt")
    old = next((p for p in candidates if p.is_dir()), None)
    if old is None:
        return {"status": "not_found", "included_in_final_results": False}
    cases = {}
    for name in BENCHMARKS:
        path = old / name / "benchmark.json"
        try:
            b = json.loads(path.read_text())
            cases[name] = {"correctness_pass": b.get("correctness", {}).get("all_pass"),
                           "resident_only_ratios": b.get("eligible_timing_ratios"),
                           "topology": "circulant shared-offset attempt; superseded and excluded"}
        except (OSError, json.JSONDecodeError):
            cases[name] = {"status": "incomplete"}
    return {"path": str(old), "included_in_final_results": False, "cases": cases}


def render(report):
    lines = ["# Native neighborhood-moments probe report", "",
             f"Overall status: **{report['status']}**", "",
             "Small real fixtures establish functional agreement. Synthetic cases establish performance only.",
             "No full-project speedup claim is made. Arithmetic is native FP32 with Tensor Cores disabled.",
             "CellRank's FP64 velocity layer was left unchanged; native stochastic-model integration was not performed.", "",
             "## Fixture correctness", "", "| Fixture | Status | Raw fields | Centered/adjusted | Deterministic downstream |", "|---|---:|---:|---:|---:|"]
    for f in report.get("fixtures", []):
        fields = f.get("raw_and_centered_fields", {})
        raw = "pass" if fields and all(fields.get(k, {}).get("pass") for k in RAW) else "fail/incomplete"
        centered = "pass" if fields and all(fields.get(k, {}).get("pass") for k in CENTERED) else "fail/incomplete"
        downstream = f.get("downstream", {})
        lines.append(f"| {f['dataset']} | {f['status']} | {raw} | {centered} | {'pass' if downstream.get('all_pass') else 'fail/incomplete'} |")
    lines += ["", "## Synthetic performance", "", "Ratios compare the matched sparse SciPy complete composition with native resident event/host-wall composition only. They exclude end-to-end startup and I/O.", ""]
    lines += ["| Case | Status | SciPy median (ms) | Native event median (ms) | Native host-wall median (ms) | Event ratio |", "|---|---:|---:|---:|---:|---:|"]
    for b in report.get("benchmarks", []):
        cpu, event, wall = b.get("scipy_sparse_composition"), b.get("native_resident_event_composition"), b.get("native_resident_host_wall_composition")
        ratio = b.get("eligible_resident_only_ratios", {}).get("scipy_median_over_native_resident_event_median")
        val = lambda x: f"{x['median_ms']:.3f}" if x else "—"
        lines.append(f"| {b['benchmark']} | {b['status']} | {val(cpu)} | {val(event)} | {val(wall)} | {ratio:.2f}× |" if ratio is not None else f"| {b['benchmark']} | {b['status']} | {val(cpu)} | {val(event)} | {val(wall)} | — |")
    lines += ["", "Each case uses five warmups and twenty measured repetitions. Variability is the population standard deviation across repetitions; CV is standard deviation divided by the mean.", "", "| Case | SciPy CV (range ms) | Native event CV (range ms) | Native wall CV (range ms) | Topology |", "|---|---:|---:|---:|---|"]
    for b in report.get("benchmarks", []):
        cpu, event, wall = b.get("scipy_sparse_composition"), b.get("native_resident_event_composition"), b.get("native_resident_host_wall_composition")
        span = lambda x: f"{x['coefficient_of_variation']:.3f} ({x['min_ms']:.3f}–{x['max_ms']:.3f})" if x else "—"
        topo = b.get("graph_topology", {})
        topology = f"{topo.get('name', '—')}, degree {topo.get('degree', '—')}, self-loops {topo.get('self_loop_policy', '—')}"
        lines.append(f"| {b['benchmark']} | {span(cpu)} | {span(event)} | {span(wall)} | {topology} |")
    lines += ["", "## Cost and memory scopes", "", "The one-iteration figures are estimates assembled from separately measured stages; they are not measured end-to-end latencies. The resident columns are repeated composition medians. Synthetic generation and fixture/artifact export costs are reported in `report.json`.", "", "| Case | Read (ms) | Prepare (ms) | Allocate (ms) | H2D (ms) | Resident composition (ms) | D2H (ms) | Output write (ms) | Estimated one iteration (ms) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in report.get("benchmarks", []):
        cost = b.get("cost_stages_ms", {}).get("native_single_iteration_estimate", {}) or {}
        values = [cost.get(key) for key in ("input_read_ms", "relation_preparation_ms", "device_allocation_ms", "host_to_device_ms", "resident_composition_event_median_ms", "device_to_host_ms", "output_write_ms")]
        total = b.get("cost_stages_ms", {}).get("native_single_iteration_estimate_total")
        fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"
        lines.append(f"| {b['benchmark']} | " + " | ".join(fmt(x) for x in values) + f" | {fmt(total)} |")
    lines += ["", "`Example-owned device bytes` counts the example's resident vectors and external relation-values buffer. Cellerator prepared-relation byte counters are shown separately and are not added to that value. Native RSS is per executable process; Python RSS is process-wide and cumulative.", "", "| Case | Example-owned device bytes | Prepared shared structural bytes | Prepared instance value bytes | Native process peak RSS (bytes) | Python cumulative peak RSS (bytes) |", "|---|---:|---:|---:|---:|---:|"]
    for b in report.get("benchmarks", []):
        memory = b.get("memory_bytes", {})
        fmt = lambda x: str(x) if x is not None else "—"
        lines.append("| " + b["benchmark"] + " | " + " | ".join(fmt(memory.get(key)) for key in ("example_owned_device_allocation_bytes", "prepared_relation_shared_structural_bytes", "prepared_relation_instance_value_bytes", "native_process_peak_rss", "python_process_peak_rss_cumulative")) + " |")
    lines += ["", "## Preserved superseded attempt", "", f"Structured circulant attempt: `{report['old_circulant_attempt'].get('path', report['old_circulant_attempt'].get('status'))}`; excluded from final results.", "", "See `report.json` for per-field errors, provenance hashes, full cost stages, GPU/toolchain metadata, and memory scopes.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, out = args.artifacts.resolve(), args.output.resolve()
    missing = []
    correctness = read(root / "correctness-summary.json", missing)
    final = read(root / "summary.json", missing)
    fixtures = [fixture_report(root, name, correctness or {}, missing) for name in FIXTURES]
    benchmarks = [benchmark_report(root, name, final or {}, missing) for name in BENCHMARKS]
    execution_errors = [*(correctness or {}).get("errors", []), *(final or {}).get("errors", [])]
    passed = (not missing and not execution_errors and all(x["status"] == "pass" for x in fixtures + benchmarks)
              and bool(correctness and correctness.get("analytic", {}).get("pass")))
    provenance = read(root / "provenance.json", [])
    if provenance is None:
        provenance = read(root / "correctness-provenance.json", missing)
    report = {
        "schema_version": 1, "status": "pass" if passed else ("incomplete" if missing else "fail"),
        "artifacts_root": str(root), "correctness_analytic": (correctness or {}).get("analytic"),
        "provenance": provenance,
        "fixed_acceptance_gates": {"analytic": {"rtol": 1e-6, "atol": 1e-6},
                                   "raw": {"rtol": 1e-5, "atol": 1e-6},
                                   "centered_adjusted": "absolute error <= 1e-6 + 1e-5 * sum(abs(expression terms))",
                                   "downstream": {"rtol": 1e-4, "atol": 1e-5, "gene_mask_identical": True}},
        "fixtures": fixtures, "benchmarks": benchmarks,
        "old_circulant_attempt": old_structured_attempt(root),
        "execution_errors": execution_errors,
        "limitations": ["Real fixtures support functional agreement, not performance claims.",
                        "Synthetic timings do not establish full-project speedup.",
                        "CellRank FP64 velocity was preserved; no native FP64 velocity route was qualified.",
                        "Native second moments were not integrated into scVelo's stochastic model.",
                        "No full-project speedup claim is made."],
        "incomplete_or_invalid_evidence": missing,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "report.md").write_text(render(report))
    print(f"{report['status']}: {out / 'report.md'}")
    return 0 if passed else (2 if missing else 1)


if __name__ == "__main__":
    raise SystemExit(main())
