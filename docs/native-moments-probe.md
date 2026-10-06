# Native neighborhood-moments probe

This probe checks whether Cellerator's prepared sparse relation and native
numeric operations can express reusable neighborhood statistics. It reads
small local scVelo and CellRank fixtures, exports the values finalized by the
current upstream recipes, invokes the standalone C++ executable, and records
numerical and timing evidence. It does not modify source H5AD files.

The composed fields are `W L`, `W R`, `W(L*L)`, `W(L*R)`, and `W(R*R)`.
The harness derives the centered variance/covariance and scVelo's adjusted
second moments without clipping. For scVelo fixtures, the graph comes from
`scvelo.pp.moments` and `get_connectivities(recurse_neighbors=False)`; for the
CellRank fixture, the harness loads the source `_moments.py` helper directly
and calls its binary self-loop row-normalization and `_knn_moments` methods.
The CellRank fixture's FP64 velocity layer remains untouched and is recorded
as a later FP64-native-route requirement.

## Run

Build `ceNativeNeighborhoodMoments` from Cellerator, then use the probe's
Python environment and an artifact directory outside the source tree:

```sh
export NUMBA_CACHE_DIR=/tmp/moments-probe-20261006/numba
export MPLCONFIGDIR=/tmp/moments-probe-20261006/mpl
python probes/native_moments/harness.py \
  --native /path/to/ceNativeNeighborhoodMoments \
  --cellerator-root /path/to/Cellerator-producer-worktree \
  --build-info /path/to/build-info.json \
  --output /tmp/native-moments-results \
  --mode correctness
```

Use `--mode benchmarks` for the two seeded synthetic workloads and `--mode
full` for fixtures plus benchmarks. `--only KEY` selects a fixture or
benchmark; repeat it to select several. `--warmups` and `--repeats` configure
the native executable, defaulting to five and twenty. GPU execution must run
under the CUDA controller's admitted-device recipe. The harness itself never
sets a GPU ordinal or starts a CUDA run outside that controller.

The executable file protocol is described in
`/tmp/moments-probe-20261006/interface.md` during this probe. Each case writes
little-endian CSR and row-major FP32 input files, a sparse SciPy `.npz` input
for the matched CPU baseline, and a provenance manifest. Results include
native raw fields, native `metrics.json`, comparison JSON, and a top-level
summary. `provenance.json` hashes the executable and scoped source files,
records each repository's HEAD plus scoped dirty status/diff identity, and
includes the supplied CMake/toolchain build receipt. The acceptance thresholds
are copied to `acceptance.json` before any computation starts. Keep generated
artifacts out of Git.

## Acceptance and interpretation

Raw moments use `rtol=1e-5, atol=1e-6`. Centered and adjusted quantities use
the absolute bound `1e-6 + 1e-5 * sum(abs(expression terms))`, which makes
cancellation visible without changing the computed value. Analytic fixtures
use `rtol=atol=1e-6`. Deterministic velocity coefficients and residuals use
`rtol=1e-4, atol=1e-5`; velocity-gene masks must match exactly. Nonfinite
reference coordinates are reported explicitly. Any failed gate remains a
failure; do not widen these limits to make a run pass.

The harness feeds native first moments into a copy of the input AnnData and
fits the unchanged deterministic `SteadyStateModel` with
`use_highly_variable=False`, `r2_adjusted=False`, `fit_offset=False`, and
`perc=(5, 95)`. It compares gamma, offset, R-squared, residual velocity, and
the selected-gene mask. Second moments are checked directly against upstream
computation. This does not integrate native moments into the stochastic
model, which recomputes them internally.

Small biological fixtures establish functional agreement. Only synthetic
measurements support performance statements. Compare all preparation,
materialization, transfer, resident, and total timing scopes from the C++
metrics with the matched sparse SciPy composition. The eligible ratios compare
the median repeated complete composition against the native resident event or
host-wall median; they do not represent end-to-end speedup. The separately
reported one-iteration estimate adds input read, allocation, relation
preparation, H2D, median composition, D2H, and output writing. Native process
total includes warmups, measured repeats, and a diagnostic iteration. Python
peak RSS is process-wide and cumulative; native peak RSS is per executable
process. A slowdown is useful when its cost is localized. The first pass uses
native FP32 arithmetic. An optional FP16 Tensor Core mode requires a later,
separately qualified numerical envelope.

The synthetic graph generator is
`independent_uniform_unique_nonself_neighbors_v1`: each row independently
samples 32 distinct columns uniformly without replacement, excluding its own
row, then stores CSR columns in ascending order and normalizes the random
positive weights. The seed and graph policy are recorded in each input
manifest. This gives fixed degree without repeating one shared offset pattern;
the two cases still describe synthetic workloads rather than biological graph
performance.

## First-probe result

All three real fixtures passed the fixed raw, centered/adjusted, and
deterministic-downstream gates, including exact velocity-gene-mask agreement.
The arithmetic and composed-operation Compute Sanitizer checks reported zero
errors; all 12 CPU harness tests passed. The deterministic fit's residual was
read from `model.residual`: in this scVelo version, `get_velocity()` returns
`None` because `fit()` does not populate `_state_dict.residual`.

The synthetic compute case measured median SciPy/native-resident-event
compositions of 4.120 s/7.864 ms (523.94x resident-only ratio). The transfer
case measured 13.921 s/50.495 ms (275.70x resident-only ratio). These timings
cover the synthetic cases only. The native timing uses retained prepared CSR
and resident fields; it is not an end-to-end project speed claim. In the
estimated one-use path, D2H materialization and output writing cost far more
than resident composition, especially for the larger transfer case. The report
bundle, including structured JSON and reproducibility artifacts, is available
at `/home/tumlinson/data/design-probes/native-moments-20261006/report/`.

This first probe uses FP32 with Tensor Cores disabled. CellRank's stored FP64
velocity was preserved without native conversion, and native moments were not
integrated into scVelo's stochastic model. The most useful next step is to
measure resident binding consumption and selective output materialization,
because those interfaces determine whether the resident-compute result can
benefit an actual consumer. The evidence also identifies two focused library
questions: Cellerator needs an explicit strict-versus-fast math admission
contract for FMA/reassociation while retaining CSR, and ownership should be
coordinated for an FP64 elementwise multiply primitive needed by a later
CellRank velocity path. No broad API refactor is indicated by this probe.
