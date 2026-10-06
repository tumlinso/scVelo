# scVelo as a Cellerator and Baseplane reference workload

## Purpose

scVelo is one of two recognizable biological workloads used to help discover
what reusable functionality Cellerator and Baseplane should expose. The goal is
to improve the library surface and demonstrate the resulting substrate on
scientifically meaningful computations. A future port may be useful as an
artifact, but reproducing scVelo's package structure or its Python/SciPy
decomposition is not the design objective.

Preserve the scientific algorithms and their numerical meaning where relevant.
Treat upstream implementations as evidence and test workloads rather than
architectural constraints. Any primitive considered should be composable and
make sense beyond scVelo. Search for reusable computational patterns that may
combine multiple apparent operations when the evidence supports doing so.

The desired primitive granularity is low-level and composable, with enough
semantic meaning for Cellerator to recognize structure, choose specialized
physical representations, fuse operations, and improve on a naive generic
tensor or sparse-matrix translation. Prepared structure, relation algebra,
sparse execution, small-width computation, and compiler machinery are areas to
investigate as possible sources of such opportunities. These are hypotheses
for later evidence gathering, not selected translation strategies, API designs,
or performance claims.

## Evidence to collect in future authorized work

This list describes evidence categories, not findings about scVelo or a
selected implementation plan. Do not infer that any item is dominant or needs
a new primitive before examining an explicitly scoped task and its sources.

- Algorithm and biological meaning, including the numerical behavior that
  must be preserved.
- Exact source location and upstream revision or release examined.
- Repeated computation, data layout and movement, sparse or graph structure,
  reductions, state transformations, iterative methods, and biological
  structures involved.
- Whether apparently separate steps express a reusable composable pattern,
  with evidence for its use beyond this package.
- Likely ownership: general numerical, state, graph, relation, statistical,
  and iterative work belongs in Cellerator; Baseplane is for intrinsically
  sequence-grounded functionality.
- Candidate design hypotheses, relevant numerical validation requirements,
  uncertainty, and counterevidence. Hypotheses remain hypotheses until
  separately reviewed; this document selects no primitive or translation.
- Reusable primitive gaps: the concrete computation and source evidence, the
  current library capability and specific gap, a proposed general-library
  owner, the reusable contract, and downstream impact. Report a basic primitive
  gap promptly to the user. Do not propose project-specific core APIs; a packed
  Tensor Core operation is an example only when evidence shows the general
  primitive is missing or insufficient.
- Native and reduced-precision behavior: preserve the upstream scientific
  algorithm and computation as the default. The only permitted deliberate
  numerical deviation is an optional mode that opportunistically uses FP16
  Tensor Core execution. Record operand, multiply, accumulation, and output
  precision, quantization, selected hardware path, and full-precision native
  fallback. Compare with the upstream/native computation; a higher-precision
  reference may supplement that comparison. Prefer FP32 accumulation where
  supported, and declare and qualify the actual accumulation and output policy.
  Use the optional mode only within its demonstrated numerical envelope; use
  the native fallback for unsupported or unsafe regimes. Check finite-value
  behavior, overflow, underflow, and cancellation; assess absolute and relative
  errors with explicit near-zero handling and relevant downstream scientific
  quantities. Define probe-specific acceptance limits before speed claims and
  report failures honestly. Numerical qualification is not biological
  validation.
- Complete performance costs: report numerical qualification separately from
  speed. Include preparation, format conversion or packing, transfers,
  allocation, peak memory, and execution in the relevant end-to-end costs, and
  state which costs are included in each timing. A faster kernel alone does not
  establish a faster workload.

## Numerical qualification and readiness boundary

An scVelo probe may discover a reusable opportunity; it does not imply that
Cellerator or Baseplane already implements or qualifies the needed path. The
existing Cellerator Tensor Core documentation is authoritative for its
implementation. Keep implementation readiness and performance unproven until
the scoped probe supplies source-bound implementation evidence, numerical
qualification, and complete cost measurements. Do not silently lower default
precision or introduce unrelated algorithmic approximations.

## Boundary for this setup

This documentation update establishes intent and workflow context only; it
starts no port and does not select a translation strategy, authorize
benchmarks or experiments, or authorize biological data processing. Begin a
port/probe only with explicit user authorization and a scoped Project Control
task under the independent `scvelo` Todo authority. That authorization
carries standing permission for additive changes to Cellerator, Baseplane, and
other relevant core libraries needed by the probe; do not seek renewed
permission solely to add a missing reusable primitive. Keep each library
general, clean, maintainable, and explicit in its contracts and validation.
Small cleanup may address demonstrated architectural friction. Discuss major
API, granularity, ownership, or architectural choices with the user before
committing to them. No active port run is established by this documentation.

The paired CellRank project has its own documentation and independent
`cellrank` Todo authority. In Project Control, both projects are aliases under
the parent workspace `accelerated-ports`; they are separate project identities
and Todo authorities, not separately registered workspaces. scVelo's project
UUID is `b6b1bce7-d113-4646-9d74-062156d3cbac`; CellRank's is
`1fe1b4ab-7379-4f47-a4f7-1865ddb446aa`. Coordinate cross-project design
decisions and final acceptance at the root controller rather than treating
either repository's notes as authority over the other's workflow state.

The workspace retains `cellrank` as its authority repository, so workspace
orientation and catalog context use CellRank's semantic authority. scVelo's
native task state remains separate; select it with its repository root when
using `next_task`. This setup does not merge state or change workspace authority.

## Project Control bootstrap note

Project Control state is managed through its supported workflow tools; generated
Todo state is not edited by hand. The initial authority bootstrap used the
Todo Orchestrator maintenance skill, version recorded by SHA-256
`0947d87f4ee1d0335460e14423ea712a68d52d8c1503e04dc21927c877a29ddb`.
Ordinary future task work should use Project Control's model-facing workflow.
