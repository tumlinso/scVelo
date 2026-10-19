# scVelo project guidance

<!-- project-control:start -->
## Project Control and native Todo workflow

Use Project Control's `overview`, `search`, `delta`, `frontier`, `history`,
`evidence`, and `impact` tools for registered-workspace context. They are not
the Todo transaction API. For Todo state and workflow, use the native `todo_*`
tools exposed by the host. Start with `todo_read_authority`; for plan changes,
call `todo_preview_plan`, inspect the returned preview, then pass that exact
preview to `todo_commit_plan` with an idempotency key. Use the corresponding
`todo_open_run`, `todo_create_workspace`, `todo_dispatch_attempt`, outcome,
verification, and recovery tools for later lifecycle steps.

Arguments use a registered workspace ID and, when necessary, an optional
registered `repository` alias. The host binds authority, identity, credentials,
profile, and providers; never pass filesystem roots or bearer credentials as
selectors. Read migrated history with `todo_search_legacy_history`, current
status with `todo_read_legacy_current_tasks`, and preserved declarations with
`todo_read_legacy_project_context`. Imported completion is a source assertion,
not native verification; unknown legacy effects require attention and are never
replayed automatically. Project Control's existing non-Todo tools remain
available for their own purposes.
<!-- project-control:end -->

This repository contains scVelo, a scientific toolkit for RNA velocity and
related single-cell analyses. Project-specific development guidance remains in
`CONTRIBUTING.rst`, while the Cellerator/Baseplane reference-workload intent
and evidence template are in `docs/library-design-probes.md`.

## Cellerator and Baseplane reference workload

scVelo is a scientific reference workload for discovering reusable Cellerator
and Baseplane library functionality. The goal is to learn which composable
primitives could express meaningful scVelo computations efficiently, while
also producing a recognizable scientific demonstration in future work. The
goal is not to recreate scVelo as a faster tool-specific implementation.

Treat upstream implementations as evidence about algorithms, data, and
workload behavior, not as architectures to copy. Preserve scientific
algorithms and numerical meaning where relevant. General numerical, state,
graph, relation, statistical, and iterative computation belongs in Cellerator;
Baseplane should own only functionality whose meaning is intrinsically
sequence-grounded. The design note records evidence needs without selecting an
implementation strategy.

This documentation update does not start a port, select a translation
strategy, authorize benchmarks or experiments, or authorize biological data
processing. Begin a port/probe only with explicit user authorization and its
scoped Project Control task. That authorization carries standing permission
for additive changes to Cellerator, Baseplane, and other relevant core
libraries needed by the probe; do not ask again solely because a missing
primitive requires an additive library change. Keep those libraries general,
clean, and maintainable, with explicit contracts and validation. Small,
evidence-driven cleanup is welcome when real architectural friction appears;
discuss major API, granularity, ownership, or architectural choices with the
user before committing to them.

Keep scientific computation at its native precision and algorithm by default.
The only permitted deliberate deviation is an optional, opportunistic FP16
Tensor Core mode. Compare it with the upstream/native computation; a higher-
precision reference may supplement that comparison. Prefer FP32 accumulation
where supported, and declare and qualify the actual accumulation and output
policy. Use the mode only within its demonstrated numerical envelope, retaining
the native fallback for unsupported or unsafe regimes. Qualify finite-value
behavior, overflow, underflow, cancellation, absolute and relative errors with
near-zero handling, and relevant downstream scientific quantities. Set probe-
specific acceptance limits before speed claims and report failures honestly;
numerical qualification is not blanket biological validation. Existing
Cellerator Tensor Core documentation is authoritative for implementation; do
not imply a path exists or is qualified without evidence.

When a basic reusable primitive appears missing or insufficient in a core
library, report it promptly to the user with the concrete computation and
evidence, existing capability and gap, proposed library owner, reusable
contract, and downstream impact. A packed Tensor Core operation is an example
to investigate only if evidence shows it is missing. Reporting a gap does not
by itself require renewed permission for additive work within the authorized
probe.

Project Control owns workflow state: use the independent `scvelo` Todo
authority for this repository and the `cellrank` authority for its sibling.
Do not edit generated Todo state directly or infer an active port run from
this documentation. Use configured Codex subagents for bounded delegated work
when authorized; keep architecture, integration, and acceptance with the root
controller.

In Project Control, `scvelo` and `cellrank` are project aliases under parent
workspace `accelerated-ports`, not separately registered workspaces. Their
project identities remain independent: scVelo UUID
`b6b1bce7-d113-4646-9d74-062156d3cbac`; CellRank UUID
`1fe1b4ab-7379-4f47-a4f7-1865ddb446aa`.

For delegated research, implementation, tests, or review, use only configured
Codex subagent profiles `scout`, `researcher`, `implementer`, `reviewer`, or
`parallel-head`, within an explicit bounded scope. Do not use generic/default
agents, Project Control `delegate_task`, or local workers for ordinary
assignments. Local workers are reserved for Project Control observers unless
the user gives a newer, explicit instruction. The root controller retains
cross-project strategy, integration, and final acceptance. These project
specific rules take precedence over broader generated workflow text where they
differ.
