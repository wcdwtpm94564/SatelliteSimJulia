# SatelliteSimGPU Modal Bench Notes (ROUND2_56)

This document is the distilled execution evidence for WF4 Modal harness rules in this round.
It intentionally omits raw terminal logs.

## Scope

- Harness launcher: `packages/SatelliteSimGPU/modal_gpu.py`
- Julia runner: `packages/SatelliteSimGPU/modal_gpu_runner.jl`
- Contract test: `packages/SatelliteSimGPU/test/test_modal_harness.py`

## 1) Correctness Gates (Executable)

The runner now enforces one unified tolerance contract:

- Bool/Int outputs: exact equality
- Float64: `rtol=1e-12`, `atol=1e-10`
- Float32 common: `rtol=1e-4`
- Float32 metric-specific `atol`:
  - scalar loss / generic scalar fields: `5e-5`
  - distance/elevation: `2e-3`
  - delay: `2e-5`

Implementation anchors:

- constants and gate selector: `F64_RTOL`, `F64_ATOL`, `F32_RTOL`, `F32_*_ATOL`, `gate_tolerance(...)`
- parity call sites: `validate_coverage`, `validate_canonical_gsl`, `validate_gsl`, `validate_isl`, `validate_registered_compute_backend`, `bench_*` parity checks, `bench_real1584_coverage_case`

## 2) Default Suite Set Contract (20 strict names)

`DEFAULT_PARALLEL_SUITES` remains a strict 20-item set with no duplicates.

Hard guards added at launcher import:

- exact length must be 20
- duplicate names forbidden
- set equality must match `EXPECTED_PARALLEL_SUITE_SET`

Pass/fail sentinel remains strict per suite:

- `exit_code == 0`
- exactly one `MODAL_GPU_VALIDATION status=PASS suite=<suite>`
- sentinel must be final output line

## 3) Logical Output-Volume Gate (Reduction)

Reduction benches now verify transfer-reduction formulas exactly before reporting:

- GSL:
  - formula: `N(1+3sizeof(T))/4`
  - code: `expected_transfer = n_satellites * (1 + 3 * sizeof(T)) / 4`
- ISL:
  - formula: `P(2+5sizeof(T))/4`
  - code: `expected_transfer = actual_pairs * (2 + 5 * sizeof(T)) / 4`

Hard floors for Float32 are enforced:

- GSL: `>= 1787.5x`
- ISL: `>= 6050x`

The checks are fail-closed:

- `transfer_reduction == expected_transfer` must hold
- Float32 floor failures raise `error(...)`

## 4) Speedup Policy (Observation-only)

Speedup is printed for diagnostics only; it is not a pass gate in this round.

- no `speedup`-based `error(...)` checks
- no `speedup >= ...` hard threshold checks
- `best_elapsed(...)` timing is retained only for reporting, not for PASS/FAIL

Note: known F64 GSL reduction counterexample (`~0.30x`) remains acceptable because speedup is non-gating.

## 5) Dirty/Clean Preflight

Launcher preflight now blocks remote Modal runs when image-source paths are dirty, unless explicitly bypassed.

- function: `_require_clean_modal_sources()`
- checked paths include:
  - `packages/SatelliteSimGPU`
  - `packages/SatelliteSimBackends`
  - `src/foundation`, `src/orbit`, `src/link`, `src/net`, `src/opt`
  - `data/tle/celestrak/starlink_gp_latest.tle`
- guidance in failure message:
  - run from a clean commit
  - or use a git-archive mirror
- explicit debug bypass only: `SATSIM_ALLOW_DIRTY_MODAL_SOURCE=1`

## 6) Known Signature Fix (threads-related invocation path)

Modal function calls that previously used positional arguments were converted to keyword form:

- `stable_cpu_validate.spawn(commit=commit)`
- `e2e_grad_cpu16.remote(engines="all")`
- `e2e_grad_cpu32.remote(engines="blockdiag")`

This avoids API/signature mismatch failures in current Modal client behavior.

## 7) Verification Commands

Required:

```bash
python3 packages/SatelliteSimGPU/test/test_modal_harness.py
```

Optional:

```bash
julia --project=packages/SatelliteSimGPU -e 'using Pkg; Pkg.test(; coverage=false)'
```

Remote Modal GPU execution is not required for this round's local contract gate.
