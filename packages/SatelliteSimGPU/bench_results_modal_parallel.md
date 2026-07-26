# Modal parallel GPU suite results (2026-07-12)

Device: NVIDIA A10G. Profile: `satsim-gpu`. Entrypoint:
`MODAL_PROFILE=satsim-gpu modal run modal_gpu.py --suites parallel`

## Wave 1 — 20 parallel suites

| Suite | Result |
|---|---|
| smoke_info | PASS |
| coverage_f64 / f32 | PASS |
| gsl_canonical_f64 / f32 | PASS |
| gsl_f64 / f32 | PASS |
| isl_f64 / f32 | PASS |
| registered_f64 / f32 | PASS |
| pipeline_adjoint | PASS |
| reductions_f64 / f32 | PASS |
| sgp4_cuda | FAIL (world-age from runtime `Base.require`) |
| bench_coverage / gsl / isl | PASS |
| bench_gsl_reduction / isl_reduction | PASS |

**TOTAL: 19 PASS / 1 FAIL**

## Wave 2 — sgp4 fix + rerun

Fix: top-level `using SatelliteToolboxSgp4` (no runtime require); image installs `SatelliteToolboxSgp4`.

| Suite | Result | Notes |
|---|---|---|
| sgp4_cuda | PASS | max_pos_err ≈ 4e-12 km (`sgp4` / `sgp4_lowper`) |

**Effective overall: 20 / 20 PASS**

## Wave 3 — differentiable SGP4 Step1 (Modal CPU)

```bash
MODAL_PROFILE=satsim-gpu modal run packages/SatelliteSimGPU/modal_gpu.py --suites sgp4_step1
```

- Run: `ap-Czf6YJpGvioVnLqQLbslvj`
- Instantiated Opt project in-container, then:
  - `STEP1 loss=0.9007554081482751 grad_norm=50.591236569613706 finite=true`
  - `STEP1 fd_max_relerr=1.1088079075743219e-7 n_checked=20`
  - `STEP1_OK`
- **PASS**

## Notable performance signals (not failures)

- GSL device-reduction Float32: strong e2e speedups (≈12–66×) vs full-download+host-reduce.
- ISL device-reduction e2e often *slower* than full download on these A10G scales (parity still PASS) — revisit kernel/launch overhead later.

## Wave 4 — CUDA runtime-pref fix + full parallel rerun (2026-07-13)

Regression seen first: a full rerun came back **20/20 FAIL**, every suite (even
`smoke_info`) dying at `CUDA is not functional` /
"CUDA.jl could not find an appropriate CUDA runtime to use". Root cause: the
project-local `Pkg.add(CUDA==6.2.1)` under the runner's narrow
`JULIA_LOAD_PATH=@:@stdlib` does **not** inherit the base image's CUDA
runtime-version preference (it lives only in the base default depot env), so the
freshly-added CUDA has no runtime on the A10G.

Image fix (`modal_gpu.py` build): after `Pkg.add` CUDA 6.2.1 + SatelliteToolboxSgp4,
locate the base `LocalPreferences.toml` and copy its `[CUDA_Runtime_jll]` into the
project, then re-precompile. Observed at build:
`COPIED_CUDA_RT_PREF from=/depot/environments/v1.12/LocalPreferences.toml cfg=("version" => "12.9")`
then `IMAGE_CUDA_OK cuda=6.2.1` under the exact offline + narrow-load-path env.
(`_run_e2e_grad` also now instantiates the Opt project before running.)

Validation: `--suites smoke_info` → PASS
(`GPU_INFO device=NVIDIA A10G total_mem_gib=22.06 compute_capability=8.6
driver=13.3.0 cuda_runtime=12.9.0 cuda_jl=6.2.1 julia=1.12.6 julia_threads=2`).

Full run: `ap-wRK6hrq0UFrqGNH90CUKwf` (`--suites parallel`, profile `satsim-gpu`,
1 attempt, no "modified during build").

| Suite | Result |
|---|---|
| smoke_info | PASS |
| coverage_f64 / f32 | PASS |
| gsl_canonical_f64 / f32 | PASS |
| gsl_f64 / f32 | PASS |
| isl_f64 / f32 | PASS |
| registered_f64 / f32 | PASS |
| pipeline_adjoint | PASS |
| reductions_f64 / f32 | PASS |
| sgp4_cuda | PASS |
| bench_coverage / gsl / isl | PASS |
| bench_gsl_reduction / isl_reduction | PASS |

**SUMMARY: TOTAL=20 PASS=20 FAIL=0 — failed suites: none**

Bonus check — `--suites sgp4_step1` (`ap-1GPX06uqUq50QUlTsSfgiU`, CPU-4thread): Opt
project instantiated in-container (`STEP1_INSTANTIATE status=PASS`), then
`STEP1 loss=0.9007554081482751 grad_norm=50.591236569613706 finite=true`,
`fd_max_relerr=1.1088079075743219e-7 n_checked=20`, `STEP1_OK` → **PASS** (matches Wave 3).
