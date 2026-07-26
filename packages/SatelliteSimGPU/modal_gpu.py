"""Modal A10G / CPU runners for SatelliteSimGPU suites.

Usage:
  MODAL_PROFILE=satsim-gpu modal run modal_gpu.py
      # parallel default suites (GPU shards; keep concurrency modest)
  MODAL_PROFILE=satsim-gpu modal run modal_gpu.py --suites coverage_f64,sgp4_cuda
  MODAL_PROFILE=satsim-gpu modal run modal_gpu.py --suites full
  MODAL_PROFILE=satsim-gpu modal run modal_gpu.py --suites real1584
      # Stage-1: 1584 real TLE forward (1×A10G + 1×CPU-2thread + 1×opt-load)
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

import modal


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
REMOTE_PACKAGE_DIR = "/opt/SatelliteSimGPU"
BACKENDS_DIR = PACKAGE_DIR.parent / "SatelliteSimBackends"
REMOTE_BACKENDS_DIR = "/opt/SatelliteSimBackends"
TLE_LOCAL = REPO_ROOT / "data" / "tle" / "celestrak" / "starlink_gp_latest.tle"
TLE_REMOTE = "/opt/data/tle/celestrak/starlink_gp_latest.tle"
SRC_REMOTE = "/opt/src"
OPT_SRC_PACKAGES = ("foundation", "orbit", "link", "net", "opt")

# Parallel default matrix: correctness shards + new SGP4/reduction + per-op benches.
DEFAULT_PARALLEL_SUITES = [
    "smoke_info",
    "coverage_f64",
    "coverage_f32",
    "gsl_canonical_f64",
    "gsl_canonical_f32",
    "gsl_f64",
    "gsl_f32",
    "isl_f64",
    "isl_f32",
    "registered_f64",
    "registered_f32",
    "pipeline_adjoint",
    "reductions_f64",
    "reductions_f32",
    "sgp4_cuda",
    "bench_coverage",
    "bench_gsl",
    "bench_isl",
    "bench_gsl_reduction",
    "bench_isl_reduction",
]
EXPECTED_PARALLEL_SUITE_SET = {
    "smoke_info",
    "coverage_f64",
    "coverage_f32",
    "gsl_canonical_f64",
    "gsl_canonical_f32",
    "gsl_f64",
    "gsl_f32",
    "isl_f64",
    "isl_f32",
    "registered_f64",
    "registered_f32",
    "pipeline_adjoint",
    "reductions_f64",
    "reductions_f32",
    "sgp4_cuda",
    "bench_coverage",
    "bench_gsl",
    "bench_isl",
    "bench_gsl_reduction",
    "bench_isl_reduction",
}
IMAGE_SOURCE_PATHS = [
    "packages/SatelliteSimGPU",
    "packages/SatelliteSimBackends",
    "src/foundation",
    "src/orbit",
    "src/link",
    "src/net",
    "src/opt",
    "data/tle/celestrak/starlink_gp_latest.tle",
]
ALLOW_DIRTY_ENV = "SATSIM_ALLOW_DIRTY_MODAL_SOURCE"


def _assert_default_suite_contract() -> None:
    if len(DEFAULT_PARALLEL_SUITES) != 20:
        raise RuntimeError(
            f"DEFAULT_PARALLEL_SUITES must contain 20 suites, found {len(DEFAULT_PARALLEL_SUITES)}"
        )
    if len(set(DEFAULT_PARALLEL_SUITES)) != len(DEFAULT_PARALLEL_SUITES):
        raise RuntimeError("DEFAULT_PARALLEL_SUITES contains duplicate names")
    if set(DEFAULT_PARALLEL_SUITES) != EXPECTED_PARALLEL_SUITE_SET:
        raise RuntimeError(
            "DEFAULT_PARALLEL_SUITES no longer matches the required strict suite set"
        )


def _require_clean_modal_sources() -> None:
    if os.environ.get(ALLOW_DIRTY_ENV) == "1":
        return
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *IMAGE_SOURCE_PATHS],
        text=True,
    )
    dirty = [line for line in status.splitlines() if line.strip()]
    if dirty:
        preview = "; ".join(dirty[:6])
        raise SystemExit(
            "dirty preflight failed for Modal image sources: "
            f"{preview}. Build from a clean commit (or git archive mirror); "
            f"set {ALLOW_DIRTY_ENV}=1 only for explicit debugging."
        )


_assert_default_suite_contract()

image_builder = (
    modal.Image.from_registry(
        "ghcr.io/juliagpu/cuda.jl@sha256:8c40fadfbeea933b98e81a1b164cc3ccb8d442c6caf9e3285e4b577d30d5dd13",
        add_python="3.12",
    )
    .entrypoint([])
    .add_local_dir(BACKENDS_DIR, REMOTE_BACKENDS_DIR, copy=True)
    .add_local_dir(PACKAGE_DIR, REMOTE_PACKAGE_DIR, copy=True)
    .add_local_file(str(TLE_LOCAL), TLE_REMOTE, copy=True)
)

for _pkg in OPT_SRC_PACKAGES:
    image_builder = image_builder.add_local_dir(
        str(REPO_ROOT / "src" / _pkg),
        f"{SRC_REMOTE}/{_pkg}",
        copy=True,
        # Parallel workers may edit test/; keep only loadable package content.
        ignore=["**/test/**", "**/.DS_Store", "**/__pycache__/**"],
    )

image = image_builder.run_commands(
    # Match src/*/Project.toml [sources] path ../../packages/SatelliteSimBackends
    "mkdir -p /opt/packages && ln -sfn /opt/SatelliteSimBackends /opt/packages/SatelliteSimBackends",
    "julia --project=/opt/SatelliteSimGPU -e '"
    "using Pkg; "
    "Pkg.instantiate(); "
    # CUDA is not a SatelliteSimGPU Project.toml dep (CPU KA tests stay light), "
    # but modal_gpu_runner.jl needs it on A10G. Pin to runner EXPECTED_CUDA_JL_VERSION.
    "Pkg.add(name=\"CUDA\", version=\"6.2.1\"); "
    "Pkg.add(\"SatelliteToolboxSgp4\"); "
    "Pkg.precompile()'",
    # A freshly Pkg.add-ed CUDA in this project does NOT inherit the base image's
    # CUDA runtime-version preference (it lives only in the base default depot env),
    # so CUDA.functional() is false on the A10G ("could not find an appropriate CUDA
    # runtime"). Locate base LocalPreferences.toml files and inherit CUDA_Runtime_jll.
    "find /depot /root/.julia /opt -maxdepth 6 -name LocalPreferences.toml 2>/dev/null "
    "| tee /tmp/lp_paths.txt; "
    "while read f; do echo \"=== $f ===\"; cat \"$f\"; done < /tmp/lp_paths.txt || true",
    "julia --project=/opt/SatelliteSimGPU -e '"
    "import TOML; "
    "paths = isfile(\"/tmp/lp_paths.txt\") ? readlines(\"/tmp/lp_paths.txt\") : String[]; "
    "src = nothing; cfg = nothing; "
    "for p in paths; isfile(p) || continue; "
    "t = try TOML.parsefile(p) catch; continue end; "
    "if haskey(t, \"CUDA_Runtime_jll\") && cfg === nothing; global src = p; global cfg = t[\"CUDA_Runtime_jll\"]; end; "
    "end; "
    "cfg === nothing && error(\"no CUDA_Runtime_jll pref among: \" * join(paths, \", \")); "
    "dst = \"/opt/SatelliteSimGPU/LocalPreferences.toml\"; "
    "proj = isfile(dst) ? TOML.parsefile(dst) : Dict{String,Any}(); "
    "proj[\"CUDA_Runtime_jll\"] = cfg; "
    "open(dst, \"w\") do io; TOML.print(io, proj); end; "
    "println(\"COPIED_CUDA_RT_PREF from=\" * src * \" cfg=\" * string(cfg))'",
    "julia --project=/opt/SatelliteSimGPU -e 'using Pkg; Pkg.precompile()'",
    # Lock the image contract: under the runner's exact offline + narrow load path,
    # CUDA (pinned) and SatelliteToolboxSgp4 must both import. Fails the build early
    # (before spawning GPU containers) if the load path is too narrow to see CUDA.
    "JULIA_PKG_OFFLINE=true JULIA_LOAD_PATH=@:@stdlib "
    "julia --project=/opt/SatelliteSimGPU -e '"
    "using CUDA, SatelliteToolboxSgp4; "
    "pkgversion(CUDA) == v\"6.2.1\" || "
    "error(\"image CUDA \" * string(pkgversion(CUDA)) * \" != 6.2.1\"); "
    "println(\"IMAGE_CUDA_OK cuda=\" * string(pkgversion(CUDA)))'",
    # Opt deps are large (Enzyme/Zygote/Lux). Instantiate at runtime in opt_load_check
    # so image builds stay short and avoid races with parallel src/opt editors.
)

app = modal.App("satellitesim-gpu-validation")


def _run_julia_suite(
    suite: str,
) -> dict[str, Any]:
    import os

    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = "2"
    env["JULIA_LOAD_PATH"] = "@:@stdlib"
    env["JULIA_PKG_OFFLINE"] = "true"
    env["SATSIM_TLE_PATH"] = TLE_REMOTE
    env["SATSIM_OPT_PROJECT"] = f"{SRC_REMOTE}/opt"
    result = subprocess.run(
        [
            "julia",
            "--threads=2",
            f"--project={REMOTE_PACKAGE_DIR}",
            f"{REMOTE_PACKAGE_DIR}/modal_gpu_runner.jl",
            suite,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    stdout = result.stdout or ""
    print(f"===== SUITE {suite} exit={result.returncode} =====")
    print(stdout, end="" if stdout.endswith("\n") else "\n")
    return {
        "suite": suite,
        "exit_code": result.returncode,
        "stdout": stdout,
        "pass": _suite_passed(result.returncode, stdout, suite),
    }


def _suite_passed(exit_code: int, stdout: str, suite: str) -> bool:
    expected = f"MODAL_GPU_VALIDATION status=PASS suite={suite}"
    lines = stdout.splitlines()
    sentinels = [
        line for line in lines if line.startswith("MODAL_GPU_VALIDATION ")
    ]
    return exit_code == 0 and len(sentinels) == 1 and lines[-1:] == [expected]


@app.function(
    image=image,
    gpu="A10G",
    cpu=2.0,
    memory=8192,
    timeout=40 * 60,
)
def validate_suite(suite: str) -> dict[str, Any]:
    return _run_julia_suite(suite)


@app.function(
    image=image,
    gpu="A10G",
    cpu=2.0,
    memory=8192,
    timeout=40 * 60,
)
def real1584_gpu() -> dict[str, Any]:
    """One A10G container: SGP4-on-device + coverage_loss_gpu F32/F64 × NT/G matrix."""
    return _run_julia_suite("bench_real1584_gpu")


@app.function(
    image=image,
    # CPU-only: multi-thread KA coverage baseline (no GPU requested).
    cpu=4.0,
    memory=8192,
    timeout=40 * 60,
)
def real1584_cpu() -> dict[str, Any]:
    """One CPU container: two-thread coverage_loss_gpu on real 1584 ECEF ephemeris."""
    return _run_julia_suite("bench_real1584_cpu")


def _run_e2e_grad(threads: int, engines: str) -> dict[str, Any]:
    """Stage-2: 1584 end-to-end gradient (SatelliteSimOpt.sgp4_e2e_gradient)."""
    import os

    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = str(threads)
    env["SATSIM_TLE_PATH"] = TLE_REMOTE
    env["SATSIM_OPT_PROJECT"] = f"{SRC_REMOTE}/opt"
    # Opt Manifest is not baked into the image; instantiate before the gradient run.
    prep = subprocess.run(
        [
            "julia",
            f"--project={SRC_REMOTE}/opt",
            "-e",
            "using Pkg; Pkg.instantiate(); println(\"E2E_GRAD_INSTANTIATE status=PASS\")",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    prep_out = prep.stdout or ""
    print(f"===== E2E_GRAD_INSTANTIATE exit={prep.returncode} =====")
    print(prep_out, end="" if prep_out.endswith("\n") else "\n")
    if prep.returncode != 0:
        return {
            "suite": f"e2e_grad_t{threads}",
            "exit_code": prep.returncode,
            "stdout": prep_out,
            "pass": False,
        }

    result = subprocess.run(
        [
            "julia",
            f"--threads={threads}",
            f"--project={SRC_REMOTE}/opt",
            f"{REMOTE_PACKAGE_DIR}/modal_e2e_grad.jl",
            engines,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    result_out = result.stdout or ""
    stdout = prep_out + result_out
    print(f"===== E2E_GRAD threads={threads} engines={engines} exit={result.returncode} =====")
    print(result_out, end="" if result_out.endswith("\n") else "\n")
    return {
        "suite": f"e2e_grad_t{threads}",
        "exit_code": result.returncode,
        "stdout": stdout,
        "pass": result.returncode == 0 and "MODAL_E2E_GRAD status=PASS" in stdout,
    }


@app.function(
    image=image,
    cpu=16.0,
    memory=20480,
    timeout=60 * 60,
)
def e2e_grad_cpu16(engines: str = "all") -> dict[str, Any]:
    return _run_e2e_grad(16, engines)


@app.function(
    image=image,
    cpu=32.0,
    memory=20480,
    timeout=60 * 60,
)
def e2e_grad_cpu32(engines: str = "all") -> dict[str, Any]:
    return _run_e2e_grad(32, engines)


@app.function(
    image=image,
    cpu=16.0,
    memory=20480,
    timeout=60 * 60,
)
def stable_cpu_validate(commit: str = "UNKNOWN") -> dict[str, Any]:
    """Stable headline CPU validation of Opt gradients at a pinned commit."""
    import os

    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = "16"
    env["SATSIM_TLE_PATH"] = TLE_REMOTE
    env["SATSIM_OPT_PROJECT"] = f"{SRC_REMOTE}/opt"
    env["SATSIM_GIT_COMMIT"] = commit
    prep = subprocess.run(
        [
            "julia",
            f"--project={SRC_REMOTE}/opt",
            "-e",
            "using Pkg; Pkg.instantiate(); println(\"STABLE_CPU_INSTANTIATE status=PASS\")",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    prep_out = prep.stdout or ""
    print(f"===== STABLE_CPU_INSTANTIATE exit={prep.returncode} =====")
    print(prep_out, end="" if prep_out.endswith("\n") else "\n")
    if prep.returncode != 0:
        return {
            "suite": "stable_cpu",
            "exit_code": prep.returncode,
            "stdout": prep_out,
            "pass": False,
        }
    result = subprocess.run(
        [
            "julia",
            "--threads=16",
            f"--project={SRC_REMOTE}/opt",
            f"{REMOTE_PACKAGE_DIR}/modal_stable_cpu.jl",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    result_out = result.stdout or ""
    stdout = prep_out + result_out
    print(f"===== STABLE_CPU exit={result.returncode} =====")
    print(result_out, end="" if result_out.endswith("\n") else "\n")
    return {
        "suite": "stable_cpu",
        "exit_code": result.returncode,
        "stdout": stdout,
        "pass": result.returncode == 0 and "MODAL_STABLE_CPU status=PASS" in stdout,
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=2.0,
    memory=8192,
    timeout=40 * 60,
)
def stable_gpu_validate() -> dict[str, Any]:
    """Stable headline A10G parity + one reduction e2e each."""
    return _run_julia_suite("stable_gpu")


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=30 * 60,
)
def opt_load_check() -> dict[str, Any]:
    """Stage-2 paving: cold `using SatelliteSimOpt` only (no gradient)."""
    import os

    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = "1"
    env["SATSIM_OPT_PROJECT"] = f"{SRC_REMOTE}/opt"
    result = subprocess.run(
        [
            "julia",
            "--threads=1",
            f"--project={SRC_REMOTE}/opt",
            f"{REMOTE_PACKAGE_DIR}/modal_opt_load.jl",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    stdout = result.stdout or ""
    print(f"===== OPT_LOAD exit={result.returncode} =====")
    print(stdout, end="" if stdout.endswith("\n") else "\n")
    return {
        "suite": "opt_load",
        "exit_code": result.returncode,
        "stdout": stdout,
        "pass": result.returncode == 0 and "MODAL_OPT_LOAD status=PASS" in stdout,
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=30 * 60,
)
def sgp4_step1_check() -> dict[str, Any]:
    """Stage-2 paving: SGP4 step-1 CPU check (4 threads, real TLE)."""
    import os

    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = "4"
    env["SATSIM_TLE_PATH"] = TLE_REMOTE
    env["SATSIM_OPT_PROJECT"] = f"{SRC_REMOTE}/opt"
    # Opt Manifest may be absent in the image; instantiate before the smoke script.
    prep = subprocess.run(
        [
            "julia",
            f"--project={SRC_REMOTE}/opt",
            "-e",
            "using Pkg; Pkg.instantiate(); println(\"STEP1_INSTANTIATE status=PASS\")",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    prep_out = prep.stdout or ""
    print(f"===== SGP4_STEP1_INSTANTIATE exit={prep.returncode} =====")
    print(prep_out, end="" if prep_out.endswith("\n") else "\n")
    if prep.returncode != 0:
        return {
            "suite": "sgp4_step1",
            "exit_code": prep.returncode,
            "stdout": prep_out,
            "pass": False,
        }

    result = subprocess.run(
        [
            "julia",
            "--threads=4",
            f"--project={SRC_REMOTE}/opt",
            f"{SRC_REMOTE}/opt/scripts/sgp4_step1_check.jl",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    stdout = (prep_out + (result.stdout or ""))
    print(f"===== SGP4_STEP1 exit={result.returncode} =====")
    print(result.stdout or "", end="" if (result.stdout or "").endswith("\n") else "\n")
    return {
        "suite": "sgp4_step1",
        "exit_code": result.returncode,
        "stdout": stdout,
        "pass": result.returncode == 0 and "STEP1_OK" in stdout,
    }


def _parse_suites(suites: str | None) -> list[str]:
    if suites is None or suites.strip() == "" or suites.strip().lower() == "parallel":
        return list(DEFAULT_PARALLEL_SUITES)
    items = [item.strip() for item in suites.split(",") if item.strip()]
    if not items:
        raise SystemExit("no suites provided")
    return items


def _print_summary(results: list[dict[str, Any]], label: str) -> None:
    print(f"\n===== {label} =====")
    failed: list[str] = []
    for item in results:
        status = "PASS" if item["pass"] else "FAIL"
        print(f"{status}  suite={item['suite']}  exit={item['exit_code']}")
        if not item["pass"]:
            failed.append(item["suite"])
    print(
        f"TOTAL={len(results)} PASS={len(results) - len(failed)} FAIL={len(failed)}"
    )
    if failed:
        raise SystemExit(f"failed suites: {','.join(failed)}")


@app.local_entrypoint()
def main(suites: str = "parallel") -> None:
    _require_clean_modal_sources()
    key = suites.strip().lower()
    if key == "real1584":
        # Stage 1: ≤1 GPU + ≤2 CPU containers, run then release (no idle hold).
        print("Submitting real1584 Stage-1 jobs (1×A10G + 1×CPU-2thread + 1×opt-load):")
        gpu_h = real1584_gpu.spawn()
        cpu_h = real1584_cpu.spawn()
        opt_h = opt_load_check.spawn()
        results = [gpu_h.get(), cpu_h.get(), opt_h.get()]
        _print_summary(results, "REAL1584 STAGE-1 SUMMARY")
        return
    if key == "stable":
        # One Modal app: ≤1×A10G + 1×CPU-16. Opt/GPU package under test = git HEAD.
        commit = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        )
        print(
            f"Submitting STABLE validation (commit={commit}): "
            "1×CPU-16 + 1×A10G in one app"
        )
        cpu_h = stable_cpu_validate.spawn(commit=commit)
        gpu_h = stable_gpu_validate.spawn()
        results = [cpu_h.get(), gpu_h.get()]
        _print_summary(results, "STABLE VALIDATION SUMMARY")
        return
    if key == "stable_gpu":
        print("Submitting STABLE GPU-only validation (1×A10G):")
        results = [stable_gpu_validate.remote()]
        _print_summary(results, "STABLE GPU SUMMARY")
        return
    if key == "opt_load":
        print("Submitting opt_load_check (1×CPU container):")
        results = [opt_load_check.remote()]
        _print_summary(results, "OPT_LOAD SUMMARY")
        return
    if key == "e2e_grad":
        # Stage 2: single 16-vCPU container, both engines × NT ∈ {20, 96}.
        print("Submitting e2e_grad_cpu16 (1×CPU-16 container, engines=all):")
        results = [e2e_grad_cpu16.remote(engines="all")]
        _print_summary(results, "E2E_GRAD SUMMARY")
        return
    if key == "e2e_grad32":
        # Optional thread-scaling comparison point.
        print("Submitting e2e_grad_cpu32 (1×CPU-32 container, engines=blockdiag):")
        results = [e2e_grad_cpu32.remote(engines="blockdiag")]
        _print_summary(results, "E2E_GRAD32 SUMMARY")
        return
    if key == "sgp4_step1":
        print("Submitting sgp4_step1_check (1×CPU-4thread container):")
        results = [sgp4_step1_check.remote()]
        _print_summary(results, "SGP4 STEP1 SUMMARY")
        return

    suite_list = _parse_suites(suites)
    print(f"Submitting {len(suite_list)} Modal GPU suites in parallel:")
    for name in suite_list:
        print(f"  - {name}")

    results = list(validate_suite.map(suite_list))
    _print_summary(results, "PARALLEL SUITE SUMMARY")
