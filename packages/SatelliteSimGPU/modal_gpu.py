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


def _repo_is_git_worktree() -> bool:
    """True only when REPO_ROOT is inside a real git work tree.

    Backups / `git archive` mirrors / rsync'd copies are legitimate build roots
    but have no `.git`; `git` exits non-zero there and must not abort the run.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        # git binary missing / not executable.
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _git_head_commit() -> str:
    """Best-effort HEAD sha; never fatal (non-git build roots are supported)."""
    if not _repo_is_git_worktree():
        return "UNKNOWN"
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return "UNKNOWN"
    return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"


def _require_clean_modal_sources() -> None:
    """Refuse to build from uncommitted image sources -- when that is knowable.

    Degrades to a warning (instead of crashing) whenever provenance simply
    cannot be checked: no `.git`, no `git` binary, or `git status` itself
    failing. Previously this raised CalledProcessError inside non-git backup
    directories and the only escape hatch was ``SATSIM_ALLOW_DIRTY_MODAL_SOURCE=1``.
    """
    if os.environ.get(ALLOW_DIRTY_ENV) == "1":
        return
    if not _repo_is_git_worktree():
        print(
            f"WARNING: {REPO_ROOT} is not a git work tree (extracted backup or "
            "git-archive mirror?); skipping the dirty-source preflight. "
            "Image provenance is NOT verifiable for this build."
        )
        return
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *IMAGE_SOURCE_PATHS],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        print(f"WARNING: cannot run git for the Modal source preflight ({exc}); skipping.")
        return
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()[:2]
        print(
            "WARNING: `git status` failed for the Modal source preflight "
            f"({'; '.join(detail) or f'exit {proc.returncode}'}); skipping."
        )
        return
    dirty = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    if dirty:
        preview = "; ".join(dirty[:6])
        raise SystemExit(
            "dirty preflight failed for Modal image sources: "
            f"{preview}. Build from a clean commit (or git archive mirror); "
            f"set {ALLOW_DIRTY_ENV}=1 only for explicit debugging."
        )


_assert_default_suite_contract()

# ---------------------------------------------------------------------------
# Image layer order (reworked 2026-07-26).
#
# Modal/Docker layers are cached strictly in order: the first layer whose inputs
# changed invalidates every layer after it. The previous build put all seven
# `add_local_dir(copy=True)` calls BEFORE `Pkg.instantiate` + CUDA precompile,
# so editing a single character of Julia source invalidated the CUDA
# precompilation layers and forced a full rebuild (~10-20 min). Measured upload
# is only ~3.2 MB / 0.7 s, i.e. transfer was never the bottleneck -- rebuild was.
#
# New order, slowest-and-most-stable first:
#   L1  base CUDA.jl registry image (immutable digest pin)
#   L2  layout scaffolding: /opt/packages symlink (no local inputs)
#   L3  dependency manifests ONLY (Project/Manifest/LocalPreferences x2)
#   L4  Pkg.instantiate + Pkg.add(CUDA, SatelliteToolboxSgp4)   <-- expensive
#   L5  CUDA_Runtime_jll preference inheritance
#   L6  Pkg.precompile() (full CUDA precompile)                 <-- expensive
#   L7  early image contract check (CUDA importable offline)
#   L8  env snapshot: freeze the resolved Project/Manifest/LocalPreferences
#   L9  TLE data file
#   L10 local sources: SatelliteSimBackends -> src/* -> SatelliteSimGPU
#   L11 restore env snapshot + cheap re-precompile + contract re-check
#
# L3..L8 are keyed only on the dependency manifests, so they stay cached across
# ordinary source edits; only L9..L11 rerun, and L11 recompiles just the two
# path-dev packages (seconds) because CUDA's cache is untouched.
# ---------------------------------------------------------------------------

# The only local inputs Pkg needs to resolve/instantiate the GPU environment.
# Kept as an explicit allowlist so no source file can sneak into the cached
# dependency layer. (LocalPreferences/Manifest entries are optional by design.)
DEP_MANIFEST_FILES: list[tuple[Path, str]] = [
    (PACKAGE_DIR / "Project.toml", f"{REMOTE_PACKAGE_DIR}/Project.toml"),
    (PACKAGE_DIR / "Manifest.toml", f"{REMOTE_PACKAGE_DIR}/Manifest.toml"),
    (PACKAGE_DIR / "LocalPreferences.toml", f"{REMOTE_PACKAGE_DIR}/LocalPreferences.toml"),
    # SatelliteSimGPU/Project.toml declares
    #   [sources] SatelliteSimBackends = {path = "../SatelliteSimBackends"}
    # so instantiate resolves through the sibling package's Project.toml.
    (BACKENDS_DIR / "Project.toml", f"{REMOTE_BACKENDS_DIR}/Project.toml"),
    (BACKENDS_DIR / "Manifest.toml", f"{REMOTE_BACKENDS_DIR}/Manifest.toml"),
]
# Resolved environment is frozen here before local sources are copied in, then
# restored afterwards. Required for correctness: `add_local_dir(PACKAGE_DIR)`
# now runs AFTER `Pkg.add`, so it would otherwise overwrite the augmented
# Project/Manifest (dropping CUDA + SatelliteToolboxSgp4) and clobber the
# inherited CUDA_Runtime_jll preference with the checked-in placeholder.
ENV_SNAPSHOT_DIR = "/opt/.satsim-env-snapshot"

# --- L1/L2: base image + layout scaffolding (no local inputs) --------------
image_builder = (
    modal.Image.from_registry(
        "ghcr.io/juliagpu/cuda.jl@sha256:8c40fadfbeea933b98e81a1b164cc3ccb8d442c6caf9e3285e4b577d30d5dd13",
        add_python="3.12",
    )
    .entrypoint([])
    .run_commands(
        # Match src/*/Project.toml [sources] path ../../packages/SatelliteSimBackends.
        # A dangling symlink here is fine; L10 materialises the target.
        f"mkdir -p /opt/packages {SRC_REMOTE} {REMOTE_PACKAGE_DIR} {REMOTE_BACKENDS_DIR} "
        f"&& ln -sfn {REMOTE_BACKENDS_DIR} /opt/packages/SatelliteSimBackends",
    )
)

# --- L3: dependency manifests only -----------------------------------------
for _dep_local, _dep_remote in DEP_MANIFEST_FILES:
    if not _dep_local.exists():
        print(f"NOTE: dependency manifest {_dep_local} absent; not baked into the image.")
        continue
    image_builder = image_builder.add_local_file(str(_dep_local), _dep_remote, copy=True)

# --- L4..L8: everything expensive, keyed only on the manifests above -------
image_builder = image_builder.run_commands(
    # Both path-dev packages must be *loadable shells* for Pkg to resolve and
    # precompile the environment; their real sources arrive in L10. Stubs are
    # overwritten there and Julia re-precompiles them by content hash, which is
    # cheap (SatelliteSimBackends has zero deps, SatelliteSimGPU only light ones).
    f"mkdir -p {REMOTE_BACKENDS_DIR}/src {REMOTE_PACKAGE_DIR}/src "
    f"&& printf 'module SatelliteSimBackends\\nend\\n' "
    f"> {REMOTE_BACKENDS_DIR}/src/SatelliteSimBackends.jl "
    f"&& printf 'module SatelliteSimGPU\\nend\\n' "
    f"> {REMOTE_PACKAGE_DIR}/src/SatelliteSimGPU.jl",
    # Resolve + download only. Auto-precompile is disabled so the stub modules
    # are never the thing that decides whether this layer succeeds.
    "JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=/opt/SatelliteSimGPU -e '"
    "using Pkg; "
    "Pkg.instantiate(); "
    # CUDA is not a SatelliteSimGPU Project.toml dep (CPU KA tests stay light),
    # but modal_gpu_runner.jl needs it on A10G. Pin to runner EXPECTED_CUDA_JL_VERSION.
    "Pkg.add(name=\"CUDA\", version=\"6.2.1\"); "
    "Pkg.add(\"SatelliteToolboxSgp4\")'",
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
    # The expensive one: full CUDA.jl precompile. Cached as long as the
    # dependency manifests (L3) and the inherited preference are unchanged.
    "julia --project=/opt/SatelliteSimGPU -e 'using Pkg; Pkg.precompile()'",
    # Lock the image contract: under the runner's exact offline + narrow load path,
    # CUDA (pinned) and SatelliteToolboxSgp4 must both import. Fails the build early
    # (before uploading sources or spawning GPU containers) if the load path is too
    # narrow to see CUDA.
    "JULIA_PKG_OFFLINE=true JULIA_LOAD_PATH=@:@stdlib "
    "julia --project=/opt/SatelliteSimGPU -e '"
    "using CUDA, SatelliteToolboxSgp4; "
    "pkgversion(CUDA) == v\"6.2.1\" || "
    "error(\"image CUDA \" * string(pkgversion(CUDA)) * \" != 6.2.1\"); "
    "println(\"IMAGE_CUDA_OK cuda=\" * string(pkgversion(CUDA)))'",
    # Freeze the resolved environment so L11 can restore it byte-identically.
    # Byte-identical restore is what keeps the CUDA precompile cache valid.
    f"mkdir -p {ENV_SNAPSHOT_DIR} && "
    f"cp {REMOTE_PACKAGE_DIR}/Project.toml {REMOTE_PACKAGE_DIR}/Manifest.toml "
    f"{ENV_SNAPSHOT_DIR}/ && "
    f"if [ -f {REMOTE_PACKAGE_DIR}/LocalPreferences.toml ]; then "
    f"cp {REMOTE_PACKAGE_DIR}/LocalPreferences.toml {ENV_SNAPSHOT_DIR}/; fi && "
    f"ls -l {ENV_SNAPSHOT_DIR}",
    # Opt deps are large (Enzyme/Zygote/Lux). Instantiate at runtime in opt_load_check
    # so image builds stay short and avoid races with parallel src/opt editors.
)

# --- L9: TLE data (refreshed occasionally, not per-edit) -------------------
image_builder = image_builder.add_local_file(str(TLE_LOCAL), TLE_REMOTE, copy=True)

# --- L10: local sources, least- to most-frequently edited ------------------
image_builder = image_builder.add_local_dir(
    BACKENDS_DIR,
    REMOTE_BACKENDS_DIR,
    copy=True,
    ignore=["**/.DS_Store", "**/__pycache__/**"],
)

for _pkg in OPT_SRC_PACKAGES:
    image_builder = image_builder.add_local_dir(
        str(REPO_ROOT / "src" / _pkg),
        f"{SRC_REMOTE}/{_pkg}",
        copy=True,
        # Parallel workers may edit test/; keep only loadable package content.
        ignore=["**/test/**", "**/.DS_Store", "**/__pycache__/**"],
    )

# Hottest input last: modal_gpu_runner.jl + src/ live here.
image_builder = image_builder.add_local_dir(
    PACKAGE_DIR,
    REMOTE_PACKAGE_DIR,
    copy=True,
    # Ignore churn that would invalidate this layer without changing the build.
    ignore=["**/.DS_Store", "**/__pycache__/**", "*.bak-*", "**/*.bak-*"],
)

# --- L11: restore resolved env, compile the real sources, re-verify --------
image = image_builder.run_commands(
    # add_local_dir(PACKAGE_DIR) just overwrote Project/Manifest/LocalPreferences
    # with the checked-in copies, which do NOT contain CUDA / SatelliteToolboxSgp4
    # and carry only a placeholder CUDA_Runtime_jll. Put the resolved ones back.
    f"cp {ENV_SNAPSHOT_DIR}/Project.toml {ENV_SNAPSHOT_DIR}/Manifest.toml "
    f"{REMOTE_PACKAGE_DIR}/ && "
    f"if [ -f {ENV_SNAPSHOT_DIR}/LocalPreferences.toml ]; then "
    f"cp {ENV_SNAPSHOT_DIR}/LocalPreferences.toml "
    f"{REMOTE_PACKAGE_DIR}/LocalPreferences.toml; fi",
    # Cheap: only SatelliteSimBackends + SatelliteSimGPU changed content hash.
    # CUDA and friends are already cached from L6 and are not recompiled.
    "julia --project=/opt/SatelliteSimGPU -e 'using Pkg; Pkg.precompile()'",
    # Re-assert the contract against the *final* image state, and prove the real
    # SatelliteSimGPU (not the L4 stub) is what loads.
    "JULIA_PKG_OFFLINE=true JULIA_LOAD_PATH=@:@stdlib "
    "julia --project=/opt/SatelliteSimGPU -e '"
    "using CUDA, SatelliteToolboxSgp4, SatelliteSimGPU, SatelliteSimBackends; "
    "pkgversion(CUDA) == v\"6.2.1\" || "
    "error(\"image CUDA \" * string(pkgversion(CUDA)) * \" != 6.2.1\"); "
    # A real symbol from each path-dev package: proves the L4 stubs were
    # replaced by the copied sources and recompiled, not silently reused.
    "isdefined(SatelliteSimGPU, :coverage_loss_gpu) || "
    "error(\"SatelliteSimGPU still resolves to the build stub\"); "
    "isdefined(SatelliteSimBackends, :AbstractOrbitBackend) || "
    "error(\"SatelliteSimBackends still resolves to the build stub\"); "
    "println(\"IMAGE_CUDA_OK cuda=\" * string(pkgversion(CUDA)))'",
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
        # Best-effort: non-git build roots report UNKNOWN instead of crashing.
        commit = _git_head_commit()
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
