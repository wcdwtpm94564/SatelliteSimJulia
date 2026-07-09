# AGENTS.md

## Cursor Cloud specific instructions

Julia monorepo (`SatelliteSimJulia`) for LEO satellite constellation simulation. All packages target **Julia 1.12** (installed via `juliaup`; `julia` is symlinked into `/usr/local/bin` so it works in any shell). There is no web frontend — the product is a Julia library plus an optional WebSocket server.

### Environment / dependencies
- The startup update script runs `julia --project=. -e 'using Pkg; Pkg.develop(path="src/viz"); Pkg.instantiate()'` from the repo root. The `Pkg.develop(path="src/viz")` step is a safeguard: `SatelliteSimViz` is declared in the root `[deps]`/`[extensions]` but historically was missing from `[sources]`, which makes a clean `Pkg.instantiate()` fail with `expected package SatelliteSimViz ... to be registered`. The root `Project.toml` now includes `SatelliteSimViz = {path = "src/viz"}`; the `develop` step keeps startup working even if that fix is not present.
- `Manifest.toml` is git-ignored, so the very first resolve/precompile on a cold machine is heavy (~10 min: Makie, Flux, Enzyme, OrdinaryDiffEq, etc.). The `~/.julia` depot is captured in the VM snapshot, so subsequent `Pkg.instantiate()` runs are fast (a few seconds).
- Sub-projects (`src/server`, `src/lab`, ...) have their own `Project.toml` and are **not** instantiated by the update script. Instantiate them on demand with `julia --project=src/<name> -e 'using Pkg; Pkg.instantiate()'` (fast — packages are already in the depot).

### Run (core library / "hello world")
- `julia --project=. -e 'using SatelliteSimJulia; demo()'` runs the full pipeline end-to-end (Walker constellation → propagation → ISL → routing → coverage → propagator comparison → AI tool listing) with no external data.

### Test
- The documented `julia --project=. -e 'using Pkg; Pkg.test()'` currently **fails at `test/runtests.jl:22`** with `UndefVarError: supports_orbit_elements not defined in SatelliteSimJulia`. This is a pre-existing source/test mismatch (the function lives in `SatelliteSimOrbit` but is never `export`ed, so `SatelliteSimJulia.supports_orbit_elements` doesn't exist), not an environment problem. Do not "fix" it as part of environment setup.
- For end-to-end validation that works today, use the repo's own scripts, e.g. `julia scripts/integration_test.jl` (activates `src/lab`, exercises Core → Net → Lab). Other helpers live in `scripts/` (`validate_packages.jl`, `quick_validate.jl`, `run_regression.jl`).

### Optional WebSocket server (for the external Unity client)
- Start: `julia --project=src/server src/server/bin/serve.jl 8080` → listens on `ws://127.0.0.1:8080`. First start compiles the server package (~40s).
- Protocol is JSON text frames with a `type` field: `list_constellations`, `describe_constellation`, `start_simulation` (server then streams `frame` messages and a final `stream_end`), `stop_simulation`.

### Notes
- There is no configured linter/formatter; precompilation during `Pkg.instantiate()` is the effective build/compile check.
- `GLMakie`/`SatelliteSimViz` visualization needs a working GL stack; viz-related test sets are guarded and skipped when `GLMakie` is unavailable, so it is not required for core work.
- The AI agent REPL (`agent_repl(LLMProvider())`) requires `DEEPSEEK_API_KEY` and outbound HTTPS; it is optional and not needed for core simulation.
