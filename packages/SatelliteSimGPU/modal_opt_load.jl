# Cold-start load check for SatelliteSimOpt inside the Modal image.
# Stage 1 only: instantiate + `using` — no gradient / no forward eval.

using Pkg

const OPT_PROJECT = get(ENV, "SATSIM_OPT_PROJECT", "/opt/src/opt")

println("OPT_LOAD_BEGIN project=$OPT_PROJECT")
isdir(OPT_PROJECT) || error("opt project missing at $OPT_PROJECT")
project_toml = joinpath(OPT_PROJECT, "Project.toml")
isfile(project_toml) || error("missing Project.toml at $project_toml")

wall0 = time()
Pkg.activate(OPT_PROJECT)

instantiate_s = @elapsed Pkg.instantiate()
println("OPT_LOAD instantiate_s=$(instantiate_s)")
flush(stdout)

using_s = @elapsed begin
    # `@eval` keeps `using` at top-level of Main so the timing is a real cold load.
    @eval Main begin
        using SatelliteSimOpt
    end
end
total_s = time() - wall0

println(
    "OPT_LOAD status=PASS using_s=$(using_s) total_wall_s=$(total_s) " *
    "julia=$VERSION julia_threads=$(Threads.nthreads()) " *
    "cpu_threads_visible=$(Sys.CPU_THREADS)",
)
try
    println("OPT_LOAD cpu_model=$(replace(Sys.cpu_info()[1].model, ' ' => '_'))")
catch
end
println("MODAL_OPT_LOAD status=PASS")
