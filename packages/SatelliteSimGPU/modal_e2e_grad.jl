# Stage-2: 1584 real-TLE end-to-end gradient benchmark on a Modal CPU container.
# TLE → flat 7N params (same construction as src/opt/test/test_sgp4_e2e.jl) →
# SatelliteSimOpt.sgp4_e2e_gradient with engine ∈ {:blockdiag, :enzyme} at
# N=1584, G=800 (ground_grid(20,40)), NT ∈ {20, 96}, λ=0.1, dt = real spacing,
# jd_ref = max(epochs). Timing: small-scale compile warmup (off the clock) →
# full-scale warmup → 2 timed runs, report min.
#
# Usage: julia --threads=T --project=/opt/src/opt modal_e2e_grad.jl [all|blockdiag|enzyme]

using Pkg

const OPT_PROJECT = get(ENV, "SATSIM_OPT_PROJECT", "/opt/src/opt")
Pkg.activate(OPT_PROJECT)
setup_instantiate_s = @elapsed Pkg.instantiate()
println("E2E_SETUP instantiate_s=$setup_instantiate_s")
flush(stdout)

load_t0 = time()
using LinearAlgebra: norm
using SatelliteToolbox
using SatelliteToolboxSgp4
using SatelliteSimOpt: sgp4_e2e_gradient, sgp4_series_ecef, coverage_loss_vjp, ground_grid
load_s = time() - load_t0

println(
    "E2E_SETUP load_s=$(round(load_s; digits=2)) julia=$VERSION " *
    "threads=$(Threads.nthreads()) cpu_visible=$(Sys.CPU_THREADS)",
)
try
    println("E2E_SETUP cpu_model=$(replace(Sys.cpu_info()[1].model, ' ' => '_'))")
catch
end
flush(stdout)

const D2R = π / 180
const REV_DAY_TO_RAD_MIN = 2π / (24 * 60)
const TLE_PATH = get(
    ENV,
    "SATSIM_TLE_PATH",
    "/opt/data/tle/celestrak/starlink_gp_latest.tle",
)

# Local M2 Max reference losses (docs/design/differentiable-sgp4.md, Step 2').
const LOCAL_LOSS_REF = Dict(20 => -0.331537585, 96 => -0.331537753)

# Identical parameter construction to test_sgp4_e2e.jl `read_starlink_params_epochs`
# (first n records verbatim, no filtering) so the cloud loss must match local.
function read_starlink_params_epochs(n::Int)
    lines = readlines(TLE_PATH)
    tles = SatelliteToolboxSgp4.TLE[]
    for i in 1:n
        l1 = strip(lines[3i - 1])
        l2 = strip(lines[3i])
        push!(tles, SatelliteToolboxSgp4.read_tle(l1, l2; verify_checksum=false))
    end
    epochs = Float64[SatelliteToolboxSgp4.tle_epoch(t) for t in tles]
    params = Float64[]
    for t in tles
        append!(params, (
            t.mean_motion * REV_DAY_TO_RAD_MIN,
            t.eccentricity,
            t.inclination * D2R,
            t.raan * D2R,
            t.argument_of_perigee * D2R,
            t.mean_anomaly * D2R,
            t.bstar,
        ))
    end
    return params, epochs
end

function timed_gradient(
    engine::Symbol, params, epochs, ts_min, gp, w;
    jd_ref, dt, n_timed::Int=2,
)
    call() = sgp4_e2e_gradient(
        params, epochs, ts_min, gp, w;
        jd_ref=jd_ref, engine=engine, dt=dt,
        min_el=10.0, τ_cov=5.0, τ_revisit=1.0, λ=0.1,
    )
    t0 = time()
    loss, grad = call()
    warm_s = time() - t0
    times = Float64[]
    for _ in 1:n_timed
        t0 = time()
        loss, grad = call()
        push!(times, time() - t0)
    end
    return loss, grad, warm_s, times
end

function main()
    engines_arg = isempty(ARGS) ? "all" : lowercase(ARGS[1])
    engines = engines_arg == "all" ? [:blockdiag, :enzyme] : [Symbol(engines_arg)]
    all(e -> e in (:blockdiag, :enzyme), engines) ||
        error("unknown engine arg $engines_arg (use all|blockdiag|enzyme)")

    N = parse(Int, get(ENV, "SATSIM_E2E_N", "1584"))
    t0 = time()
    params, epochs = read_starlink_params_epochs(N)
    read_s = time() - t0
    jd_ref = maximum(epochs)
    gp, w = ground_grid(20, 40)   # G = 800
    println(
        "E2E_CONFIG N=$N G=$(size(gp, 1)) NTs=20,96 lambda=0.1 jd_ref=$jd_ref " *
        "epoch_span_min=$(round((jd_ref - minimum(epochs)) * 1440; digits=1)) " *
        "tle=$TLE_PATH read_s=$(round(read_s; digits=3))",
    )
    flush(stdout)

    # Small-scale warmup: absorb Enzyme/ForwardDiff LLVM compilation off the clock.
    p_small = params[1:70]
    e_small = epochs[1:10]
    ts_small = collect(range(0.0, 20.0; length=5))
    gp_small, w_small = ground_grid(5, 10)
    for engine in engines
        t0 = time()
        sgp4_e2e_gradient(
            p_small, e_small, ts_small, gp_small, w_small;
            jd_ref=jd_ref, engine=engine, dt=ts_small[2] - ts_small[1],
            min_el=10.0, τ_cov=5.0, τ_revisit=1.0, λ=0.1,
        )
        println("E2E_COMPILE engine=$engine small_scale_s=$(round(time() - t0; digits=2))")
        flush(stdout)
    end

    grads = Dict{Tuple{Symbol,Int},Vector{Float64}}()
    for NT in (20, 96)
        ts_min = collect(range(0.0, 95.0; length=NT))
        dt = ts_min[2] - ts_min[1]
        for engine in engines
            loss, grad, warm_s, times = timed_gradient(
                engine, params, epochs, ts_min, gp, w;
                jd_ref=jd_ref, dt=dt, n_timed=2,
            )
            grads[(engine, NT)] = grad
            finite = isfinite(loss) && all(isfinite, grad)
            nonzero = count(!iszero, grad)
            ref = get(LOCAL_LOSS_REF, NT, NaN)
            println(
                "E2E_GRAD engine=$engine NT=$NT G=$(size(gp, 1)) dt=$dt lambda=0.1 " *
                "loss=$loss grad_norm=$(norm(grad)) finite=$finite " *
                "nonzero=$nonzero/$(length(grad)) " *
                "warm_s=$(round(warm_s; digits=3)) " *
                "times_s=$(join(round.(times; digits=3), ",")) " *
                "min_s=$(round(minimum(times); digits=3)) " *
                "local_ref_loss=$ref delta_vs_local=$(loss - ref) " *
                "threads=$(Threads.nthreads())",
            )
            flush(stdout)
            finite || error("non-finite loss/grad for engine=$engine NT=$NT")
        end
        if length(engines) == 2
            g_e = grads[(:enzyme, NT)]
            g_b = grads[(:blockdiag, NT)]
            println("E2E_XCHECK NT=$NT engines_rel_l2=$(norm(g_e - g_b) / norm(g_e))")
        end
        # Phase breakdown: shows the single-threaded handwritten adjoint (vjp)
        # dominates blockdiag, i.e. how much extra cores can/cannot help.
        gmsts = Float64[SatelliteToolbox.jd_to_gmst(jd_ref + t / 1440.0) for t in ts_min]
        t0 = time()
        pos = sgp4_series_ecef(params, epochs, ts_min, gmsts; jd_ref=jd_ref)
        series_s = time() - t0
        t0 = time()
        coverage_loss_vjp(
            pos, gp, w;
            min_el=10.0, τ_cov=5.0, dt=dt, τ_revisit=1.0, λ=0.1,
        )
        vjp_s = time() - t0
        println(
            "E2E_PHASES NT=$NT series_s=$(round(series_s; digits=3)) " *
            "vjp_single_thread_s=$(round(vjp_s; digits=3))",
        )
        flush(stdout)
    end

    println("E2E_MAXRSS_GIB=$(round(Sys.maxrss() / 2^30; digits=2))")
    println("MODAL_E2E_GRAD status=PASS")
    return nothing
end

main()
