# Stable Modal validation against SatelliteSimOpt at a pinned git commit.
# Headline only: connectivity λ₂ @ N=1584 NT=4; soft latency @ N=200;
# coverage e2e @ N=1584 NT=20. Correctness first, then min-of-2 timed runs.
#
# Usage:
#   julia --threads=16 --project=/opt/src/opt modal_stable_cpu.jl

using Pkg

const OPT_PROJECT = get(ENV, "SATSIM_OPT_PROJECT", "/opt/src/opt")
const GIT_COMMIT = get(ENV, "SATSIM_GIT_COMMIT", "UNKNOWN")
const TLE_PATH = get(
    ENV,
    "SATSIM_TLE_PATH",
    "/opt/data/tle/celestrak/starlink_gp_latest.tle",
)
const LOCAL_E2E_LOSS_NT20 = -0.331537585
const CONN_RELERR_BUDGET = 1e-8   # docs claim ~1e-11; allow float noise floor

Pkg.activate(OPT_PROJECT)
inst_s = @elapsed Pkg.instantiate()
println("STABLE_CPU commit=$GIT_COMMIT instantiate_s=$(round(inst_s; digits=2))")
flush(stdout)

load_t0 = time()
using LinearAlgebra
using SatelliteToolbox
using SatelliteToolboxSgp4
using SatelliteSimOpt:
    sgp4_e2e_gradient,
    sgp4_series_ecef,
    sgp4_network_kpi_gradient,
    network_kpi_config,
    soft_connectivity_loss_vjp,
    soft_algebraic_connectivity,
    soft_isl_adjacency,
    soft_expected_latency_ms,
    hard_isl_adjacency,
    dijkstra_latency,
    network_kpi_loss_grad_positions,
    default_od_pairs,
    ground_grid
load_s = time() - load_t0

println(
    "STABLE_CPU load_s=$(round(load_s; digits=2)) julia=$VERSION " *
    "threads=$(Threads.nthreads()) cpu_visible=$(Sys.CPU_THREADS)",
)
flush(stdout)

const D2R = π / 180
const REV_DAY_TO_RAD_MIN = 2π / (24 * 60)

function read_params_epochs(n::Int)
    lines = readlines(TLE_PATH)
    avail = length(lines) ÷ 3
    n = min(n, avail)
    tles = SatelliteToolboxSgp4.TLE[]
    for i in 1:n
        push!(
            tles,
            SatelliteToolboxSgp4.read_tle(
                strip(lines[3i - 1]),
                strip(lines[3i]);
                verify_checksum=false,
            ),
        )
    end
    epochs = Float64[SatelliteToolboxSgp4.tle_epoch(t) for t in tles]
    params = Float64[]
    for t in tles
        append!(
            params,
            (
                t.mean_motion * REV_DAY_TO_RAD_MIN,
                t.eccentricity,
                t.inclination * D2R,
                t.raan * D2R,
                t.argument_of_perigee * D2R,
                t.mean_anomaly * D2R,
                t.bstar,
            ),
        )
    end
    return params, epochs, n
end

function timed_min(f; n_warm::Int=1, n_timed::Int=2)
    local out
    for _ in 1:n_warm
        out = f()
    end
    times = Float64[]
    for _ in 1:n_timed
        t0 = time()
        out = f()
        push!(times, time() - t0)
    end
    return out, times, minimum(times)
end

function run_connectivity()
    N_want, NT = 1584, 4
    params, epochs, N = read_params_epochs(N_want)
    jd_ref = maximum(epochs)
    ts_min = collect(range(0.0, 90.0; length=NT))
    gmsts = Float64[SatelliteToolbox.jd_to_gmst(jd_ref + t / 1440.0) for t in ts_min]
    d_thresh, τ, fiedler_K = 5500.0, 200.0, 1500

    pos = sgp4_series_ecef(params, epochs, ts_min, gmsts; jd_ref=jd_ref)
    cfg = network_kpi_config(N, NT; kind=:connectivity, d_thresh=d_thresh, τ=τ, fiedler_K=fiedler_K)

    # Correctness: power-iter λ₂ vs exact eigvals at t=1.
    A = soft_isl_adjacency(pos[:, 1, :]; d_thresh=d_thresh, τ=τ)
    d = vec(sum(A; dims=2))
    L = Diagonal(d) - A
    t_eig = @elapsed ev = sort(eigvals(Symmetric(Matrix(L))))
    λ2_exact = ev[2]
    λ2_soft = soft_algebraic_connectivity(
        pos[:, 1:1, :];
        d_thresh=d_thresh,
        τ=τ,
        fiedler_K=fiedler_K,
    )
    relerr = abs(λ2_soft - λ2_exact) / abs(λ2_exact)
    println(
        "STABLE_CONN hard_ref N=$N NT=$NT λ2_exact=$λ2_exact λ2_power=$λ2_soft " *
        "relerr=$relerr gap=$(ev[3] - ev[2]) eig_s=$(round(t_eig; digits=2))",
    )
    relerr <= CONN_RELERR_BUDGET ||
        error("connectivity λ₂ relerr $relerr exceeds budget $CONN_RELERR_BUDGET")

    (loss, grad), times, min_s = timed_min() do
        sgp4_network_kpi_gradient(
            params,
            epochs,
            ts_min;
            jd_ref=jd_ref,
            gmsts=gmsts,
            engine=:blockdiag,
            kind=:connectivity,
            d_thresh=d_thresh,
            τ=τ,
            fiedler_K=fiedler_K,
        )
    end
    finite = isfinite(loss) && all(isfinite, grad)
    println(
        "STABLE_CONN grad N=$N NT=$NT loss=$loss mean_lambda2=$(-loss) " *
        "grad_norm=$(norm(grad)) finite=$finite nonzero=$(count(!iszero, grad))/$(length(grad)) " *
        "times_s=$(join(round.(times; digits=3), ",")) min_s=$(round(min_s; digits=3)) " *
        "timing=warmup1_then_min2_excludes_compile engine=blockdiag",
    )
    finite || error("connectivity gradient non-finite")
    return nothing
end

function run_latency()
    N_want, NT = 200, 6
    params, epochs, N = read_params_epochs(N_want)
    jd_ref = maximum(epochs)
    ts_min = collect(range(0.0, 90.0; length=NT))
    gmsts = Float64[SatelliteToolbox.jd_to_gmst(jd_ref + t / 1440.0) for t in ts_min]
    pos = sgp4_series_ecef(params, epochs, ts_min, gmsts; jd_ref=jd_ref)

    # Same threshold construction as src/opt/scripts/sgp4_step3_network_kpi.jl
    ds = Float64[]
    for i in 1:N, j in (i + 1):N
        push!(
            ds,
            sqrt(
                (pos[i, 1, 1] - pos[j, 1, 1])^2 +
                (pos[i, 1, 2] - pos[j, 1, 2])^2 +
                (pos[i, 1, 3] - pos[j, 1, 3])^2,
            ),
        )
    end
    dmed = sort(ds)[max(1, length(ds) ÷ 2)]
    d_thresh = 1.15 * dmed
    τ, τsp, bellman_K = 400.0, 80.0, 24
    od = default_od_pairs(N; count=16)
    kw = (d_thresh=d_thresh, τ=τ, τsp=τsp, bellman_K=bellman_K, penalty_km=5.0e5)

    Ah = hard_isl_adjacency(pos[:, 1, :]; d_thresh=d_thresh)
    lat_hard, _ = dijkstra_latency(Ah; od_pairs=[(s, d) for (s, d) in od])
    soft_ms_t1 = soft_expected_latency_ms(pos[:, 1:1, :]; od_pairs=od, kw...)
    mean_hard = isempty(lat_hard) ? NaN : sum(lat_hard) / length(lat_hard)
    println(
        "STABLE_LAT hard_ref N=$N NT=$NT d_thresh=$(round(d_thresh; digits=1)) " *
        "dijkstra_mean_ms=$mean_hard soft_ms_t1=$soft_ms_t1 reachable=$(length(lat_hard))/$(length(od))",
    )

    (loss, grad), times, min_s = timed_min() do
        sgp4_network_kpi_gradient(
            params,
            epochs,
            ts_min;
            jd_ref=jd_ref,
            gmsts=gmsts,
            engine=:blockdiag,
            kind=:latency,
            od_pairs=od,
            kw...,
        )
    end
    finite = isfinite(loss) && all(isfinite, grad)
    println(
        "STABLE_LAT grad N=$N NT=$NT loss_ms=$loss grad_norm=$(norm(grad)) " *
        "finite=$finite nonzero=$(count(!iszero, grad))/$(length(grad)) " *
        "times_s=$(join(round.(times; digits=3), ",")) min_s=$(round(min_s; digits=3)) " *
        "timing=warmup1_then_min2_excludes_compile engine=blockdiag",
    )
    finite || error("latency gradient non-finite")
    return nothing
end

function run_coverage_e2e()
    N_want, NT, G = 1584, 20, 800
    params, epochs, N = read_params_epochs(N_want)
    jd_ref = maximum(epochs)
    ts_min = collect(range(0.0, 95.0; length=NT))
    dt = ts_min[2] - ts_min[1]
    gp, w = ground_grid(20, 40)  # G=800
    size(gp, 1) == G || error("expected G=$G, got $(size(gp, 1))")

    # Small compile warmup (off the clock for headline min).
    sgp4_e2e_gradient(
        params[1:70],
        epochs[1:10],
        ts_min[1:5],
        ground_grid(5, 10)...;
        jd_ref=jd_ref,
        engine=:blockdiag,
        dt=ts_min[2] - ts_min[1],
        λ=0.1,
    )

    (loss, grad), times, min_s = timed_min(; n_warm=1, n_timed=2) do
        sgp4_e2e_gradient(
            params,
            epochs,
            ts_min,
            gp,
            w;
            jd_ref=jd_ref,
            engine=:blockdiag,
            dt=dt,
            λ=0.1,
        )
    end
    delta = loss - LOCAL_E2E_LOSS_NT20
    finite = isfinite(loss) && all(isfinite, grad)
    println(
        "STABLE_E2E grad N=$N NT=$NT G=$G dt=$dt lambda=0.1 loss=$loss " *
        "local_ref=$LOCAL_E2E_LOSS_NT20 delta_vs_local=$delta " *
        "grad_norm=$(norm(grad)) finite=$finite nonzero=$(count(!iszero, grad))/$(length(grad)) " *
        "times_s=$(join(round.(times; digits=3), ",")) min_s=$(round(min_s; digits=3)) " *
        "timing=small_compile_plus_warmup1_then_min2 engine=blockdiag",
    )
    finite || error("e2e gradient non-finite")
    abs(delta) < 1e-8 || error("e2e loss delta $delta vs local exceeds 1e-8")
    return nothing
end

function main()
    println("STABLE_CPU_BEGIN commit=$GIT_COMMIT tle=$TLE_PATH")
    run_connectivity()
    run_latency()
    run_coverage_e2e()
    println("STABLE_CPU_MAXRSS_GIB=$(round(Sys.maxrss() / 2^30; digits=2))")
    println("MODAL_STABLE_CPU status=PASS commit=$GIT_COMMIT")
    return nothing
end

main()
