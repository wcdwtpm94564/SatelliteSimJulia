# 基准入口：完整下载 vs 设备聚合（P1 归约变体）
#
# 目的：量化"只要摘要"的调用方省下的**传输量**。真机基准发现 GSL/ISL 的端到端加速被
# 下载 (N,M,NT) / (pairs,NT) 大数组吃掉；本脚本对比两条路径：
#   1. full-download：设备算完整链路矩阵 → 下载完整数组 → host 归约得摘要；
#   2. device-aggregate：设备端归约核直接算摘要 → 只下载小结果。
#
# 本脚本用 KernelAbstractions **CPU 后端**（无 GPU 也能跑）：下载在 CPU 上近乎恒等，
# 故这里报告的是 (a) **传输元素/字节量对比**（GPU 上的真实收益来源）与 (b) CPU 后端上的
# 计算耗时。GPU 端到端（含 H2D/D2H）的对比见 modal_gpu_runner.jl 的 bench_reduction_case。
#
# 用法：julia --project=. bench_reduction.jl [N M NT]
#   N=卫星数, M=地面站数, NT=时刻数（默认 550 40 90）。

using Printf
using Random
using KernelAbstractions
using SatelliteSimGPU

function random_positions(n_satellites, n_times, T)
    positions = Array{T}(undef, n_satellites, n_times, 3)
    for satellite_index in 1:n_satellites, time_index in 1:n_times
        direction = randn(T, 3)
        direction ./= sqrt(sum(abs2, direction))
        positions[satellite_index, time_index, :] .= T(6900.0 + 100.0 * rand()) .* direction
    end
    return positions
end

function random_velocities(n_satellites, n_times, T)
    velocities = Array{T}(undef, n_satellites, n_times, 3)
    for satellite_index in 1:n_satellites, time_index in 1:n_times
        direction = randn(T, 3)
        direction ./= sqrt(sum(abs2, direction))
        velocities[satellite_index, time_index, :] .= T(7.6) .* direction
    end
    return velocities
end

function random_stations(n_stations)
    return [
        (
            -70.0 + 140.0 * (k - 1) / max(n_stations - 1, 1),
            mod(37.0 * k + 13.0, 360.0) - 180.0,
            0.2 + 1.5 * mod(k, 5) / 4,
        )
        for k in 1:n_stations
    ]
end

function make_pairs(n_satellites, n_pairs)
    pairs = Tuple{Int,Int}[]
    stride = 1
    while length(pairs) < n_pairs && stride < n_satellites
        for i in 1:n_satellites
            length(pairs) >= n_pairs && break
            j = mod(i - 1 + stride, n_satellites) + 1
            i == j && continue
            push!(pairs, (i, j))
        end
        stride += 1
    end
    return pairs
end

best(f, samples=3) = minimum(@elapsed(f()) for _ in 1:samples)

const N, M, NT = length(ARGS) == 3 ? parse.(Int, ARGS) : (550, 40, 90)
Random.seed!(1234)
positions = random_positions(N, NT, Float64)
velocities = random_velocities(N, NT, Float64)
stations = random_stations(M)
ground_ecef, ned_rotation = SatelliteSimGPU._gsl_station_geometry(stations, Float64)
n_pairs = 2 * N
pairs = make_pairs(N, n_pairs)
n_pairs = length(pairs)

println("BENCH_REDUCTION_BEGIN backend=", string(typeof(get_backend(positions))))
@printf("scenario N=%d M=%d NT=%d pairs=%d\n", N, M, NT, n_pairs)

# ── GSL：完整下载 + host 归约  vs  设备聚合 ──
gsl_full_reduce() = begin
    full = evaluate_gsl_batch_gpu(
        positions, ground_ecef, ned_rotation;
        gsl_min_elevation_deg=25.0, gsl_max_range_km=2000.0,
    )
    host = map(SatelliteSimGPU.to_host, full)      # 下载完整 (N,M,NT)×4
    Int32.(dropdims(sum(host[1]; dims=1); dims=1))  # host 归约 → (M,NT)
end
gsl_aggregate() = SatelliteSimGPU.to_host(gsl_visible_counts_gpu(
    positions, ground_ecef, ned_rotation;
    gsl_min_elevation_deg=25.0, gsl_max_range_km=2000.0,
))
@assert gsl_full_reduce() == gsl_aggregate()
gsl_full_s = best(gsl_full_reduce)
gsl_agg_s = best(gsl_aggregate)
gsl_full_elems = N * M * NT           # available (忽略另外 3 个 float 数组，只算最小可比项)
gsl_agg_elems = M * NT
@printf(
    "GSL_REDUCTION visible_counts full_download_s=%.6f device_aggregate_s=%.6f speedup=%.2f download_elems_full=%d download_elems_agg=%d transfer_reduction=%.1fx\n",
    gsl_full_s, gsl_agg_s, gsl_full_s / gsl_agg_s,
    gsl_full_elems, gsl_agg_elems, gsl_full_elems / gsl_agg_elems,
)

gsl_ratio_agg() = SatelliteSimGPU.to_host(gsl_station_visible_ratio_gpu(
    positions, ground_ecef, ned_rotation;
    gsl_min_elevation_deg=25.0, gsl_max_range_km=2000.0,
))
gsl_ratio_s = best(gsl_ratio_agg)
@printf(
    "GSL_REDUCTION station_ratio device_aggregate_s=%.6f download_elems_full=%d download_elems_agg=%d transfer_reduction=%.1fx\n",
    gsl_ratio_s, gsl_full_elems, M, gsl_full_elems / M,
)

# ── ISL：完整下载 + host 归约  vs  设备聚合 ──
isl_full_reduce() = begin
    full = evaluate_isl_batch_gpu(positions, pairs; velocities=velocities)
    avail = SatelliteSimGPU.to_host(full.available)  # 下载完整 (pairs,NT)
    Int32.(vec(sum(avail; dims=1)))                  # host 归约 → (NT,)
end
isl_aggregate() =
    SatelliteSimGPU.to_host(isl_available_counts_gpu(positions, pairs; velocities=velocities))
@assert isl_full_reduce() == isl_aggregate()
isl_full_s = best(isl_full_reduce)
isl_agg_s = best(isl_aggregate)
isl_full_elems = n_pairs * NT
@printf(
    "ISL_REDUCTION available_counts full_download_s=%.6f device_aggregate_s=%.6f speedup=%.2f download_elems_full=%d download_elems_agg=%d transfer_reduction=%.1fx\n",
    isl_full_s, isl_agg_s, isl_full_s / isl_agg_s,
    isl_full_elems, NT, isl_full_elems / NT,
)

isl_ratio_agg() = SatelliteSimGPU.to_host(
    isl_pair_available_ratio_gpu(positions, pairs; velocities=velocities),
)
isl_ratio_s = best(isl_ratio_agg)
@printf(
    "ISL_REDUCTION pair_ratio device_aggregate_s=%.6f download_elems_full=%d download_elems_agg=%d transfer_reduction=%.1fx\n",
    isl_ratio_s, isl_full_elems, n_pairs, isl_full_elems / n_pairs,
)

# P3 下游：每卫星可用链路度（(N,NT)，不下载 (pairs,NT)）
isl_degree_agg() = SatelliteSimGPU.to_host(
    isl_satellite_degree_gpu(positions, pairs; velocities=velocities),
)
isl_degree_s = best(isl_degree_agg)
@printf(
    "ISL_REDUCTION satellite_degree device_aggregate_s=%.6f download_elems_full=%d download_elems_agg=%d transfer_reduction=%.1fx\n",
    isl_degree_s, isl_full_elems, N * NT, isl_full_elems / (N * NT),
)

println("BENCH_REDUCTION_END")
