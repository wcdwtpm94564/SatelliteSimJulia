# ===== 链路评估 =====
# ISL / GSL 物理链路评估，使用 core/geometry.jl + constraints 层。

using LinearAlgebra: norm
using SatelliteToolbox: SatelliteToolbox

export evaluate_isl_batch, evaluate_gsl_batch

# ═══════════════════════════════════════════════
# ISL 评估
# ═══════════════════════════════════════════════

"""
    evaluate_isl(pos_a, pos_b; constraints, vel_a, vel_b, time_horizon)
        -> (available, distance, los, delay_ms, details)

评估一条 ISL 的物理状态，包含距离、LOS、仰角、方位角、持续时间检查。

返回的 details 是命名元组 (elevation_ok, azimuth_ok, duration_ok, elevation_deg, cos_psi, duration_s)。
"""
function evaluate_isl(
    pos_a::NTuple{3,Real},
    pos_b::NTuple{3,Real};
    constraints::PhysicalConstraints=LEO_DEFAULTS,
    vel_a::Union{NTuple{3,Real},Nothing}=nothing,
    vel_b::Union{NTuple{3,Real},Nothing}=nothing,
    time_horizon::Float64=300.0,
    terminal_id::Int=4,  # 默认使用右侧终端
)
    d = distance_km(pos_a, pos_b)
    los = has_los(pos_a, pos_b)

    # 1. 基本检查（距离 + LOS）
    available = check_isl(d, los; constraints=constraints)

    # 2. 仰角检查（需速度计算 RTN）
    elev_ok = true
    elevation = 90.0
    if available && vel_a !== nothing
        r, t, n = compute_rtn_coordinates(pos_a, vel_a, pos_b)
        elevation = compute_elevation_from_rtn(r, t, n)
        elev_ok = check_isl_elevation(elevation; constraints=constraints)
        available &= elev_ok
    end

    # 3. 方位角检查（需速度）
    azim_ok = true
    cos_psi = 1.0
    if available && vel_a !== nothing
        r, t, n = compute_rtn_coordinates(pos_a, vel_a, pos_b)
        cos_psi = compute_azimuth_from_rtn(t, n)
        azim_ok = check_isl_azimuth(cos_psi, terminal_id; constraints=constraints)
        available &= azim_ok
    end

    # 4. 持续时间检查（需双方速度）
    dur_ok = true
    duration = 0.0
    if available && vel_a !== nothing && vel_b !== nothing
        duration = estimate_link_duration(pos_a, vel_a, pos_b, vel_b, time_horizon)
        dur_ok = check_isl_duration(duration; constraints=constraints)
        available &= dur_ok
    end

    delay = propagation_delay_ms(d)

    details = (
        elevation_ok=elev_ok, azimuth_ok=azim_ok, duration_ok=dur_ok,
        elevation_deg=elevation, cos_psi=cos_psi, duration_s=duration,
    )

    return available, d, los, delay, details
end

"""
    estimate_link_duration(pos_a, vel_a, pos_b, vel_b, time_horizon) -> Float64

用直线外推法估算 ISL 在最大距离约束下的持续时长（秒）。
"""
function estimate_link_duration(
    pos_a::NTuple{3,Real},
    vel_a::NTuple{3,Real},
    pos_b::NTuple{3,Real},
    vel_b::NTuple{3,Real},
    time_horizon::Float64;
    constraints::PhysicalConstraints=LEO_DEFAULTS,
)
    rel_pos = Float64.(pos_b .- pos_a)
    rel_vel = Float64.(vel_b .- vel_a)
    max_range = constraints.isl_max_range_km

    # 直线外推：每隔 1s 检查一次距离
    for t in 1.0:1.0:time_horizon
        p = rel_pos .+ t .* rel_vel
        if norm(p) > max_range  # 超过 ISL 最大距离则视为断链
            return t  # 返回可持续秒数
        end
    end
    return time_horizon
end

"""
    evaluate_isl_batch(pos_matrix, isl_pairs; constraints, vel_matrix, time_horizon) -> Vector

批量评估多条 ISL。若有速度矩阵则自动启用激光终端检查。
"""
function evaluate_isl_batch(
    pos_matrix::AbstractMatrix{Float64},
    isl_pairs::Vector{Tuple{Int,Int}};
    constraints::PhysicalConstraints=LEO_DEFAULTS,
    vel_matrix::Union{AbstractMatrix{Float64},Nothing}=nothing,
    time_horizon::Float64=300.0,
)
    results = []
    has_vel = vel_matrix !== nothing
    for (i, j) in isl_pairs
        a = (pos_matrix[i,1], pos_matrix[i,2], pos_matrix[i,3])
        b = (pos_matrix[j,1], pos_matrix[j,2], pos_matrix[j,3])
        va = has_vel ? (vel_matrix[i,1], vel_matrix[i,2], vel_matrix[i,3]) : nothing
        vb = has_vel ? (vel_matrix[j,1], vel_matrix[j,2], vel_matrix[j,3]) : nothing

        avail, d, los, lat, det = evaluate_isl(
            a, b; constraints=constraints,
            vel_a=va, vel_b=vb, time_horizon=time_horizon,
        )
        push!(results, (
            source_id=i, target_id=j,
            available=avail, distance_km=d,
            line_of_sight=los, latency_ms=lat,
            elevation_deg=det.elevation_deg,
            cos_psi=det.cos_psi, duration_s=det.duration_s,
            elevation_ok=det.elevation_ok,
            azimuth_ok=det.azimuth_ok, duration_ok=det.duration_ok,
        ))
    end
    return results
end

# ═══════════════════════════════════════════════
# GSL 评估
# ═══════════════════════════════════════════════

"""
    evaluate_gsl(sat_ecef, gs_lat, gs_lon, gs_alt; constraints=LEO_DEFAULTS)
        -> (available, distance, elevation, delay_ms)

评估一条 GSL 的物理状态。
"""
function evaluate_gsl(
    sat_ecef::NTuple{3,Real},
    gs_lat::Real, gs_lon::Real, gs_alt::Real;
    constraints::PhysicalConstraints=LEO_DEFAULTS,
)
    gs_ecef = geodetic_to_ecef_km(gs_lat, gs_lon, gs_alt)
    d = distance_km(sat_ecef, gs_ecef)
    el = elevation_deg(sat_ecef, gs_lat, gs_lon, gs_alt)
    available = check_gsl(d, el; constraints=constraints)
    delay = propagation_delay_ms(d)
    return available, d, el, delay
end

"""
    evaluate_gsl_batch(pos_matrix, gs_stations; constraints=LEO_DEFAULTS)
        -> (avail_mat, dist_mat, elev_mat, delay_mat)

批量评估 N 颗卫星对 M 个地面站的 GSL。
"""
function evaluate_gsl_batch(
    pos_matrix::AbstractMatrix{Float64},
    gs_stations::Vector{NTuple{3,Float64}};
    constraints::PhysicalConstraints=LEO_DEFAULTS,
)
    N = size(pos_matrix, 1)
    M = length(gs_stations)
    avail_mat = zeros(Bool, N, M)
    dist_mat = zeros(N, M)
    elev_mat = zeros(N, M)
    delay_mat = zeros(N, M)

    for i in 1:N
        sat = (pos_matrix[i,1], pos_matrix[i,2], pos_matrix[i,3])
        for (j, (lat, lon, alt)) in enumerate(gs_stations)
            avail, d, el, lat_ms = evaluate_gsl(sat, lat, lon, alt; constraints=constraints)
            avail_mat[i,j] = avail
            dist_mat[i,j] = d
            elev_mat[i,j] = el
            delay_mat[i,j] = lat_ms
        end
    end
    return avail_mat, dist_mat, elev_mat, delay_mat
end
