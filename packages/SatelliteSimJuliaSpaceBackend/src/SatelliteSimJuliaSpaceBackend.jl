module SatelliteSimJuliaSpaceBackend

using SatelliteSimBackends
using SatelliteToolbox
import SatelliteSimBackends: backend_capabilities, propagate_orbit, register_orbit_backend!

export JuliaSpaceOrbitBackend

# EGM2008 constants used by the independent secular-element implementation.
# They intentionally match the physical constants selected by the Native Orbit
# package, while the propagation code path below does not call Native or the
# SatelliteToolbox propagator implementations.
const _EARTH_RADIUS_M = 6_378_137.0
const _EARTH_MU_M3_S2 = 3.986004415e14
const _EARTH_J2 = 0.0010826261738522227
const _EARTH_J4 = -1.6198975999169731e-6
const _NORMALIZED_MU = sqrt(_EARTH_MU_M3_S2 / _EARTH_RADIUS_M^3)
const _SUPPORTED_PROPAGATORS = (:two_body, :j2, :j4)

"""Independent secular Kepler/J2/J4 implementation behind the stable backend contract."""
Base.@kwdef struct JuliaSpaceOrbitBackend <: AbstractOrbitBackend
    propagator::Symbol = :two_body
end

backend_capabilities(::JuliaSpaceOrbitBackend) = (
    frames = (:ecef,),
    deterministic = true,
    propagators = _SUPPORTED_PROPAGATORS,
    implementation = :independent_secular_elements,
)

function _true_to_mean_anomaly(e::Float64, f::Float64)::Float64
    eccentric_anomaly = 2 * atan(
        sqrt(1 - e) * sin(f / 2),
        sqrt(1 + e) * cos(f / 2),
    )
    return eccentric_anomaly - e * sin(eccentric_anomaly)
end

function _mean_to_true_anomaly(e::Float64, mean_anomaly::Float64)::Float64
    normalized_mean = mod(mean_anomaly, 2π)
    eccentric_anomaly = e < 0.8 ? normalized_mean : π
    for _ in 1:20
        residual = eccentric_anomaly - e * sin(eccentric_anomaly) - normalized_mean
        correction = residual / (1 - e * cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        abs(correction) <= 8eps(Float64) && break
    end
    denominator = 1 - e * cos(eccentric_anomaly)
    sin_f = sqrt(1 - e^2) * sin(eccentric_anomaly) / denominator
    cos_f = (cos(eccentric_anomaly) - e) / denominator
    return atan(sin_f, cos_f)
end

function _secular_rates(element, propagator::Symbol)
    a = Float64(element.a)
    e = Float64(element.e)
    inclination = Float64(element.i)
    a > 0 || throw(ArgumentError("semi-major axis must be positive"))
    0 <= e < 1 || throw(ArgumentError("eccentricity must be in [0, 1)"))

    normalized_a = a / _EARTH_RADIUS_M
    e2 = e^2
    p = normalized_a * (1 - e2)
    p2 = p^2
    n0 = _NORMALIZED_MU / sqrt(normalized_a^3)
    propagator === :two_body && return n0, 0.0, 0.0

    sin_i, cos_i = sincos(inclination)
    sin_i2 = sin_i^2
    beta2 = 1 - e2
    beta = sqrt(beta2)

    if propagator === :j2
        kn2 = _EARTH_J2 / p2 * beta
        mean_motion = n0 * (1 + 3 / 4 * kn2 * (2 - 3sin_i2))
        k2_bar = mean_motion * _EARTH_J2 / p2
        raan_rate = -3 / 2 * k2_bar * cos_i
        argp_rate = 3 / 4 * k2_bar * (4 - 5sin_i2)
        return mean_motion, raan_rate, argp_rate
    end

    propagator === :j4 || throw(ArgumentError(
        "unsupported JuliaSpace propagator :$propagator; expected one of " *
        join(":" .* String.(_SUPPORTED_PROPAGATORS), ", "),
    ))

    p4 = p^4
    sin_i4 = sin_i^4
    cos_i4 = cos_i^4
    j2_squared = _EARTH_J2^2
    kn2 = _EARTH_J2 / p2 * beta
    kn22 = j2_squared / p4 * beta
    kn4 = _EARTH_J4 / p4 * beta
    mean_motion = n0 * (
        1 +
        3 / 4 * kn2 * (2 - 3sin_i2) +
        3 / 128 * kn22 * (
            120 + 64beta - 40beta2 +
            (-240 - 192beta + 40beta2) * sin_i2 +
            (105 + 144beta + 25beta2) * sin_i4
        ) -
        45 / 128 * kn4 * e2 * (-8 + 40sin_i2 - 35sin_i4)
    )

    k2_bar = mean_motion * _EARTH_J2 / p2
    k22_bar = mean_motion * j2_squared / p4
    k22 = n0 * j2_squared / p4
    k4 = n0 * _EARTH_J4 / p4

    raan_rate =
        -3 / 2 * k2_bar * cos_i +
        3 / 32 * k22_bar * cos_i * (
            -36 - 4 * e2 + 48beta + (40 - 5 * e2 - 72beta) * sin_i2
        ) +
        15 / 32 * k4 * cos_i * (8 + 12 * e2 - (14 + 21 * e2) * sin_i2)

    argp_rate =
        3 / 4 * k2_bar * (4 - 5sin_i2) +
        3 / 128 * k22_bar * (
            384 + 96 * e2 - 384beta +
            (-824 - 116 * e2 + 1056beta) * sin_i2 +
            (430 - 5 * e2 - 720beta) * sin_i4
        ) -
        15 / 16 * k22 * e2 * cos_i4 -
        15 / 128 * k4 * (
            64 + 72 * e2 - (248 + 252 * e2) * sin_i2 +
            (196 + 189 * e2) * sin_i4
        )

    return mean_motion, raan_rate, argp_rate
end

function _eci_position_m(element, elapsed_s::Float64, propagator::Symbol)
    a = Float64(element.a)
    e = Float64(element.e)
    inclination = Float64(element.i)
    initial_raan = Float64(element.Ω)
    initial_argp = Float64(element.ω)
    initial_true_anomaly = Float64(element.f)

    mean_motion, raan_rate, argp_rate = _secular_rates(element, propagator)
    initial_mean_anomaly = _true_to_mean_anomaly(e, initial_true_anomaly)
    true_anomaly = _mean_to_true_anomaly(e, initial_mean_anomaly + mean_motion * elapsed_s)
    raan = mod(initial_raan + raan_rate * elapsed_s, 2π)
    argument_of_perigee = mod(initial_argp + argp_rate * elapsed_s, 2π)

    radius = a * (1 - e^2) / (1 + e * cos(true_anomaly))
    argument_of_latitude = argument_of_perigee + true_anomaly
    sin_raan, cos_raan = sincos(raan)
    sin_u, cos_u = sincos(argument_of_latitude)
    sin_i, cos_i = sincos(inclination)

    x = radius * (cos_raan * cos_u - sin_raan * sin_u * cos_i)
    y = radius * (sin_raan * cos_u + cos_raan * sin_u * cos_i)
    z = radius * (sin_u * sin_i)
    return x, y, z
end

function _ecef_position_km(element, elapsed_s::Float64, propagator::Symbol)
    x, y, z = _eci_position_m(element, elapsed_s, propagator)
    rotation = SatelliteToolbox.r_eci_to_ecef(
        SatelliteToolbox.TEME(),
        SatelliteToolbox.PEF(),
        elapsed_s / 86_400.0,
    )
    ecef_x = rotation[1, 1] * x + rotation[1, 2] * y + rotation[1, 3] * z
    ecef_y = rotation[2, 1] * x + rotation[2, 2] * y + rotation[2, 3] * z
    ecef_z = rotation[3, 1] * x + rotation[3, 2] * y + rotation[3, 3] * z
    return ecef_x / 1000, ecef_y / 1000, ecef_z / 1000
end

function _independent_positions(elements, times::Vector{Float64}, propagator::Symbol)
    propagator in _SUPPORTED_PROPAGATORS || throw(ArgumentError(
        "unsupported JuliaSpace propagator :$propagator; expected one of " *
        join(":" .* String.(_SUPPORTED_PROPAGATORS), ", "),
    ))
    all(isfinite, times) || throw(ArgumentError("propagation times must be finite"))
    positions = Array{Float64,3}(undef, length(elements), length(times), 3)
    Threads.@threads for satellite_index in eachindex(elements)
        element = elements[satellite_index]
        for (time_index, elapsed_s) in pairs(times)
            x, y, z = _ecef_position_km(element, elapsed_s, propagator)
            positions[satellite_index, time_index, 1] = x
            positions[satellite_index, time_index, 2] = y
            positions[satellite_index, time_index, 3] = z
        end
    end
    return positions
end

function propagate_orbit(backend::JuliaSpaceOrbitBackend, elements, tspan; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "JuliaSpaceOrbitBackend does not accept propagation keyword arguments; " *
        "configure it through OrbitBackendSpec options",
    ))
    times = Float64.(collect(tspan))
    positions = _independent_positions(elements, times, backend.propagator)
    result = OrbitResult(positions, Dict{String,Any}(
        "backend" => backend_name(backend),
        "frame" => "ecef",
        "propagator" => String(backend.propagator),
        "implementation" => "independent_secular_elements",
        "frame_transform" => "SatelliteToolbox TEME-to-PEF",
    ))
    return validate_orbit_result(
        result;
        expected_satellites=length(elements),
        expected_times=length(times),
    )
end

function __init__()
    register_orbit_backend!(
        :julia_space,
        options -> JuliaSpaceOrbitBackend(; options...);
        replace=true,
    )
    return nothing
end

end
