module SatelliteSimStubBackend

using SatelliteSimBackends
import SatelliteSimBackends: backend_capabilities, propagate_orbit, register_orbit_backend!

export StubOrbitBackend

"""Deterministic, dependency-light backend for offline CI and contract tests."""
Base.@kwdef struct StubOrbitBackend <: AbstractOrbitBackend
    origin_ecef_km::NTuple{3,Float64} = (7000.0, 0.0, 0.0)
    velocity_ecef_km_s::NTuple{3,Float64} = (0.0, 7.5, 0.0)
    satellite_spacing_km::Float64 = 10.0
end

backend_capabilities(::StubOrbitBackend) = (frames = (:ecef,), deterministic = true)

function propagate_orbit(backend::StubOrbitBackend, elements, tspan; kwargs...)
    times = Float64.(collect(tspan))
    n_satellites = length(elements)
    positions = Array{Float64,3}(undef, n_satellites, length(times), 3)
    for sat in 1:n_satellites, (time_index, elapsed_s) in pairs(times), axis in 1:3
        satellite_offset = axis == 3 ? (sat - 1) * backend.satellite_spacing_km : 0.0
        positions[sat, time_index, axis] = backend.origin_ecef_km[axis] +
            backend.velocity_ecef_km_s[axis] * elapsed_s + satellite_offset
    end
    result = OrbitResult(positions, Dict{String,Any}(
        "backend" => backend_name(backend),
        "frame" => "ecef",
        "deterministic" => true,
    ))
    return validate_orbit_result(
        result;
        expected_satellites=n_satellites,
        expected_times=length(times),
    )
end

function __init__()
    register_orbit_backend!(:stub, options -> StubOrbitBackend(; options...); replace=true)
    return nothing
end

end
