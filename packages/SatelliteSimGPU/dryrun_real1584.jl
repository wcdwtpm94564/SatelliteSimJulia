# Local dry-run: TLE parse + host SGP4→PEF position smoke (no Modal, no CUDA required).
# Usage: julia --project=packages/SatelliteSimGPU packages/SatelliteSimGPU/dryrun_real1584.jl

using Pkg
Pkg.activate(@__DIR__)
try
    Pkg.instantiate()
catch
end

using SatelliteSimGPU

const TLE = get(
    ENV,
    "SATSIM_TLE_PATH",
    joinpath(@__DIR__, "..", "..", "data", "tle", "celestrak", "starlink_gp_latest.tle"),
)

function parse_bstar(line1::AbstractString)
    field = strip(line1[54:61])
    isempty(field) && return 0.0
    sign_char = field[1]
    body = sign_char in ('+', '-') ? field[2:end] : field
    length(body) >= 2 || return 0.0
    mantissa = parse(Float64, body[1:(end - 2)]) * 1e-5
    exponent = parse(Int, body[(end - 1):end])
    value = mantissa * 10.0^exponent
    return sign_char == '-' ? -value : value
end

function load_n(path, n_want)
    lines = readlines(path)
    n0 = Float64[]; e0 = Float64[]; i0 = Float64[]
    raan = Float64[]; argp = Float64[]; M0 = Float64[]; bstar = Float64[]
    index = 1
    while index + 2 <= length(lines) && length(n0) < n_want
        line1 = lines[index + 1]
        line2 = lines[index + 2]
        index += 3
        startswith(line1, "1 ") && startswith(line2, "2 ") || continue
        try
            n_rev_day = parse(Float64, strip(line2[53:63]))
            n_rad_min = n_rev_day * 2π / 1440
            (2π / n_rad_min >= 225) && continue
            push!(n0, n_rad_min)
            push!(e0, parse(Float64, "0." * strip(line2[27:33])))
            push!(i0, deg2rad(parse(Float64, strip(line2[9:16]))))
            push!(raan, deg2rad(parse(Float64, strip(line2[18:25]))))
            push!(argp, deg2rad(parse(Float64, strip(line2[35:42]))))
            push!(M0, deg2rad(parse(Float64, strip(line2[44:51]))))
            push!(bstar, parse_bstar(line1))
        catch
            continue
        end
    end
    return n0, e0, i0, raan, argp, M0, bstar
end

n0, e0, i0, raan, argp, M0, bstar = load_n(TLE, 1584)
println("DRYRUN parsed=$(length(n0)) from $TLE")
length(n0) == 1584 || error("parse failed")

el = sgp4_init_host(n0, e0, i0, raan, argp, M0, bstar)
tspan = collect(0.0:1.0:19.0)
teme = sgp4_propagate_gpu(el, tspan)
# Explicit synthetic UT1 epoch is sufficient for this frame smoke.
epoch_jd_ut1 = 2.4605e6
elapsed_s = 60.0 .* tspan
pef = teme_to_pef_gpu(teme, elapsed_s; epoch_jd_ut1=epoch_jd_ut1)
println("DRYRUN teme=$(size(teme)) pef=$(size(pef)) finite=$(all(isfinite, pef))")
all(isfinite, pef) || error("non-finite PEF")
println("DRYRUN_REAL1584 status=PASS")
