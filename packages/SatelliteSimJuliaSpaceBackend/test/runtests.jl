using Test
using SatelliteSimBackends
using SatelliteSimJuliaSpaceBackend
using SatelliteSimOrbit

# Reference vector for the version-locked native two-body path.  Units are km,
# frame is ECEF, and the tolerance is 1 m so backend numerical drift is visible.
const TWO_BODY_ECEF_GOLDEN_KM = [
    -3105.134797541  6185.569497983    0.0;
    -3316.764813342  6063.846253906  363.564520598;
    -3515.279883685  5917.798234203  725.556090761;
]
const TWO_BODY_ECEF_ATOL_KM = 1e-3

@testset "SatelliteSimJuliaSpaceBackend" begin
    @test orbit_backend_registered(:julia_space)
    configured = create_orbit_backend(:julia_space; propagator=:j2)
    @test configured isa JuliaSpaceOrbitBackend
    @test configured.propagator == :j2
    capabilities = backend_capabilities(configured)
    @test capabilities.implementation == :independent_secular_elements
    @test capabilities.propagators == (:two_body, :j2, :j4)

    elements = generate_walker_delta(T=4, P=2, F=1, alt_km=550.0, inc_deg=53.0)
    result = propagate_orbit(JuliaSpaceOrbitBackend(), elements, [0.0, 10.0])
    @test size(result.positions_ecef_km) == (4, 2, 3)
    @test result.metadata["propagator"] == "two_body"
    @test result.metadata["implementation"] == "independent_secular_elements"
    @test result.metadata["frame_transform"] == "SatelliteToolbox TEME-to-PEF"

    golden_elements = generate_walker_delta(T=1, P=1, F=0, alt_km=550.0, inc_deg=53.0)
    golden_result = propagate_orbit(
        JuliaSpaceOrbitBackend(), golden_elements, [0.0, 60.0, 120.0],
    )
    @test golden_result.metadata["frame"] == "ecef"
    @test golden_result.positions_ecef_km[1, :, :] ≈ TWO_BODY_ECEF_GOLDEN_KM atol=TWO_BODY_ECEF_ATOL_KM rtol=0

    # The Orbit facade must return exactly the bare array contract used downstream.
    facade_positions = propagate_to_ecef(
        JuliaSpaceOrbitBackend(), golden_elements, [0.0, 60.0, 120.0],
    )
    @test facade_positions == golden_result.positions_ecef_km

    # Cross-implementation checks exercise shape, frame, unit, time ordering,
    # multiple satellites, and all supported propagator selections.  The adapter
    # uses an independent secular-elements implementation, while the explicit 1 m
    # threshold protects the stable backend contract from formula drift.
    comparison_elements = generate_walker_delta(
        T=12, P=3, F=1, alt_km=550.0, inc_deg=71.0,
    )
    comparison_times = [0.0, 17.0, 60.0, 600.0]

    # Focused regression for the J4 formula: an ASCII variable named `e2` must
    # never be juxtaposed with a coefficient (`4e2` is the literal 400.0 in
    # Julia).  These version-locked rates catch that parsing error directly.
    j4_rates = SatelliteSimJuliaSpaceBackend._secular_rates(comparison_elements[1], :j4)
    @test j4_rates[1] ≈ 0.0010943101162149806 rtol=1e-13
    @test j4_rates[2] ≈ -4.899407978729303e-7 rtol=1e-13
    @test j4_rates[3] ≈ -3.5464822850866567e-7 rtol=1e-13

    for propagator in (:two_body, :j2, :j4)
        native = propagate_to_ecef(
            comparison_elements, comparison_times; propagator=propagator,
        )
        adapted = propagate_to_ecef(
            JuliaSpaceOrbitBackend(propagator), comparison_elements, comparison_times,
        )
        @test size(adapted) == size(native) == (12, 4, 3)
        @test maximum(abs.(adapted .- native)) <= TWO_BODY_ECEF_ATOL_KM
    end

    time_view = @view comparison_times[2:4]
    native_view = propagate_to_ecef(comparison_elements, collect(time_view); propagator=:j2)
    adapted_view = propagate_to_ecef(
        JuliaSpaceOrbitBackend(:j2), comparison_elements, time_view,
    )
    @test maximum(abs.(adapted_view .- native_view)) <= TWO_BODY_ECEF_ATOL_KM
end
