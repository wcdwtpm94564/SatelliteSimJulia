using Test
using SatelliteSimBackends

struct MissingBackend <: AbstractOrbitBackend end
struct ConfiguredBackend <: AbstractOrbitBackend
    scale::Float64
end
struct ConfiguredComputeBackend <: AbstractComputeBackend
    precision::Symbol
end
struct IdentityComputeBackend <: AbstractComputeBackend
    name::String
    revision::Int
    reported_name::String
end
struct ContractCPUDeviceBackend <: AbstractComputeBackend end
mutable struct MutableCapabilityComputeBackend <: AbstractComputeBackend
    operations::Vector{Symbol}
    details::Dict{Symbol,Vector{String}}
end

SatelliteSimBackends.compute_backend_name(backend::IdentityComputeBackend) =
    backend.name
SatelliteSimBackends.compute_backend_capabilities(::IdentityComputeBackend) = (
    operations=(:gsl_series,),
    device=:test,
    input_residency=:host,
    output_residency=:host,
)
SatelliteSimBackends.compute_backend_cache_token(backend::IdentityComputeBackend) =
    (revision=backend.revision,)
SatelliteSimBackends.compute_backend_source_files(::IdentityComputeBackend) =
    [@__FILE__]
SatelliteSimBackends.compute_backend_capabilities(::ContractCPUDeviceBackend) = (
    operations=(:gsl_series,),
    device=:cpu,
    input_residency=:host,
    output_residency=:host,
)
SatelliteSimBackends.compute_backend_capabilities(backend::MutableCapabilityComputeBackend) = (
    operations=backend.operations,
    device=:test,
    input_residency=:host,
    output_residency=:host,
    details=backend.details,
)

function SatelliteSimBackends.evaluate_gsl_series(
    backend::IdentityComputeBackend,
    positions,
    stations;
    gsl_min_elevation_deg,
    gsl_max_range_km,
)
    output_size = (size(positions, 1), length(stations), size(positions, 2))
    return GSLSeriesResult(
        falses(output_size),
        zeros(output_size),
        zeros(output_size),
        zeros(output_size),
        Dict{String,Any}("backend" => backend.reported_name),
    )
end

@testset "SatelliteSimBackends contract" begin
    @test backend_name(MissingBackend()) == "MissingBackend"
    @test backend_capabilities(MissingBackend()).frames == (:ecef,)
    @test_throws MethodError propagate_orbit(MissingBackend(), [1], [0.0])

    valid = OrbitResult(zeros(2, 3, 3), Dict{String,Any}())
    @test validate_orbit_result(valid; expected_satellites=2, expected_times=3) === valid
    @test_throws ArgumentError validate_orbit_result(OrbitResult(zeros(2, 3, 2), Dict{String,Any}()))
end

@testset "Compute backend contract" begin
    cpu = create_compute_backend(:cpu)
    @test cpu isa CPUComputeBackend
    @test compute_backend_name(cpu) == "cpu"
    @test isempty(compute_backend_capabilities(cpu).operations)
    @test compute_backend_fingerprint(cpu).name == "cpu"
    @test compute_backend_cache_token(cpu) === nothing
    @test isempty(compute_backend_source_files(cpu))
    @test compute_backend_registered(:cpu)
    @test :cpu in available_compute_backends()
    @test !unregister_compute_backend!(:cpu)
    @test_throws ArgumentError create_compute_backend(:cpu; precision=:float32)
    @test_throws ArgumentError register_compute_backend!(:cpu, _ -> cpu)

    valid = GSLSeriesResult(
        falses(2, 3, 4),
        zeros(2, 3, 4),
        zeros(2, 3, 4),
        zeros(2, 3, 4),
        Dict{String,Any}("backend" => "cpu"),
    )
    @test validate_gsl_series_result(
        valid;
        expected_satellites=2,
        expected_stations=3,
        expected_times=4,
    ) === valid
    @test_throws ArgumentError validate_gsl_series_result(
        GSLSeriesResult(
            falses(2, 3, 4),
            zeros(2, 3, 3),
            zeros(2, 3, 4),
            zeros(2, 3, 4),
            Dict{String,Any}(),
        ),
    )

    valid_isl = ISLSeriesResult(
        falses(1, 2),
        zeros(1, 2),
        zeros(1, 2),
        trues(1, 2),
        reshape([0.0, 90.0], 1, 2),
        reshape([-1.0, 1.0], 1, 2),
        zeros(1, 2),
        Dict{String,Any}("backend" => "test"),
    )
    @test validate_isl_series_result(
        valid_isl;
        expected_pairs=1,
        expected_times=2,
    ) === valid_isl
    for invalid_elevation in (-0.1, 90.1)
        @test_throws ArgumentError validate_isl_series_result(
            ISLSeriesResult(
                falses(1, 1),
                zeros(1, 1),
                zeros(1, 1),
                trues(1, 1),
                fill(invalid_elevation, 1, 1),
                zeros(1, 1),
                zeros(1, 1),
                Dict{String,Any}(),
            ),
        )
    end
    for invalid_cos_psi in (-1.1, 1.1)
        @test_throws ArgumentError validate_isl_series_result(
            ISLSeriesResult(
                falses(1, 1),
                zeros(1, 1),
                zeros(1, 1),
                trues(1, 1),
                fill(45.0, 1, 1),
                fill(invalid_cos_psi, 1, 1),
                zeros(1, 1),
                Dict{String,Any}(),
            ),
        )
    end
end

@testset "Orbit backend discovery and configuration" begin
    spec = OrbitBackendSpec("contract_test"; scale=2)
    @test spec.name == :contract_test
    @test spec.options == (scale=2,)
    @test OrbitBackendSpec("contract_test", (scale=2,)) == spec
    @test_throws ArgumentError OrbitBackendSpec(Symbol(""))

    unregister_orbit_backend!(:contract_test)
    register_orbit_backend!(
        :contract_test,
        options -> ConfiguredBackend(Float64(get(options, :scale, 1.0))),
    )
    try
        @test orbit_backend_registered("contract_test")
        @test :contract_test in available_orbit_backends()
        backend = create_orbit_backend(spec)
        @test backend isa ConfiguredBackend
        @test backend.scale == 2.0
        @test_throws ArgumentError register_orbit_backend!(:contract_test, _ -> MissingBackend())
    finally
        @test unregister_orbit_backend!(:contract_test)
    end
    @test !orbit_backend_registered(:contract_test)

    error = try
        create_orbit_backend(:not_loaded)
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin("load the optional backend package", sprint(showerror, error))

    register_orbit_backend!(:invalid_factory, _ -> :not_a_backend; replace=true)
    try
        @test_throws ArgumentError create_orbit_backend(:invalid_factory)
    finally
        unregister_orbit_backend!(:invalid_factory)
    end
end

@testset "Compute backend discovery and configuration" begin
    spec = ComputeBackendSpec("contract_compute"; precision="float32")
    @test spec.name == :contract_compute
    @test spec.options == (precision="float32",)
    @test ComputeBackendSpec("contract_compute", (precision="float32",)) == spec
    @test_throws ArgumentError ComputeBackendSpec(Symbol(""))

    unregister_compute_backend!(:contract_compute)
    register_compute_backend!(
        :contract_compute,
        options -> ConfiguredComputeBackend(Symbol(get(options, :precision, "float64"))),
    )
    try
        @test compute_backend_registered("contract_compute")
        @test :contract_compute in available_compute_backends()
        backend = create_compute_backend(spec)
        @test backend isa ConfiguredComputeBackend
        @test backend.precision == :float32
        @test_throws ArgumentError register_compute_backend!(
            :contract_compute,
            _ -> ConfiguredComputeBackend(:float64),
        )
    finally
        @test unregister_compute_backend!(:contract_compute)
    end
    @test !compute_backend_registered(:contract_compute)

    error = try
        create_compute_backend(:not_loaded)
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin("register the device backend", sprint(showerror, error))

    register_compute_backend!(:invalid_compute_factory, _ -> :not_a_backend; replace=true)
    try
        @test_throws ArgumentError create_compute_backend(:invalid_compute_factory)
    finally
        unregister_compute_backend!(:invalid_compute_factory)
    end
end

@testset "Compute backend capability snapshots are immutable" begin
    name = :mutable_capability_snapshot
    unregister_compute_backend!(name)
    backend_ref = Ref{Union{Nothing,MutableCapabilityComputeBackend}}(nothing)
    register_compute_backend!(
        name,
        _ -> begin
            backend = MutableCapabilityComputeBackend(
                [:gsl_series],
                Dict(:driver => ["v1"]),
            )
            backend_ref[] = backend
            backend
        end,
    )
    try
        resolution = resolve_compute_backend(name)
        capabilities = compute_backend_capabilities(resolution)
        @test capabilities.operations == (:gsl_series,)
        @test capabilities.details == (:driver => ("v1",),)

        push!(backend_ref[].operations, :mutated_operation)
        push!(backend_ref[].details[:driver], "v2")
        backend_ref[].details[:runtime] = ["changed"]

        @test compute_backend_capabilities(resolution) == capabilities
        @test compute_backend_provenance(resolution).capabilities == capabilities
    finally
        unregister_compute_backend!(name)
    end
end

@testset "Compute backend resolution identity and lifecycle" begin
    name = :resolution_lifecycle
    unregister_compute_backend!(name)
    factory_calls = Ref(0)
    spec = ComputeBackendSpec(name; precision=:float64)
    positions = zeros(1, 1, 3)
    stations = [(0.0, 0.0, 0.0)]

    register_compute_backend!(
        name,
        _ -> begin
            factory_calls[] += 1
            IdentityComputeBackend("lifecycle-v1", 1, "lifecycle-v1")
        end,
    )
    try
        first_resolution = resolve_compute_backend(spec)
        @test first_resolution isa ResolvedComputeBackend
        @test factory_calls[] == 1
        @test compute_backend_spec(first_resolution) == spec
        first_provenance = compute_backend_provenance(first_resolution)
        @test first_provenance.requested_spec.name == name
        @test first_provenance.implementation.name == "lifecycle-v1"
        @test first_provenance.capabilities.device == :test
        @test first_provenance.call_count == 0
        @test_throws MethodError ResolvedComputeBackend(
            spec,
            IdentityComputeBackend("forged", 0, "forged"),
        )

        @test compute_backend_cache_token(first_resolution) == (revision=1,)
        @test compute_backend_fingerprint(first_resolution).name == "lifecycle-v1"
        @test compute_backend_source_files(first_resolution) == [@__FILE__]
        delegated_fingerprint = compute_backend_fingerprint(first_resolution)
        @test !hasproperty(delegated_fingerprint, :registration_generation)
        @test !hasproperty(delegated_fingerprint, :resolution_id)

        first_result = evaluate_gsl_series(
            first_resolution,
            positions,
            stations;
            gsl_min_elevation_deg=0.0,
            gsl_max_range_km=10_000.0,
        )
        @test first_result.metadata["backend"] == "lifecycle-v1"
        @test compute_backend_provenance(first_resolution).call_count == 1

        register_compute_backend!(
            name,
            _ -> begin
                factory_calls[] += 1
                IdentityComputeBackend("lifecycle-v2", 2, "lifecycle-v2")
            end;
            replace=true,
        )
        second_resolution = resolve_compute_backend(spec)
        second_provenance = compute_backend_provenance(second_resolution)
        @test factory_calls[] == 2
        @test second_provenance.registration_generation >
              first_provenance.registration_generation
        @test second_provenance.resolution_id != first_provenance.resolution_id
        @test second_provenance.implementation.name == "lifecycle-v2"

        old_result = evaluate_gsl_series(
            first_resolution,
            positions,
            stations;
            gsl_min_elevation_deg=0.0,
            gsl_max_range_km=10_000.0,
        )
        @test old_result.metadata["backend"] == "lifecycle-v1"

        @test unregister_compute_backend!(name)
        @test_throws ArgumentError resolve_compute_backend(spec)
        surviving_result = evaluate_gsl_series(
            second_resolution,
            positions,
            stations;
            gsl_min_elevation_deg=0.0,
            gsl_max_range_km=10_000.0,
        )
        @test surviving_result.metadata["backend"] == "lifecycle-v2"

        register_compute_backend!(
            name,
            _ -> begin
                factory_calls[] += 1
                IdentityComputeBackend("lifecycle-v3", 3, "lifecycle-v3")
            end,
        )
        third_resolution = resolve_compute_backend(spec)
        third_provenance = compute_backend_provenance(third_resolution)
        @test factory_calls[] == 3
        @test third_provenance.registration_generation >
              second_provenance.registration_generation
        @test third_provenance.resolution_id != second_provenance.resolution_id

        calls_before_create = factory_calls[]
        created = create_compute_backend(spec)
        @test created isa IdentityComputeBackend
        @test factory_calls[] == calls_before_create + 1
    finally
        unregister_compute_backend!(name)
    end
end

@testset "Backend options schema (draft)" begin
    schema = BackendOptionsSchema(
        backend=:sgp4,
        version=2,
        options=[
            BackendOptionSpec(:scale, Real; required=true),
            BackendOptionSpec(:precision, Symbol; required=false, default=:float64,
                              allowed=(:float32, :float64)),
            BackendOptionSpec(:verbose, Bool; required=false, default=false),
            BackendOptionSpec(:mode, Symbol; required=false),
        ],
    )

    # (a) valid options pass; defaults are filled in schema order; no-default key omitted
    result = validate_backend_options(schema, (scale=1.5,))
    @test result.scale === 1.5
    @test result.precision === :float64
    @test result.verbose === false
    @test !haskey(result, :mode)

    # (b) unknown key rejected
    @test_throws ArgumentError validate_backend_options(schema, (scale=1.0, unknown_key=42))

    # (c) missing required key rejected
    @test_throws ArgumentError validate_backend_options(schema, (precision=:float32,))

    # (d) type mismatch rejected
    @test_throws ArgumentError validate_backend_options(schema, (scale="not_a_number",))

    # (e) allowed-set violation rejected
    @test_throws ArgumentError validate_backend_options(schema, (scale=1.0, precision=:bfloat16))

    # (f) conflict pair rejected
    conflict_schema = BackendOptionsSchema(
        backend=:conflict_backend,
        options=[
            BackendOptionSpec(:alpha, Real; required=false, default=1.0),
            BackendOptionSpec(:beta, Real; required=false, default=1.0),
        ],
        conflicts=[(:alpha, :beta)],
    )
    @test_throws ArgumentError validate_backend_options(conflict_schema, (alpha=0.5, beta=0.5))
    # only one present is fine
    r_conflict = validate_backend_options(conflict_schema, (alpha=0.5,))
    @test r_conflict.alpha === 0.5

    # (g) convenience method via OrbitBackendSpec
    orbit_spec = OrbitBackendSpec(:sgp4; scale=2.0, precision=:float32)
    r2 = validate_backend_options(schema, orbit_spec)
    @test r2.scale === 2.0
    @test r2.precision === :float32
    # name mismatch throws
    wrong_spec = OrbitBackendSpec(:other_backend; scale=1.0)
    @test_throws ArgumentError validate_backend_options(schema, wrong_spec)

    # (h) migrate no-op when from_version == schema.version
    opts = (scale=1.0, precision=:float32)
    @test migrate_backend_options(schema, opts; from_version=2) === opts

    # (i) migrate applies a rename
    old_opts = (scale_factor=3.0,)
    migrated = migrate_backend_options(
        schema, old_opts;
        from_version=1,
        renames=Dict(:scale_factor => :scale),
    )
    @test migrated == (scale=3.0,)

    # (j) migrate backward (from_version > schema.version) throws
    @test_throws ArgumentError migrate_backend_options(schema, opts; from_version=99)
end

@testset "Compute backend resolution rejects false identity" begin
    positions = zeros(1, 1, 3)
    stations = [(0.0, 0.0, 0.0)]

    register_compute_backend!(
        :mismatched_metadata,
        _ -> IdentityComputeBackend("expected-backend", 1, "different-backend");
        replace=true,
    )
    try
        resolution = resolve_compute_backend(:mismatched_metadata)
        @test_throws ArgumentError evaluate_gsl_series(
            resolution,
            positions,
            stations;
            gsl_min_elevation_deg=0.0,
            gsl_max_range_km=10_000.0,
        )
        @test compute_backend_provenance(resolution).call_count == 1
    finally
        unregister_compute_backend!(:mismatched_metadata)
    end

    register_compute_backend!(:cpu_alias, _ -> CPUComputeBackend(); replace=true)
    register_compute_backend!(:cpu_device, _ -> ContractCPUDeviceBackend(); replace=true)
    try
        @test_throws ArgumentError create_compute_backend(:cpu_alias)
        @test_throws ArgumentError resolve_compute_backend(:cpu_alias)
        @test_throws ArgumentError create_compute_backend(:cpu_device)
        @test_throws ArgumentError resolve_compute_backend(:cpu_device)
    finally
        unregister_compute_backend!(:cpu_alias)
        unregister_compute_backend!(:cpu_device)
    end
end
