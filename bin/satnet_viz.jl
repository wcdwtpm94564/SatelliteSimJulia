#!/usr/bin/env julia
# satnet_viz.jl — Viz 子进程入口
# 通过 julia --project=envs/viz bin/satnet_viz.jl <subcommand> [args...]
# 由 satnet.jl 中的 viz 子命令调用，避免主链预编译 Makie

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using SatelliteSimViz

function main(args::Vector{String})
    if isempty(args) || args[1] in ("-h", "--help")
        println("""
        SatelliteSimJulia Viz CLI (envs/viz 子进程)

        Usage:
          julia --project=envs/viz bin/satnet_viz.jl snapshot [--output path.png]
          julia --project=envs/viz bin/satnet_viz.jl coverage  <jld2_file> [--output path.png]
          julia --project=envs/viz bin/satnet_viz.jl czml      <jld2_file> [--output path.czml]
        """)
        return
    end

    cmd = args[1]
    rest = args[2:end]

    if cmd == "snapshot"
        output = get_flag(rest, "--output", "snapshot.png")
        @info "Viz snapshot → $output"
        if isdefined(SatelliteSimViz, :quick_snapshot)
            SatelliteSimViz.quick_snapshot(; output)
        else
            @warn "SatelliteSimViz.quick_snapshot not defined"
        end
    elseif cmd == "coverage"
        isempty(rest) && (println("需要 jld2_file 参数"); return)
        jld2_file = rest[1]
        output = get_flag(rest, "--output", "coverage.png")
        @info "Viz coverage $jld2_file → $output"
        if isdefined(SatelliteSimViz, :plot_coverage)
            SatelliteSimViz.plot_coverage(jld2_file; output)
        else
            @warn "SatelliteSimViz.plot_coverage not defined"
        end
    else
        @warn "未知 viz 子命令: $cmd，使用 --help 查看帮助"
        exit(1)
    end
end

function get_flag(args, flag, default)
    idx = findfirst(==(flag), args)
    isnothing(idx) || idx == length(args) ? default : args[idx + 1]
end

main(ARGS)
