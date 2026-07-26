#!/usr/bin/env julia
# satnet.jl — SatelliteSimJulia CLI 入口点
# 使用 using SatelliteSimJulia 加载整个平台，然后调用 CLI。

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using SatelliteSimJulia

const VIZ_SCRIPT = joinpath(@__DIR__, "satnet_viz.jl")
const ENVS_VIZ   = joinpath(@__DIR__, "..", "envs", "viz")

function dispatch_viz(args::Vector{String})
    julia = Base.julia_cmd()
    cmd = `$julia --project=$ENVS_VIZ $VIZ_SCRIPT $args`
    run(cmd)
end

function satnet(args::Vector{String})
    if isempty(args) || args[1] in ("-h", "--help")
        println("""
        SatelliteSimJulia CLI

        Usage:
          julia bin/satnet.jl study list
          julia bin/satnet.jl study run <name> [--export csv]
          julia bin/satnet.jl experiment list
          julia bin/satnet.jl constellation list
          julia bin/satnet.jl sweep run <template>
          julia bin/satnet.jl viz snapshot [--output path.png]
          julia bin/satnet.jl viz coverage <jld2> [--output path.png]
        """)
        return
    end

    # viz 子命令走独立子进程（envs/viz），避免主链预编译 Makie
    if args[1] == "viz"
        dispatch_viz(args[2:end])
        return
    end

    if isdefined(SatelliteSimJulia, :satnet)
        SatelliteSimJulia.satnet(args)
    else
        println("CLI not available. Use the Julia REPL:")
        println("  julia --project=. -e 'using SatelliteSimJulia'")
    end
end

if basename(PROGRAM_FILE) == "satnet.jl"
    satnet(ARGS)
end
