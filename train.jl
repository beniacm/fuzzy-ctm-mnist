#!/usr/bin/env julia
# train.jl — full MNIST pipeline for the Fuzzy Convolutional Tsetlin Machine.
# Downloads + adaptively binarizes MNIST, trains, and reports test accuracy (~99.2%).
#
#   julia -t auto train.jl                          # CPU (portable), default 99.2% config
#   julia -t auto train.jl --backend gpu            # AMD ROCm acceleration (needs AMDGPU.jl)
#   julia -t auto train.jl --epochs 15 --cpc 160    # quicker, ~99.0%
#
# Flags: --backend cpu|gpu  --cpc 320  --lf 16  --epochs 40  --train-n N  --eval-n N
#        --data DIR (default ./data, auto-downloads)  --no-adaptive (global threshold)  --seed 0xHEX

include(joinpath(@__DIR__, "src", "MNISTData.jl")); using .MNISTData
include(joinpath(@__DIR__, "src", "FuzzyCTM.jl"));  using .FuzzyCTM

function parse_args()
    a = Dict{String,String}(); i = 1
    while i <= length(ARGS)
        if startswith(ARGS[i], "--")
            k = ARGS[i][3:end]
            if i+1 <= length(ARGS) && !startswith(ARGS[i+1], "--"); a[k]=ARGS[i+1]; i+=2
            else a[k]="true"; i+=1 end
        else i += 1 end
    end
    return a
end

function main()
    a = parse_args()
    geti(k,d) = haskey(a,k) ? parse(Int, a[k]) : d
    backend = get(a, "backend", "cpu")
    cpc=geti("cpc",320); lf=geti("lf",16); epochs=geti("epochs",40); stride=geti("stride",1)
    data = get(a, "data", joinpath(@__DIR__, "data"))
    adaptive = !haskey(a, "no-adaptive")
    seed = haskey(a,"seed") ? parse(UInt64, a["seed"]) : 0x123456789ABCDEF1

    println("loading MNIST (adaptive=$adaptive) ...")
    Xtr, Ytr, Xte, Yte = load_mnist(data; adaptive=adaptive)
    tn = min(geti("train-n", size(Xtr,2)), size(Xtr,2))
    en = min(geti("eval-n",  size(Xte,2)), size(Xte,2))
    Xtr, Ytr, Xte, Yte = Xtr[:,1:tn], Ytr[1:tn], Xte[:,1:en], Yte[1:en]
    println("train $tn / test $en | backend=$backend | CPC=$cpc lf=$lf stride=$stride epochs=$epochs | threads=$(Threads.nthreads())")

    if backend == "cpu"
        m = FCTM(; cpc=cpc, lf=lf, stride=stride, seed=seed)
        best = fit!(m, Xtr, Ytr; epochs=epochs, evalX=Xte, evalY=Yte)
    else
        include(joinpath(@__DIR__, "src", "FuzzyCTMGPU.jl"))   # loads AMDGPU only when --backend gpu
        best = Base.invokelatest(Main.FuzzyCTMGPU.gpu_train, Xtr, Ytr, Xte, Yte;
                                 cpc=cpc, lf=lf, stride=stride, epochs=epochs, seed=seed)
    end
    println("="^48)
    println("BEST TEST ACCURACY = $(round(best, digits=4))   (CPC=$cpc, lf=$lf, stride=$stride, $epochs epochs)")
end

main()
