"""
    MNISTData

Pure-Julia MNIST loader with the **adaptive-Gaussian booleanization** that the Fuzzy
Convolutional Tsetlin Machine needs to reach ~99.2% (a single global threshold caps ~1pt lower).

Downloads the IDX files on first use (or reads a local `dir`), then binarizes each pixel against
its 11×11 Gaussian-weighted local mean minus a constant — i.e. OpenCV's
`adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=11, C=2)`, reimplemented here.
"""
module MNISTData

using Downloads, Base.Threads
export load_mnist

const MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
const FILES  = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte")

# CodecZlib is only needed to decompress freshly-downloaded .gz files (lazy-loaded so the
# module works with local uncompressed IDX even if CodecZlib isn't installed).
function gunzip(src::Vector{UInt8})
    CodecZlib = Base.require(Base.PkgId(
        Base.UUID("944b1d66-785c-5afd-91f1-9de20f533193"), "CodecZlib"))
    return transcode(CodecZlib.GzipDecompressor, src)
end

function ensure_files(dir::String)
    mkpath(dir)
    for f in FILES
        path = joinpath(dir, f)
        isfile(path) && continue
        gz = path * ".gz"
        @info "downloading $f"
        Downloads.download(MIRROR * f * ".gz", gz)
        write(path, gunzip(read(gz))); rm(gz; force=true)
    end
end

read_labels(path) = read(path)[9:end]                    # skip 8-byte IDX header
function read_images(path)                               # -> (n, raw row-major image bytes)
    raw = read(path)
    n = Int(raw[5])<<24 | Int(raw[6])<<16 | Int(raw[7])<<8 | Int(raw[8])
    return n, raw[17:end]
end

gauss_kernel(L, σ) = (h=(L-1)/2; g=[exp(-((i-h)^2)/(2σ^2)) for i in 0:L-1]; g ./ sum(g))
@inline clampi(x, lo, hi) = x < lo ? lo : (x > hi ? hi : x)

# adaptive Gaussian threshold (11×11, σ=2.0, C=2, edge-replicate) -> Bool[784,n] (column = image, row-major)
function adaptive_binarize(raw::Vector{UInt8}, n::Int)
    L = 11; σ = 2.0; C = 2.0; pad = L ÷ 2; H = 28; W = 28
    g = gauss_kernel(L, σ)
    X = Matrix{Bool}(undef, H*W, n)
    @threads for i in 0:n-1
        off = i*784; tmp = Matrix{Float64}(undef, H, W)
        @inbounds for r in 0:H-1, c in 0:W-1            # horizontal Gaussian pass
            acc = 0.0
            for k in 0:L-1
                cc = clampi(c + (k-pad), 0, W-1)
                acc += g[k+1] * Float64(raw[off + r*W + cc + 1])
            end
            tmp[r+1, c+1] = acc
        end
        @inbounds for r in 0:H-1, c in 0:W-1            # vertical pass + threshold
            acc = 0.0
            for k in 0:L-1
                rr = clampi(r + (k-pad), 0, H-1)
                acc += g[k+1] * tmp[rr+1, c+1]
            end
            v = Float64(raw[off + r*W + c + 1])
            X[r*W + c + 1, i+1] = v > (acc - C)
        end
    end
    return X
end

function threshold_binarize(raw::Vector{UInt8}, n::Int, thr::Int)
    X = Matrix{Bool}(undef, 784, n)
    @inbounds for i in 0:n-1, p in 1:784; X[p, i+1] = raw[i*784 + p] > thr; end
    return X
end

"""
    load_mnist(dir="data"; adaptive=true, thr=75)
        -> (Xtr::Matrix{Bool}, Ytr::Vector{UInt8}, Xte::Matrix{Bool}, Yte::Vector{UInt8})

`X` is `784 × N` (column = image, row-major). `adaptive=true` uses the Gaussian local-threshold
booleanization (recommended; reaches ~99.2%); `adaptive=false` uses a single global `thr`.
"""
function load_mnist(dir::String="data"; adaptive::Bool=true, thr::Int=75)
    ensure_files(dir)
    ntr, rtr = read_images(joinpath(dir, "train-images-idx3-ubyte"))
    nte, rte = read_images(joinpath(dir, "t10k-images-idx3-ubyte"))
    bin = adaptive ? (r,n)->adaptive_binarize(r,n) : (r,n)->threshold_binarize(r,n,thr)
    Xtr = bin(rtr, ntr); Xte = bin(rte, nte)
    Ytr = read_labels(joinpath(dir, "train-labels-idx1-ubyte"))
    Yte = read_labels(joinpath(dir, "t10k-labels-idx1-ubyte"))
    return Xtr, Ytr, Xte, Yte
end

end # module
