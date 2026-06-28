"""
    FuzzyCTM

A multi-threaded **Fuzzy Convolutional Tsetlin Machine** for image classification.

The Tsetlin Machine learns interpretable conjunctive clauses over boolean features.
The *convolutional* variant shares each clause across all sliding patches of the image
(weight sharing → translation-tolerant, M20K/cache-cheap), and the *fuzzy* variant lets a
clause cast a graded vote `max(0, LF - failed)` instead of a hard fire/not-fire — both act
as regularizers, which is why this model generalizes well (small train/test gap).

Public API:
    m = FCTM(; cpc=320, lf=16)         # 10-class, 10x10 patch on 28x28 by default
    fit!(m, Xtr, Ytr; epochs=40, evalX=Xte, evalY=Yte)
    acc = accuracy(m, Xte, Yte)

`X` is a `Matrix{Bool}` of size `784 × N` (column i = image i, row-major), `Y` a vector of
labels in `0:9`. Start Julia with threads (`julia -t auto`) for the clause-parallel speedup.
"""
module FuzzyCTM

using Base.Threads, Printf
export FCTM, fit!, accuracy, predict

# ----- fixed boolean-feature geometry: 100 patch pixels + 36 position bits, padded to 160;
#       literals = {feat, ~feat} = 320 bits = 5 × UInt64 words -----
const IL         = UInt8(60)            # include threshold (TA state ≥ IL ⇒ literal included)
const SMAX       = UInt8(63)            # max TA state
const INIT_S     = UInt8(59)            # initial TA state (just below include)
const GOLDEN_MUL = 0x9E3779B97F4A7C15
const F_PAD      = 160
const NTA        = 2 * F_PAD            # 320 literals / clause
const WORDS      = NTA ÷ 64             # 5 UInt64 words

@inline function xs64(v::UInt64)        # xorshift64 PRNG step
    v ⊻= (v << 13); v ⊻= (v >> 7); v ⊻= (v << 17); return v
end

clog2(n::Int) = (n <= 1 ? 0 : (b = 0; while (1 << b) < n; b += 1; end; b))

# draw a random "negative" class q ≠ y from the PRNG word (rejection over packed slots)
@inline function qdraw(r::UInt64, y::Int, C::Int)
    cbits  = clog2(C); qslots = cbits == 0 ? 1 : (64 ÷ cbits)
    mask   = (UInt64(1) << cbits) - UInt64(1)
    @inbounds for k in 0:qslots-1
        cand = Int((r >> (cbits*k)) & mask)
        cand < C && cand != y && return cand
    end
    return y == 0 ? 1 : 0
end

mutable struct FCTM
    PW::Int; IMG::Int; C::Int; CPC::Int
    NP1::Int; NPOS::Int; PXF::Int; POSF::Int; F::Int; NCLA::Int
    LF::Int; L::Int; s::Int; T::Int
    seed::UInt64
    ta::Matrix{UInt8}          # NTA × NCLA  (column c = clause c's 320 TA states)
    rng::Vector{UInt64}        # one xorshift stream per clause (race-free parallel feedback)
    qrng::UInt64
    posbits::Matrix{Bool}      # POSF × NPOS  (x/y thermometer position features)
end

"""
    FCTM(; img=28, pw=10, n_classes=10, cpc=320, lf=16, L=16, s=3, T=64, seed=…, use_pos=true)

`cpc` = clauses per class (half vote +, half vote −). `lf` = fuzziness (graded-vote width);
`L` = include cap, `s` = Type-Ib decrement count, `T` = vote target (power of two).
Defaults reproduce the ~99.2% MNIST configuration.
"""
function FCTM(; img=28, pw=10, n_classes=10, cpc=320,
               lf=16, L=16, s=3, T=64, seed=0x123456789ABCDEF1, use_pos=true)
    @assert (T & (T-1)) == 0 "T must be a power of two"
    @assert cpc % 2 == 0     "cpc must be even (half +, half − polarity)"
    NP1  = img - pw + 1; NPOS = NP1*NP1
    PXF  = pw*pw; POSF = 2*(NP1-1); F = PXF + POSF
    @assert F <= F_PAD "feature width $F exceeds envelope $F_PAD (reduce patch/position)"
    NCLA = n_classes * cpc
    posbits = zeros(Bool, POSF, NPOS)
    if use_pos
        for p in 0:NPOS-1
            px = p % NP1; py = p ÷ NP1
            for j in 0:NP1-2
                posbits[1 + j, p+1]            = (j < px)   # x thermometer
                posbits[1 + (NP1-1) + j, p+1]  = (j < py)   # y thermometer
            end
        end
    end
    m = FCTM(pw, img, n_classes, cpc, NP1, NPOS, PXF, POSF, F, NCLA, lf, L, s, T,
             UInt64(seed), fill(INIT_S, NTA, NCLA), Vector{UInt64}(undef, NCLA),
             UInt64(seed), posbits)
    reset!(m); return m
end

function reset!(m::FCTM)
    fill!(m.ta, INIT_S)
    @inbounds for c in 0:m.NCLA-1; m.rng[c+1] = m.seed ⊻ (UInt64(c) * GOLDEN_MUL); end
    m.qrng = m.seed; return m
end

# bit-pack each patch's 320 literal bits ({feat, ~feat}) into 5 UInt64 words.
function patch_literals!(litw::Matrix{UInt64}, m::FCTM, img::AbstractVector{Bool})
    NP1=m.NP1; PW=m.PW; PXF=m.PXF; POSF=m.POSF; IMG=m.IMG
    @inbounds for p in 0:m.NPOS-1
        ox = p % NP1; oy = p ÷ NP1
        w1=UInt64(0); w2=UInt64(0); w3=UInt64(0)
        for ry in 0:PW-1
            base = (oy+ry)*IMG + ox
            for rx in 0:PW-1
                if img[base + rx + 1]
                    f = ry*PW + rx
                    f < 64 ? (w1 |= (UInt64(1) << f)) : (w2 |= (UInt64(1) << (f-64)))
                end
            end
        end
        for j in 0:POSF-1
            if m.posbits[j+1, p+1]
                f = PXF + j
                if     f < 64;  w1 |= (UInt64(1) << f)
                elseif f < 128; w2 |= (UInt64(1) << (f-64))
                else;           w3 |= (UInt64(1) << (f-128)); end
            end
        end
        nf1 = ~w1; nf2 = ~w2; nf3 = (~w3) & 0x00000000FFFFFFFF
        w3 |= (nf1 << 32)
        w4  = (nf1 >> 32) | (nf2 << 32)
        w5  = (nf2 >> 32) | (nf3 << 32)
        litw[1,p+1]=w1; litw[2,p+1]=w2; litw[3,p+1]=w3; litw[4,p+1]=w4; litw[5,p+1]=w5
    end
    return litw
end

@inline function build_include!(incw::Matrix{UInt64}, ta::Matrix{UInt8}, c::Int)
    @inbounds begin
        o = c*NTA
        for w in 0:WORDS-1
            acc = UInt64(0); b0 = o + w*64
            for b in 0:63; ta[b0+1+b] >= IL && (acc |= (UInt64(1) << b)); end
            incw[w+1, c+1] = acc
        end
    end
end

# graded vote of clause c = LF − (failed literals at its best-matching patch)
@inline function eval_clause!(m::FCTM, c::Int, litw, incw, fv_buf, bp_buf, bf_buf)
    @inbounds begin
        col=c+1
        i1=incw[1,col]; i2=incw[2,col]; i3=incw[3,col]; i4=incw[4,col]; i5=incw[5,col]
        best_failed = typemax(Int); best_p = 0
        for p in 1:m.NPOS
            fl = count_ones(i1 & ~litw[1,p]) + count_ones(i2 & ~litw[2,p]) +
                 count_ones(i3 & ~litw[3,p]) + count_ones(i4 & ~litw[4,p]) +
                 count_ones(i5 & ~litw[5,p])
            fl < best_failed && (best_failed = fl; best_p = p)
        end
        fv_buf[col] = best_failed <= m.LF ? (m.LF - best_failed) : 0
        bp_buf[col] = best_p; bf_buf[col] = best_failed
    end
end

# Type I / Ib / II feedback applied to the best patch's literals of clause c
@inline function apply!(m::FCTM, c::Int, litw, bp::Int, mode::Int, incok::Bool, ib_seed::UInt64)
    @inbounds begin
        ta = m.ta; o = c*NTA
        if mode == 0                       # Type I: reinforce matching, weaken absent
            for w in 0:WORDS-1
                lw = litw[w+1, bp]; base = o + w*64
                for b in 0:63
                    st = ta[base+1+b]
                    if (lw >> b) & 1 == 1
                        incok && st < SMAX && (ta[base+1+b] = st + 0x01)
                    else
                        st < IL && st > 0x00 && (ta[base+1+b] = st - 0x01)
                    end
                end
            end
        elseif mode == 1                   # Type Ib: random forgetting (per word, s draws)
            for k in 0:WORDS-1
                d = xs64(ib_seed ⊻ UInt64(k)); base = o + k*64
                for mm in 0:m.s-1
                    idx = Int((d >> (6*mm)) & 63); st = ta[base+1+idx]
                    st > 0x00 && (ta[base+1+idx] = st - 0x01)
                end
            end
        else                               # Type II: include absent literals (discriminate)
            for w in 0:WORDS-1
                lw = litw[w+1, bp]; base = o + w*64
                for b in 0:63
                    if (lw >> b) & 1 == 0
                        st = ta[base+1+b]; st < IL && (ta[base+1+b] = st + 0x01)
                    end
                end
            end
        end
    end
end

@inline function feedback_one!(m::FCTM, cabs::Int, ci::Int, label::Bool,
                               upsel::Int, mask2t::Int, litw, incw, bp_buf, bf_buf)
    @inbounds begin
        half  = m.CPC ÷ 2
        yteam = label ? (ci < half) : !(ci < half)
        r0    = m.rng[cabs+1]
        sel   = Int((r0 & 0x7FF) & mask2t) < upsel
        fl    = bf_buf[cabs+1]; fv = fl <= m.LF ? (m.LF - fl) : 0
        col   = cabs+1
        incn  = count_ones(incw[1,col])+count_ones(incw[2,col])+count_ones(incw[3,col])+
                count_ones(incw[4,col])+count_ones(incw[5,col])
        incok = incn < m.L
        mode  = yteam ? (fv > 0 ? 0 : 1) : 2
        r1    = xs64(r0); ib_seed = UInt64(0)
        if yteam && fv == 0; ib_seed = r1; m.rng[cabs+1] = xs64(r1)
        else; m.rng[cabs+1] = r1; end
        sel && (yteam || fv != 0) && apply!(m, cabs, litw, bp_buf[cabs+1], mode, incok, ib_seed)
    end
end

# one online training sample (eval all clauses, then Type I/II feedback for y and a random q≠y)
function train_sample!(m::FCTM, litw, incw, y::Int, fv_buf, bp_buf, bf_buf)
    NCLA=m.NCLA; CPC=m.CPC; C=m.C; T=m.T; half = CPC ÷ 2
    @threads :static for c in 0:NCLA-1
        build_include!(incw, m.ta, c); eval_clause!(m, c, litw, incw, fv_buf, bp_buf, bf_buf)
    end
    m.qrng = xs64(m.qrng); q = qdraw(m.qrng, y, C)
    cs_y = 0; cs_q = 0
    @inbounds begin
        by = y*CPC; bq = q*CPC
        for ci in 0:CPC-1
            pol = ci < half ? 1 : -1
            cs_y += fv_buf[by+ci+1]*pol; cs_q += fv_buf[bq+ci+1]*pol
        end
    end
    cs_y = clamp(cs_y, -T, T); cs_q = clamp(cs_q, -T, T)
    upsel_y = T - cs_y; upsel_q = T + cs_q; mask2t = 2*T - 1
    by = y*CPC; bq = q*CPC
    @threads :static for t in 0:2*CPC-1     # y-clauses and q-clauses are disjoint ⇒ race-free
        if t < CPC
            feedback_one!(m, by+t, t, true,  upsel_y, mask2t, litw, incw, bp_buf, bf_buf)
        else
            ci = t - CPC
            feedback_one!(m, bq+ci, ci, false, upsel_q, mask2t, litw, incw, bp_buf, bf_buf)
        end
    end
end

function predict_clauses!(m::FCTM, litw, incw, fv_buf, bp_buf, bf_buf)
    @threads for c in 0:m.NCLA-1
        build_include!(incw, m.ta, c); eval_clause!(m, c, litw, incw, fv_buf, bp_buf, bf_buf)
    end
    half = m.CPC ÷ 2; best_k = 0; best_cs = typemin(Int)
    @inbounds for k in 0:m.C-1
        s = 0; b = k*m.CPC
        for ci in 0:m.CPC-1; s += fv_buf[b+ci+1]*(ci < half ? 1 : -1); end
        s > best_cs && (best_cs = s; best_k = k)
    end
    return best_k
end

_bufs(m) = (Matrix{UInt64}(undef, WORDS, m.NPOS), Matrix{UInt64}(undef, WORDS, m.NCLA),
            Vector{Int}(undef, m.NCLA), Vector{Int}(undef, m.NCLA), Vector{Int}(undef, m.NCLA))

"""
    predict(m, X) -> Vector{Int}

Predicted class (0-based) for each column of `X` (`784 × N` Bool).
"""
function predict(m::FCTM, X::AbstractMatrix{Bool})
    litw, incw, fv, bp, bf = _bufs(m)
    out = Vector{Int}(undef, size(X,2))
    for i in 1:size(X,2)
        patch_literals!(litw, m, @view X[:,i])
        out[i] = predict_clauses!(m, litw, incw, fv, bp, bf)
    end
    return out
end

"""
    accuracy(m, X, Y) -> Float64

Test/Top-1 accuracy of `m` on `X` (`784 × N` Bool) with labels `Y` (in 0:9).
"""
function accuracy(m::FCTM, X::AbstractMatrix{Bool}, Y)
    litw, incw, fv, bp, bf = _bufs(m)
    cor = 0
    for i in 1:size(X,2)
        patch_literals!(litw, m, @view X[:,i])
        predict_clauses!(m, litw, incw, fv, bp, bf) == Int(Y[i]) && (cor += 1)
    end
    return cor / size(X,2)
end

"""
    fit!(m, X, Y; epochs=40, evalX=nothing, evalY=nothing, verbose=true) -> best_acc

Train `m` online for `epochs` passes over `X` (`784 × N` Bool) / `Y`. If `evalX`/`evalY`
are given, reports test accuracy after each epoch and returns the best seen.
"""
function fit!(m::FCTM, X::AbstractMatrix{Bool}, Y; epochs::Int=40,
              evalX=nothing, evalY=nothing, verbose::Bool=true)
    litw, incw, fv, bp, bf = _bufs(m)
    N = size(X,2); best = 0.0
    for ep in 1:epochs
        t0 = time()
        for i in 1:N
            patch_literals!(litw, m, @view X[:,i])
            train_sample!(m, litw, incw, Int(Y[i]), fv, bp, bf)
        end
        dt = time() - t0
        if evalX !== nothing
            acc = accuracy(m, evalX, evalY); best = max(best, acc)
            verbose && @printf("  epoch %2d  test_acc=%.4f  best=%.4f  [%.1fs, %.0f samp/s]\n",
                               ep, acc, best, dt, N/dt)
        else
            verbose && @printf("  epoch %2d  [%.1fs, %.0f samp/s]\n", ep, dt, N/dt)
        end
        flush(stdout)
    end
    return best
end

end # module
