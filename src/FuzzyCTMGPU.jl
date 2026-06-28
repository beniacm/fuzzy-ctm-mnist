"""
    FuzzyCTMGPU

Optional **AMD ROCm / AMDGPU.jl** accelerator for the Fuzzy Convolutional Tsetlin Machine —
algorithmically identical (and bit-exact) to the CPU `FuzzyCTM`, ~3–10× faster.

Online TM training is sequential over *samples* but fully parallel over the `NCLA = C·CPC`
*clauses* — so each sample launches one eval kernel (1 work-item/clause scans all patches) and
one feedback kernel (1 work-item per active clause; the y- and q-class clause sets are disjoint
⇒ race-free). Only the per-clause votes cross the bus for the host class-sum + q-draw.

    using AMDGPU                       # requires an AMD GPU + ROCm
    best = FuzzyCTMGPU.gpu_train(Xtr, Ytr, Xte, Yte; cpc=320, lf=16, epochs=40)
"""
module FuzzyCTMGPU

using Printf, AMDGPU

const IL=UInt8(60); const SMAX=UInt8(63); const INIT_S=UInt8(59)
const GOLDEN_MUL=0x9E3779B97F4A7C15
const F_PAD=160; const NTA=2*F_PAD; const WORDS=NTA÷64

@inline xs64(v::UInt64) = (v⊻=(v<<13); v⊻=(v>>7); v⊻=(v<<17); v)
clog2(n::Int) = (n<=1 ? 0 : (b=0; while (1<<b)<n; b+=1; end; b))
@inline function qdraw(r::UInt64, y::Int, C::Int)
    cbits=clog2(C); qslots=cbits==0 ? 1 : (64÷cbits); mask=(UInt64(1)<<cbits)-UInt64(1)
    @inbounds for k in 0:qslots-1
        cand=Int((r>>(cbits*k))&mask); cand<C && cand!=y && return cand
    end
    return y==0 ? 1 : 0
end

# ---- host: bit-pack each sample's NPOS patch literals (5 words) into a flat array ----
function build_posbits(NP1, NPOS, POSF)
    pb = zeros(Bool, POSF, NPOS)
    for p in 0:NPOS-1
        px = p % NP1; py = p ÷ NP1
        for j in 0:NP1-2
            pb[1+j, p+1] = (j < px); pb[1+(NP1-1)+j, p+1] = (j < py)
        end
    end
    return pb
end
function sample_literals!(dst, base, img, pb, NP1, NPOS, PW, PXF, POSF, IMG, ST)
    @inbounds for p in 0:NPOS-1
        ox=(p%NP1)*ST; oy=(p÷NP1)*ST; w1=UInt64(0); w2=UInt64(0); w3=UInt64(0)
        for ry in 0:PW-1
            b=(oy+ry)*IMG+ox
            for rx in 0:PW-1
                if img[b+rx+1]; f=ry*PW+rx; f<64 ? (w1|=UInt64(1)<<f) : (w2|=UInt64(1)<<(f-64)); end
            end
        end
        for j in 0:POSF-1
            if pb[j+1,p+1]
                f=PXF+j
                if f<64; w1|=UInt64(1)<<f elseif f<128; w2|=UInt64(1)<<(f-64) else w3|=UInt64(1)<<(f-128) end
            end
        end
        nf1=~w1; nf2=~w2; nf3=(~w3)&0x00000000FFFFFFFF
        w3|=(nf1<<32); w4=(nf1>>32)|(nf2<<32); w5=(nf2>>32)|(nf3<<32)
        o=base+p*WORDS; dst[o+1]=w1; dst[o+2]=w2; dst[o+3]=w3; dst[o+4]=w4; dst[o+5]=w5
    end
end
function precompute_all(X, n, pb, NP1, NPOS, PW, PXF, POSF, IMG, ST)
    litw = Vector{UInt64}(undef, WORDS*NPOS*n)
    Threads.@threads for i in 1:n
        sample_literals!(litw, (i-1)*WORDS*NPOS, @view(X[:,i]), pb, NP1, NPOS, PW, PXF, POSF, IMG, ST)
    end
    return litw
end

# ---- GPU kernels ----
function eval_kernel(ta, litw, off::Int32, fv, bp, bf, incw, NCLA::Int32, NPOS::Int32, IL_::UInt8, LF::Int32)
    c = (workgroupIdx().x-Int32(1))*workgroupDim().x + workitemIdx().x
    c > NCLA && return
    @inbounds begin
        o=(c-Int32(1))*Int32(NTA); i1=UInt64(0);i2=UInt64(0);i3=UInt64(0);i4=UInt64(0);i5=UInt64(0)
        for b in 0:63
            ta[o+1+b]  >=IL_ && (i1|=UInt64(1)<<b); ta[o+65+b] >=IL_ && (i2|=UInt64(1)<<b)
            ta[o+129+b]>=IL_ && (i3|=UInt64(1)<<b); ta[o+193+b]>=IL_ && (i4|=UInt64(1)<<b)
            ta[o+257+b]>=IL_ && (i5|=UInt64(1)<<b)
        end
        ic=(c-Int32(1))*Int32(WORDS); incw[ic+1]=i1;incw[ic+2]=i2;incw[ic+3]=i3;incw[ic+4]=i4;incw[ic+5]=i5
        bestf=typemax(Int32); bestp=Int32(0)
        for p in Int32(0):(NPOS-Int32(1))
            lo=off+p*Int32(WORDS)
            fl=Int32(count_ones(i1&~litw[lo+1]))+Int32(count_ones(i2&~litw[lo+2]))+
               Int32(count_ones(i3&~litw[lo+3]))+Int32(count_ones(i4&~litw[lo+4]))+
               Int32(count_ones(i5&~litw[lo+5]))
            fl<bestf && (bestf=fl; bestp=p+Int32(1))
        end
        fv[c]=bestf<=LF ? (LF-bestf) : Int32(0); bp[c]=bestp; bf[c]=bestf
    end
    return
end

function feedback_kernel(ta, litw, off::Int32, bp, bf, incw, rng, CPC::Int32, IL_::UInt8, SMAX_::UInt8,
                         LF::Int32, Lmax::Int32, sparam::Int32, yclass::Int32, qclass::Int32,
                         upsel_y::Int32, upsel_q::Int32, mask2t::Int32)
    t = (workgroupIdx().x-Int32(1))*workgroupDim().x + workitemIdx().x
    t > Int32(2)*CPC && return
    @inbounds begin
        half=CPC÷Int32(2)
        if t<=CPC; ci=t-Int32(1); cabs=yclass*CPC+ci; label=true; upsel=upsel_y
        else; ci=t-Int32(1)-CPC; cabs=qclass*CPC+ci; label=false; upsel=upsel_q; end
        yteam = label ? (ci<half) : !(ci<half)
        r0=rng[cabs+1]; sel=Int32((r0&0x7FF)&UInt64(mask2t))<upsel
        bpp=bp[cabs+1]; fl=bf[cabs+1]; fv=fl<=LF ? (LF-fl) : Int32(0)
        ic=cabs*Int32(WORDS)
        incn=Int32(count_ones(incw[ic+1]))+Int32(count_ones(incw[ic+2]))+Int32(count_ones(incw[ic+3]))+
             Int32(count_ones(incw[ic+4]))+Int32(count_ones(incw[ic+5]))
        incok=incn<Lmax; mode = yteam ? (fv>Int32(0) ? Int32(0) : Int32(1)) : Int32(2)
        r1=xs64(r0); ib=UInt64(0)
        if yteam && fv==Int32(0); ib=r1; rng[cabs+1]=xs64(r1) else rng[cabs+1]=r1 end
        if sel && (yteam || fv!=Int32(0))
            o=cabs*Int32(NTA); lb=off+(bpp-Int32(1))*Int32(WORDS)
            if mode==Int32(0)
                for w in Int32(0):Int32(4)
                    lw=litw[lb+w+1]; bse=o+w*Int32(64)
                    for b in 0:63
                        st=ta[bse+1+b]
                        if (lw>>b)&1==1; incok && st<SMAX_ && (ta[bse+1+b]=st+0x01)
                        else st<IL_ && st>0x00 && (ta[bse+1+b]=st-0x01) end
                    end
                end
            elseif mode==Int32(1)
                for k in Int32(0):Int32(4)
                    d=xs64(ib⊻UInt64(k)); bse=o+k*Int32(64)
                    for mm in 0:(sparam-Int32(1))
                        idx=Int((d>>(6*mm))&63); st=ta[bse+1+idx]; st>0x00 && (ta[bse+1+idx]=st-0x01)
                    end
                end
            else
                for w in Int32(0):Int32(4)
                    lw=litw[lb+w+1]; bse=o+w*Int32(64)
                    for b in 0:63
                        if (lw>>b)&1==0; st=ta[bse+1+b]; st<IL_ && (ta[bse+1+b]=st+0x01) end
                    end
                end
            end
        end
    end
    return
end

const GS = 256

"""
    gpu_train(Xtr, Ytr, Xte, Yte; cpc=320, lf=16, L=16, s=3, T=64, epochs=40,
              img=28, pw=10, seed=0x123456789ABCDEF1, verbose=true) -> best_test_acc
"""
function gpu_train(Xtr, Ytr, Xte, Yte; cpc=320, lf=16, L=16, s=3, T=64, epochs=40,
                   img=28, pw=10, stride=1, seed::UInt64=0x123456789ABCDEF1, verbose=true)
    @assert (T&(T-1))==0 && cpc%2==0 && stride>=1
    C=10; NP1=(img-pw)÷stride+1; NPOS=NP1*NP1; PXF=pw*pw; POSF=2*(NP1-1); NCLA=C*cpc
    ntr=size(Xtr,2); nte=size(Xte,2)
    pb = build_posbits(NP1, NPOS, POSF)
    verbose && (print("precompute + upload literals..."); flush(stdout))
    litw_tr = ROCArray(precompute_all(Xtr, ntr, pb, NP1, NPOS, pw, PXF, POSF, img, stride))
    litw_te = ROCArray(precompute_all(Xte, nte, pb, NP1, NPOS, pw, PXF, POSF, img, stride))
    ta  = AMDGPU.zeros(UInt8, NTA*NCLA); fill!(ta, INIT_S)
    rngh = Vector{UInt64}(undef, NCLA); for c in 0:NCLA-1; rngh[c+1]=seed⊻(UInt64(c)*GOLDEN_MUL); end
    rng = ROCArray(rngh)
    fv=AMDGPU.zeros(Int32,NCLA); bp=AMDGPU.zeros(Int32,NCLA); bf=AMDGPU.zeros(Int32,NCLA)
    incw=AMDGPU.zeros(UInt64,WORDS*NCLA); fv_h=Vector{Int32}(undef,NCLA)
    AMDGPU.synchronize(); verbose && println(" done")
    spp = WORDS*NPOS; qrng = Ref(seed); half = cpc÷2

    eval_at(litw,off) = @roc groupsize=GS gridsize=cld(NCLA,GS) eval_kernel(
        ta, litw, Int32(off), fv, bp, bf, incw, Int32(NCLA), Int32(NPOS), IL, Int32(lf))

    function train_sample(off, y)
        eval_at(litw_tr, off); copyto!(fv_h, fv)
        qrng[] = xs64(qrng[]); q = qdraw(qrng[], y, C)
        csy=0; csq=0; by=y*cpc; bq=q*cpc
        @inbounds for ci in 0:cpc-1
            pol = ci<half ? 1 : -1; csy += Int(fv_h[by+ci+1])*pol; csq += Int(fv_h[bq+ci+1])*pol
        end
        csy=clamp(csy,-T,T); csq=clamp(csq,-T,T)
        @roc groupsize=GS gridsize=cld(2*cpc,GS) feedback_kernel(ta, litw_tr, Int32(off), bp, bf, incw, rng,
            Int32(cpc), IL, SMAX, Int32(lf), Int32(L), Int32(s), Int32(y), Int32(q),
            Int32(T-csy), Int32(T+csq), Int32(2*T-1))
    end
    function predict(off)
        eval_at(litw_te, off); copyto!(fv_h, fv)
        bk=0; bcs=typemin(Int)
        @inbounds for k in 0:C-1
            sm=0; b=k*cpc; for ci in 0:cpc-1; sm += Int(fv_h[b+ci+1])*(ci<half ? 1 : -1); end
            sm>bcs && (bcs=sm; bk=k)
        end
        return bk
    end

    train_sample(0, Int(Ytr[1])); predict(0); AMDGPU.synchronize()   # JIT warmup
    fill!(ta, INIT_S); copyto!(rng, rngh); qrng[] = seed
    best=0.0
    for ep in 1:epochs
        t0=time()
        for i in 1:ntr; train_sample((i-1)*spp, Int(Ytr[i])); end
        AMDGPU.synchronize(); dt=time()-t0
        cor=0; for i in 1:nte; predict((i-1)*spp)==Int(Yte[i]) && (cor+=1); end
        acc=cor/nte; best=max(best,acc)
        verbose && @printf("  epoch %2d  test_acc=%.4f  best=%.4f  [%.1fs, %.0f samp/s]\n",
                           ep, acc, best, dt, ntr/dt); flush(stdout)
    end
    return best
end

end # module
