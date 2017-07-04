# This file implements Adaptive ABC-PT algorithm

export jumpParams!

typealias ScalarOrVec{T} Union{T,Vector{T}}

#= 

Adapted version Lacki and Miasojedow (2016) "State-dependent swap strategies and automatic 
reduction of number of temperatures in adaptive parallel tempering algotirhm", 
Statistical Computing, 16:95 1-964 

Also see Andrieu and Thomas (2008), "A tutorial on adadptive MCMC", Statistical Computing, 18: 343-373
for more details on more refined implementation of block sampling or local scale adjustments.

=#

type ABCPTChain <: AbstractChain
    id          :: Int             # chain id
    i           :: Int              # current index
    infos        ::DataFrame   # DataFrameionary of arrays(L,1) with eval, ACC and others
    parameters   ::DataFrame   # DataFrameionary of arrays(L,1), 1 for each parameter
    moments      ::DataFrame   # DataFrameionary of DataArrays(L,1), 1 for each moment
    dist_tol     ::Float64     # Threshold to accept draw generating SimMoments: ie.. do MH step iff ρ(SimMoments,DataMoments) < dist_tol
    reltemp      ::Float64     # reltemp = log(temp of hotter adjacent chain - temp chain) 

    params_nms   ::Array{Symbol,1}  # names of parameters (i.e. exclusive of "id" or "iter", etc)
    moments_nms  ::Array{Symbol,1}  # names of moments
    params2s_nms ::Array{Symbol,1}  # DataFrame names of parameters to sample 

    tempering  :: Float64               # tempering in update probability
    shock_wgts :: ScalarOrVec{Float64} 
    shock_sd   :: ScalarOrVec{Float64}  # sd of shock to covariance  
    shock_id   :: Int64
    mu         :: Vector{Float64}
    F          :: Matrix{Float64}  # Current estimate of Cholesky decomp of covariance within chain: Σ = F[:L]*F[:L]' = F[:U]'*F[:U]

    function ABCPTChain(id,MProb,L,temp,shock,dist_tol,reltemp)
        infos      = DataFrame(chain_id = [id for i=1:L], 
                                iter=1:L, 
                                evals = zeros(Float64,L), 
                                accept = zeros(Bool,L), 
                                status = zeros(Int,L), 
                                init_id=zeros(Int,L),
                                prob=zeros(Float64,L),
                                perc_new_old=zeros(Float64,L),
                                accept_rate=zeros(Float64,L),
                                shock_sd = [mean(shock);zeros(Float64,L-1)],
                                eval_time=zeros(Float64,L),
                                tempering=zeros(Float64,L))
        parameters = hcat(DataFrame(chain_id = [id for i=1:L], iter=1:L), convert(DataFrame,zeros(L,length(ps2s_names(MProb)))))
        moments    = hcat(DataFrame(chain_id = [id for i=1:L], iter=1:L), convert(DataFrame,zeros(L,length(ms_names(MProb)))))
        par_nms    = sort(Symbol[ Symbol(x) for x in ps_names(MProb) ])
        par2s_nms  = Symbol[ Symbol(x) for x in ps2s_names(MProb) ]
        mom_nms    = sort(Symbol[ Symbol(x) for x in ms_names(MProb) ])
        names!(parameters,[:chain_id;:iter; par2s_nms])
        names!(moments   ,[:chain_id;:iter; mom_nms])
        D = length(MProb.initial_value)
        shock_wgts = length(shock)==1 ? 1.0 : ones(D)./D
        mu = zeros(Float64, D)
        F  = eye(D)   # Just for initiation
        
        return new(id,0,infos,parameters,moments,dist_tol,reltemp,par_nms,mom_nms,par2s_nms,temp,shock_wgts,shock,1,mu,F)
    end
end

type ABCPTChains
    MChains :: Array{ABCPTChain,1}
end

type MAlgoABCPT <: MAlgo
    m               :: MProb # an MProb
    opts            :: Dict # list of options
    i               :: Int  # iteration
    current_param   :: Array{Dict,1}  # current param value: one Dict for each chain
    MChains         :: Array{ABCPTChain,1}    # collection of Chains: if N==1, length(chains) = 1
    swapdict        :: Dict{Int64, Vector{Int64}}
  
    function MAlgoABCPT(m::MProb,opts::Dict{String,Any}=Dict("N"=>3,
                                                            "shock_sd"=>0.1,
                                                            "maxiter"=>100,
                                                            "maxstarttemp"=> 100,
                                                            "min_disttol"=>1,
                                                            "max_disttol"=>10))

        temps     = logspace(0,log10(opts["maxstarttemp"]),opts["N"])
        shocksd   = [opts["shock_sd"] for i in 1:opts["N"]]  
        disttol   = logspace(log10(opts["min_disttol"]),log10(opts["max_disttol"]),opts["N"])           # In normalised resid_ssq think of disttol as (average num of sd away from moments)^2
        reltemps = [0.]
        append!(reltemps, log(temps[2:end] - temps[1:end-1]))
        chains = [ABCPTChain(i,m,opts["maxiter"],temps[i],shocksd[i],disttol[i],reltemps[i]) for i=1:opts["N"] ]
        # current param values
        cpar = [ deepcopy(m.initial_value) for i=1:opts["N"] ]
        D = length(m.initial_value)
        for i in eachindex(chains)                                 # Fill-in chain covariances (Σ₀[ch] =[ 2.38²eye(D)/D ]^[1/T] - p16 BGP2012 example)  
            chains[i].F = eye(D)
        end
        swapdict = Dict{Int64, Vector{Int64}}()
        n = 0
        for ch in 1:opts["N"], ch2 in ch+1:opts["N"]
            n+=1
            swapdict[n] = [ch,ch2]
        end
        return new(m,opts,0,cpar,chains,swapdict)
    end
end

# this appends ACCEPTED values only.
function appendEval!(chain::ABCPTChain, ev:: Eval, ACC::Bool, prob::Float64)
    chain.infos[chain.i,:evals]  = ev.value
    chain.infos[chain.i,:prob]   = prob
    chain.infos[chain.i,:accept] = ACC
    chain.infos[chain.i,:status] = ev.status
    chain.infos[chain.i,:eval_time] = ev.time
    chain.infos[chain.i,:tempering] = chain.tempering
    for k in names(chain.moments)
        if !(k in [:chain_id,:iter])
            chain.moments[chain.i,k] = ev.simMoments[k]
        end
    end
    for k in names(chain.parameters)
        if !(k in [:chain_id,:iter])
            chain.parameters[chain.i,k] = ev.params[k]
        end
    end
    return nothing
end

# changing only the eval fields. used in swapRows!
function appendEval!(chain::ABCPTChain, ev:: Eval)
    chain.infos[chain.i,:evals]  = ev.value
    chain.infos[chain.i,:status] = ev.status
    chain.infos[chain.i,:eval_time] = ev.time
    for k in names(chain.moments)
        if !(k in [:chain_id,:iter])
            chain.moments[chain.i,k] = ev.simMoments[k]
        end
    end
    for k in names(chain.parameters)
        if !(k in [:chain_id,:iter])
            chain.parameters[chain.i,k] = ev.params[k]
        end
    end
    return nothing
end


# save algo chains component-wise to HDF5 file
function save(algo::MAlgoABCPT, filename::AbstractString)
    # step 1, create the file if it does not exist

    ff5 = h5open(filename, "w")

    vals = String[]
    keys = String[]
    for (k,v) in algo.opts
        if typeof(v) <: Number
            push!(vals,"$v")
        else
            push!(vals,v)
        end
        push!(keys,k)
    end
    write(ff5,"algo/opts/keys",keys)
    write(ff5,"algo/opts/vals",vals)

    # saving the chains
    for cc in 1:algo["N"]
        saveChainToHDF5(algo.MChains[cc], ff5, "chain/$cc")
    end

    close(ff5)
end

function readAlgoABCPT(filename::AbstractString)

    ff5 = h5open(filename, "r")
    keys = HDF5.read(ff5,"algo/opts/keys")
    vals = HDF5.read(ff5,"algo/opts/vals")
    opts = Dict()
    for k in 1:length(keys)
        opts[keys[k]] = vals[k]
    end

    # each chain has 3 data.frames: parameters, moments and infos
    n = parse(Int,opts["N"])
    params = simpleDataFrameRead(ff5,joinpath("chain","1","parameters"))
    moments = simpleDataFrameRead(ff5,joinpath("chain","1","moments"))
    infos = simpleDataFrameRead(ff5,joinpath("chain","1","infos"))
    if n>1
        for ich in 2:n
            params = vcat(params, simpleDataFrameRead(ff5,joinpath("chain","$ich","parameters")))
            moments = vcat(moments, simpleDataFrameRead(ff5,joinpath("chain","$ich","moments")))
            infos = vcat(infos, simpleDataFrameRead(ff5,joinpath("chain","$ich","infos")))
        end
    end
    close(ff5)
    return Dict("opts" => opts, "params"=> params, "moments"=>moments,"infos"=>infos)
end


function show(io::IO,MA::MAlgoABCPT)
    print(io,"\n")
    print(io,"BGP: ABC-PT Algorithm with $(MA["N"]) chains\n")
    print(io,"============================\n")
    print(io,"\n")
    print(io,"Algorithm\n")
    print(io,"---------\n")
    print(io,"Current iteration: $(MA.i)\n")
    print(io,"Number of params to estimate: $(length(MA.m.params_to_sample))\n")
    print(io,"Number of moments to match: $(length(MA.m.moments))\n")
    print(io,"\n")
    print(io,"Chains\n")
    print(io,"------\n")
    print(io,"Tempering range: [$(MA.MChains[1].tempering),$(MA.MChains[end].tempering)]\n")
end
