# This file implements Adaptive ABC-PT algorithm

export jumpParams!

#= 

Adapted version Lacki and Miasojedow (2016) "State-dependent swap strategies and automatic 
reduction of number of temperatures in adaptive parallel tempering algotirhm", 
Statistical Computing, 16:95 1-964 for use in BGP (2012) ABC-PT algorithm

=#

type ABCPTChain <: AbstractChain
    id::Int             # chain id
    i::Int              # current index
    infos        ::DataFrame   # DataFrameionary of arrays(L,1) with eval, ACC and others
    parameters   ::DataFrame   # DataFrameionary of arrays(L,1), 1 for each parameter
    moments      ::DataFrame   # DataFrameionary of DataArrays(L,1), 1 for each moment
    dist_tol     ::Float64     # Threshold to accept draw generating SimMoments: ie.. do MH step iff ρ(SimMoments,DataMoments) < dist_tol
    reltemp      ::Float64     # reltemp = log(hotter adjacent) 

    params_nms   ::Array{Symbol,1}  # names of parameters (i.e. exclusive of "id" or "iter", etc)
    moments_nms  ::Array{Symbol,1}  # names of moments
    params2s_nms ::Array{Symbol,1}  # DataFrame names of parameters to sample 

    tempering  ::Float64 # tempering in update probability
    shock_sd   ::Float64 # sd of shock to covariance  
    mu         :: Vector{Float64}
    Sigma      :: Matrix{Float64}  # Current estimate of Covariance within chain

    function ABCPTChain(id,MProb,L,temp,shock,dist_tol,reltemp)
        infos      = DataFrame(chain_id = [id for i=1:L], iter=1:L, evals = zeros(Float64,L), accept = zeros(Bool,L), status = zeros(Int,L), exchanged_with=zeros(Int,L),prob=zeros(Float64,L),perc_new_old=zeros(Float64,L),accept_rate=zeros(Float64,L),shock_sd = [shock;zeros(Float64,L-1)],eval_time=zeros(Float64,L),tempering=zeros(Float64,L))
        parameters = hcat(DataFrame(chain_id = [id for i=1:L], iter=1:L), convert(DataFrame,zeros(L,length(ps2s_names(MProb)))))
        moments    = hcat(DataFrame(chain_id = [id for i=1:L], iter=1:L), convert(DataFrame,zeros(L,length(ms_names(MProb)))))
        par_nms    = sort(Symbol[ Symbol(x) for x in ps_names(MProb) ])
        par2s_nms  = Symbol[ Symbol(x) for x in ps2s_names(MProb) ]
        mom_nms    = sort(Symbol[ Symbol(x) for x in ms_names(MProb) ])
        names!(parameters,[:chain_id;:iter; par2s_nms])
        names!(moments   ,[:chain_id;:iter; mom_nms])
        D = length(m.initial_value)
        mu = zeros(Float64, D)
        Sigma  = eye(D)   # Just for initiation
        
        return new(id,0,infos,parameters,moments,dist_tol,reltemp,par_nms,mom_nms,par2s_nms,temp,shock,mu,Sigma)
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
  
    function MAlgoABCPT(m::MProb,opts::Dict{String,Any}=Dict("N"=>3,"shock_sd"=>0.1,"maxiter"=>100,"maxtemp"=> 100,"min_disttol"=>1,"max_disttol"=>10))

        temps     = logspace(0,log10(opts["maxtemp"]),opts["N"])
        shocksd   = opts["shock_sd"]*ones(Float64,opts["N"])  
        disttol   = logspace(log10(opts["min_disttol"]),log10(opts["max_disttol"]),opts["N"])           # In normalised resid_ssq think of disttol as (average num of sd away from moments)^2
        reltemp = vcat(0, log(temps[2:end] - temps[1:end-1]))
        chains = [ABCPTChain(i,m,opts["maxiter"],temps[i],shocksd[i],disttol[i],reltemp[i]) for i=1:opts["N"] ]
        # current param values
        cpar = [ deepcopy(m.initial_value) for i=1:opts["N"] ]
        D = length(m.initial_value)
        for i in eachindex(chains)                                 # Fill-in chain covariances (Σ₀[ch] =[ 0.1²eye(d)/d ]^[1/T] - p16 BGP2012 example)  
            chains.Sigma[i] = diagm((0.1^(2/chains[i].tempering))*ones(D)/D )
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


# ---------------------------  ABC-PT ALGORITHM ----------------------------

# computes new candidate vectors for each chain
# accepts/rejects that vector on each chain, according to some rule
# *) computes N new parameter vectors
# *) applies a criterion to accept/reject any new params
# *) stores the result in chains
function computeNextIteration!( algo::MAlgoABCPT )
    # here is the meat of your algorithm:
    # how to go from p(t) to p(t+1) ?
    # how to go from p(t) to p(t+1) ?

    incrementChainIter!(algo.MChains)

    # check algo index is the same on all chains
    for ic in 1:algo["N"]
        @assert algo.i == algo.MChains[ic].i
    end

    # New Candidates
    # --------------
    if algo.i > 1
        getNewCandidates!(algo)     # Return a vector of parameters dicts in algo.current_param
    end

    # evaluate objective on all chains
    # --------------------------------
    EV = pmap( x -> evaluateObjective(algo.m,x), algo.current_param)

    # Part 1) LOCAL MOVES ABC-MCMC for i={1,...,N}. accept/reject
    # -----------------------------------------------------------
    doAcceptReject!(algo,EV)

    # Part 2) EXCHANGE MOVES 
    # ----------------------
    # starting mixing in period 3
    if algo.i>=2 && algo["N"] > 1 
        exchangeMoves!(algo)
        tempAdapt(algo)
    end

    # Can also prune chains, but not implemented

end

# notice: higher tempering draws candiates further spread out,
# but accepts lower function values with lower probability
function doAcceptReject!(algo::MAlgoABCPT,EV::Array{Eval})
    for ch in eachindex(EV)
        
        # Bind variable to prob and ACC : to be updated by Accept / Reject
        prob = 1.0
        ACC = true

        # Read in previous Eval in chain
        eval_old = getEval(algo.MChains[ch],algo.i-1)

        # Accept/Reject
        if algo.i == 1                                          # Always accept first iteration as algorithm being initiated 
            prob = 1.0
            ACC = true
            algo.MChains[ch].infos[algo.i,:accept_rate] = 0.1
        elseif EV[ch].value > algo.MChains[ch].dist_tol         # If not within tolerance for chain, reject wp 1. If pass here, then criteria met and do MH, Could turn this off to increase likelihood of acceptance.... 
            prob = 0.
            ACC = false
        elseif  EV[ch].value < eval_old.value                   # If obj fun of candidate draw is better old accept wp 1 
            prob = 1.0
            ACC = true
        else                                                    # If obj fun of candidate draw worse then old ... 
            eta = EV[ch].value - eval_old.value                 # Since new > old this is always negative
            eta /= algo.MChains[ch].tempering                   # Scale by Temperature: T⤒, prob = exp(eta) → 1
            prob = exp(eta)                                     # Get prob = exp(x) where x∈[0,1]
            ACC = prob > rand()                                 # Accept prob > draw from Unif[0,1]
        end

        #= 
            Put commands for *if accepted* actions here - ACC::Bool passed...
        =#

        # Record result and some diagnostic details
        appendEval!(algo.MChains[ch],EV[ch],ACC,prob)
        algo.MChains[ch].infos[algo.i,:perc_new_old] = (EV[ch].value - eval_old.value) / abs(eval_old.value)
 
        # Random Walk Adaptations
        rwAdapt!(algo,ACC)
 
    end
end

# Random Walk adaptations: see Lacki and Miasojedow (2016) "State-dependent swap strategies ...."
function rwAdapt!(algo::MAlgoABCPT, ACC::Bool)
        
        step = (algo.MChains[ch].i+1)^(-0.5)  # Declining step size over iterations

        # Get value of parameters in chain after MH 
        Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]

        # Update Covariance matrix (before update mu)
        dev = Xtilde - algo.MChains[ch].mu
        algo.MChains[ch].Sigma -= step * (algo.MChains[ch].Sigma- A_mul_Bt(dev,dev))

        # Update Mean of Parameters 
        if algo.i==1
            algo.MChains[ch].mu = Xtilde
        else
            algo.MChains[ch].mu -= step * (algo.MChains[ch].mu - Xtilde) 
        end

        # Update Sampling Variance (If acceptance above long run target, set net wider by increasing variance, otherwise reduce it)
        algo.MChains[ch].infos[algo.i,:accept_rate]   = 0.9 * algo.MChains[ch].infos[algo.i-1,:accept_rate] + 0.1 * ACC
        algo.MChains[ch].shock_sd                     = algo.MChains[ch].shock_sd + step * (algo.MChains[ch].infos[algo.i,:accept_rate]- 0.234)
        algo.MChains[ch].infos[algo.i,:shock_sd]      = algo.MChains[ch].shock_sd
end

# State-dependent swap strategies (max swapping - more efficient than random pairs)
function exchangeMoves!(algo::MAlgoABCPT)
 
    swapprob = Float64[]
    for ch in 1:algo["N"], ch2 in ch+1:algo["N"]
        v1 = getEval(algo.MChains[ch],algo.MChains[ch].i).value
        v2 = getEval(algo.MChains[ch2],algo.MChains[ch2].i).value 
        push!(swapprob, exp(-abs(v1 - v2))) 
    end
    push!(swapprob, 1-sum(swapprob))    # No swap option added

    swaplist = rand(Categorical(swapprob), algo["N"])
    for i in swaplist
        if i<length(swaplist)
            ch, ex_with = algo.swapdict[i]
            v2 = getEval(algo.MChains[ex_with],algo.MChains[ex_with].i).value 
            if v2 < algo.MChains[ch].dist_tol
                swapRows!(algo,Pair(ch,ex_with),algo.i)
            end
        else 
            nothing
        end
    end

end

function swapRows!(algo::MAlgoABCPT,pair::Pair,i::Int)

    e1 = getEval(algo.MChains[pair.first] ,i)
    e2 = getEval(algo.MChains[pair.second],i)

    # swap
    appendEval!(algo.MChains[pair.first  ],e2)
    appendEval!(algo.MChains[pair.second ],e1)

    # make a note in infos
    algo.MChains[pair.first].infos[i,:exchanged_with] = pair.second
    algo.MChains[pair.second].infos[i,:exchanged_with] = pair.first

end

# Temperature Adaption - note: use objective values after swapping & 
function tempAdapt!(algo::MAlgoABCPT)

    step = (algo.MChains[ch].i+1)^(-0.5)  # Declining step size over iterations
    
    # Get adjustment for temperature of all but coldest chain
    for ch in 2:algo["N"]
        v1 = getEval(algo.MChains[ch-1],algo.MChains[ch].i).value
        b1 = 1/algo.MChains[ch-1].tempering
        
        v2 = getEval(algo.MChains[ch],algo.MChains[ch2].i).value
        b2 = 1/algo.MChains[ch].tempering
        
        xi = min(1, exp((b2-b1)*(v2 - v1)) )
        algo.reltemps[ch] += step * (xi - 0.234)
    end

    # Get new temp spacing: Start at T1, add new spacing on to define new temperatures
    for ch in 2:algo["N"]
        algo.MChains[ch].tempering = algo.MChains[ch-1].tempering + exp(algo.reltemps[ch])
    end    
end


# function getNewCandidates!
function getNewCandidates!(algo::MAlgoABCPT)

    # update chain by chain
    for ch in 1:algo["N"]
        shock = rand(MvNormal(algo.MChains[ch].Sigma)) * exp(algo.MChains[ch].shock_sd)    # Draw shocks scaled to ensure acceptance rate targeted at 0.234 (See Lacki and Meas)
        shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
        jumpParams!(algo,ch,shockd)                              # Add to parameters
    end

end


function jumpParams!(algo::MAlgoABCPT,ch::Int,shock::Dict)
    eval_old = getLastEval(algo.MChains[ch])
    for k in keys(eval_old.params)
        algo.current_param[ch][k] = eval_old.params[k] + shock[k]
    end
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
