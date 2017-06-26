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
        infos      = DataFrame(chain_id = [id for i=1:L], iter=1:L, evals = zeros(Float64,L), accept = zeros(Bool,L), status = zeros(Int,L), init_id=zeros(Int,L),prob=zeros(Float64,L),perc_new_old=zeros(Float64,L),accept_rate=zeros(Float64,L),shock_sd = [mean(shock);zeros(Float64,L-1)],eval_time=zeros(Float64,L),tempering=zeros(Float64,L))
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
  
    function MAlgoABCPT(m::MProb,opts::Dict{String,Any}=Dict("N"=>3,"shock_sd"=>0.1,"maxiter"=>100,"maxtemp"=> 100,"min_disttol"=>1,"max_disttol"=>10))

        temps     = logspace(0,log10(opts["maxtemp"]),opts["N"])
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


# ---------------------------  ABC-PT ALGORITHM ----------------------------

# computes new candidate vectors for each chain
# accepts/rejects that vector on each chain, according to some rule
# *) computes N new parameter vectors
# *) applies a criterion to accept/reject any new params
# *) stores the result in chains
function computeNextIteration!( algo::MAlgoABCPT, mcm::Dict )
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
        if algo["mc_update"]==:GLOBAL
            MOpt.getNewCandidates!(algo)               # Return a vector of parameters dicts in algo.current_param
        elseif algo["mc_update"]==:COMPWISE
            ρbar = MOpt.getNewCandidatesCompWise!(algo)       # Return a vector of parameters dicts in algo.current_param
        elseif algo["mc_update"]==:PPCA
            ρbar = MOpt.getNewCandidatesPPCA!(algo,algo["ppca_method"])       # Return a vector of parameters dicts in algo.current_param
        end
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
    if algo.i>=3 && algo["N"] > 1 
        exchangeMoves!(algo)
        tempAdapt!(algo)
    end

    # Can also prune chains, but not implemented

end

# Density for MCMC
make_π(V::Float64,T::Float64) = exp(-V/T)

# notice: higher tempering draws candiates further spread out,
# but accepts lower function values with lower probability
function doAcceptReject!(algo::MAlgoABCPT,EV::Array{Eval})
    for ch in eachindex(EV)
        
        # Bind variable to prob and ACC : to be updated by Accept / Reject
        prob = 1.0
        ACC = true
        # Accept/Reject
        if algo.i == 1                                          # Always accept first iteration as algorithm being initiated 
            prob = 1.0
            ACC = true
            appendEval!(algo.MChains[ch],EV[ch],ACC,prob)
            algo.MChains[ch].infos[algo.i,:init_id] = ch
        else
            
            eval_old = getEval(algo.MChains[ch],algo.i-1)       # Read in previous Eval in chain
            ΔV = EV[ch].value - eval_old.value
            algo.MChains[ch].infos[algo.i,:perc_new_old] = ΔV / abs(eval_old.value)
            prob = min(1.0,make_π(ΔV,algo.MChains[ch].tempering))         
            if EV[ch].value > algo.MChains[ch].dist_tol         # If not within tolerance for chain, reject wp 1. If pass here, then criteria met and do MH, Could turn this off to increase likelihood of acceptance.... 
                prob = 0.
                ACC = false
            elseif  prob==1.0                                    # If obj fun of candidate draw is better old accept wp 1 
                ACC = true
            else                                                # If obj fun of candidate draw worse then old ... 
                ACC = prob > rand()                              # Accept prob > draw from Unif[0,1]
            end

            # Insert Chain ID as it was before
            algo.MChains[ch].infos[algo.i,:init_id] = algo.MChains[ch].infos[algo.i-1,:init_id]
        end

        # append last accepted value
        if ACC
            appendEval!(algo.MChains[ch],EV[ch],ACC,prob)
        else
            appendEval!(algo.MChains[ch],eval_old,ACC,prob)
        end

        # Random Walk Adaptations
        if algo["mc_update"]==:GLOBAL
            rwAdapt!(algo,ch)        
            #MOpt.rwAdapt!(algo,prob,ch)
            #algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i
            #algo.MChains[ch].infos[algo.i,:shock_sd] = algo.MChains[ch].shock_sd            
        else 
            rwAdaptLocal!(algo,ch)
            #MOpt.rwAdaptLocal!(algo,prob,ch)
            #algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i
            #algo.MChains[ch].infos[algo.i,:shock_sd] = dot(algo.MChains[ch].shock_sd,algo.MChains[ch].shock_wgts)
        end

    end
end


# Rank-one-update with stochastic approximation : See p358 of Andrieu and Thoms (2008) 'A tutorial on adaptive MCMC', Journal of Statistical Computing
rank1update(F::Matrix{Float64}, x::Vector{Float64}) = tril(inv(tril(F))*A_mul_Bt(x,x)*inv(tril(F)') - eye(length(x)))


# Random Walk adaptations: see Lacki and Miasojedow (2016) "State-dependent swap strategies ...."
function rwAdapt!(algo::MAlgoABCPT, ch::Int64)
    
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
    dx = Xtilde - algo.MChains[ch].mu

    # Get Cholesky Factorisation of Covariance matrix (before update mu)
    algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,dx)
  
    # Update mu
    algo.MChains[ch].mu +=  step * dx

    # Update Sampling Variance (If acceptance above long run target, set net wider by increasing variance, otherwise reduce it)

    #algo.MChains[ch].infos[algo.i,:accept_rate] = 0.9*algo.MChains[ch].infos[algo.i-1,:accept_rate] + 0.1*algo.MChains[ch].infos[algo.i,:accept]
    algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i
    algo.MChains[ch].shock_sd += step * (algo.MChains[ch].infos[algo.i,:accept_rate]- 0.234)
    algo.MChains[ch].infos[algo.i,:shock_sd] = algo.MChains[ch].shock_sd
end

# Random Walk adaptations: see Lacki and Miasojedow (2016) "State-dependent swap strategies ...."
function rwAdaptLocal!(algo::MAlgoABCPT, ch::Int64)
    
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
    dx = Xtilde - algo.MChains[ch].mu

    # Get Cholesky Factorisation of Covariance matrix (before update mu)
    algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,dx)
  
    # Update mu
    algo.MChains[ch].mu +=  step * dx

    # Update acceptance rate THIS IS NOT WORKING - I NEED AR FOREACH PARAM - MORE INFRASTRUCTURE NEEDED
    #algo.MChains[ch].shock_sd[algo.MChains[ch].shock_id] += step * (prob_accept - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i
    algo.MChains[ch].infos[algo.i,:shock_sd] = dot(algo.MChains[ch].shock_sd,algo.MChains[ch].shock_wgts)
    
 end

# Random Walk adaptations: see Lacki and Miasojedow (2016) "State-dependent swap strategies ...."
function rwAdapt!(algo::MAlgoABCPT, prob_accept::Float64, ch::Int64)
    
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
    dx = Xtilde - algo.MChains[ch].mu

    # Get Cholesky Factorisation of Covariance matrix (before update mu)
    algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,dx)
  
    # Update mu
    algo.MChains[ch].mu +=  step * dx

    # Update Sampling Variance (If acceptance above long run target, set net wider by increasing variance, otherwise reduce it)
    algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i
    #algo.MChains[ch].infos[algo.i,:accept_rate] = 0.9*algo.MChains[ch].infos[algo.i-1,:accept_rate] + 0.1*algo.MChains[ch].infos[algo.i,:accept]
    #algo.MChains[ch].shock_sd += step * (algo.MChains[ch].infos[algo.i,:accept_rate]- 0.234)

    # Update acceptance rate
    algo.MChains[ch].shock_sd += step * (prob_accept - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    algo.MChains[ch].infos[algo.i,:shock_sd] = algo.MChains[ch].shock_sd

end

# Random Walk adaptations: see Lacki and Miasojedow (2016) "State-dependent swap strategies ...."
function rwAdaptLocal!(algo::MAlgoABCPT, prob_accept::Float64, ch::Int64)
    
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
    dx = Xtilde - algo.MChains[ch].mu

    # Get Cholesky Factorisation of Covariance matrix (before update mu)
    algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,dx)
  
    # Update mu
    algo.MChains[ch].mu +=  step * dx

    # Update acceptance rate
    algo.MChains[ch].shock_sd[algo.MChains[ch].shock_id] += step * (prob_accept - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    
 end

# N randonly chosen pairs with replacement - this is BGP
function exchangeMovesRA!(algo::MAlgoABCPT)

    swaplist = unique(rand(1:length(algo.swapdict), algo["N"]))     # Since deterministic swap rule, no point repeating - only at draw
    for i in swaplist
        cold, hot = algo.swapdict[i]

        vcold = getEval(algo.MChains[cold],algo.i).value
        bcold = 1/algo.MChains[cold].tempering
        
        vhot = getEval(algo.MChains[hot],algo.i).value
        bhot = 1/algo.MChains[hot].tempering
        
        # Δb:=(bhot-bcold)<0. ACC wp 1 if vhot<vcol, ACC wp∈[0,1] if vhot>vcold. More likely to accept at similar temps i.e. Δb→0 
        xi = min(1, exp((bhot-bcold)*(vhot-vcold)))

        if <(vhot,algo.MChains[cold].dist_tol) & >(xi,rand())
            swapRows!(algo,Pair(cold,hot),algo.i)
        else 
            nothing
        end
    end

end

# Draw swap prob prop to swap prob
draw_swap_pair(x::Vector{Float64}) = rand(Categorical(softmax(x)))
draw_swap_pair(x::Vector{Any}) = rand(Categorical(softmax(convert(Vector{Float64},x))))

# State-dependent swap strategies (max swapping - more efficient than random pairs)
function exchangeMoves!(algo::MAlgoABCPT)
 
    swapprob = Float64[]
    for ch in 1:algo["N"], ch2 in ch+1:algo["N"]
        v1 = getEval(algo.MChains[ch],algo.i).value
        v2 = getEval(algo.MChains[ch2],algo.i).value 
        push!(swapprob, exp(-abs(v1 - v2))) 
    end

    # Get pair draw prop drawn to swap probabilities defined above.
    cold, hot = algo.swapdict[draw_swap_pair(swapprob)]

    vcold = getEval(algo.MChains[cold],algo.i).value
    bcold = 1/algo.MChains[cold].tempering
    
    vhot = getEval(algo.MChains[hot],algo.i).value
    bhot = 1/algo.MChains[hot].tempering
    
    # Δb:=(bhot-bcold)<0. ACC wp 1 if vhot<vcol, ACC wp∈[0,1] if vhot>vcold. More likely to accept at similar temps i.e. Δb→0 
    xi = min(1, exp((bhot-bcold)*(vhot-vcold)))

    if xi > rand()
        swapRows!(algo,CC,Pair(cold,hot),algo.i)
    end

end


function swapRows!(algo::MAlgoABCPT,pair::Pair,i::Int)

    e1 = getEval(algo.MChains[pair.first] ,i)
    e2 = getEval(algo.MChains[pair.second],i)

    # Initial ID instead of exchanging
    init_id1 = algo.MChains[pair.first].infos[i,:init_id]
    init_id2 = algo.MChains[pair.second].infos[i,:init_id]

    # swap
    appendEval!(algo.MChains[pair.first  ],e2)
    appendEval!(algo.MChains[pair.second ],e1)

    # make a note in infos
    algo.MChains[pair.first].infos[i,:init_id] = init_id2
    algo.MChains[pair.second].infos[i,:init_id] = init_id1

end

# Temperature Adaption - note: use objective values after swapping 
function tempAdapt!(algo::MAlgoABCPT)

    step = (algo.i+1)^(-0.5)  # Declining step size over iterations

    # Get adjustment for temperature of all but coldest chain
    for ch in 2:algo["N"]
        v1 = getEval(algo.MChains[ch-1],algo.i).value
        b1 = 1/algo.MChains[ch-1].tempering
        
        v2 = getEval(algo.MChains[ch],algo.i).value
        b2 = 1/algo.MChains[ch].tempering
        
        xi = min(1, exp((b2-b1)*(v2 - v1))) # Since b2-b1<0, will accept wp1 when v2-v1<0 (i.e. v2<v1). If similar temps, Δb→0 more likely to accept.   
        algo.MChains[ch].reltemp += step * (xi - 0.234)
    end

    # Get new temp spacing: Start at T1, add new spacing on to define new temperatures
    for ch in 2:algo["N"]
        algo.MChains[ch].tempering = algo.MChains[ch-1].tempering + exp(algo.MChains[ch].reltemp)
    end    
end

# function getNewCandidates!
function getNewCandidates!(algo::MAlgoABCPT)

    # Number of parameters
    D = size(algo.MChains[1].F,1)

    # update chain by chain
    for ch in 1:algo["N"]
        # shock parameters on chain index ch
        shock = exp(algo.MChains[ch].shock_sd[1]).*tril(algo.MChains[ch].F)*randn(D)    # Draw shocks scaled to ensure acceptance rate targeted at 0.234 (See Lacki and Meas)
        shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
        jumpParamsBnd!(algo,ch,shockd)                              # Add to parameters
    end

end

# No boundaries
function jumpParams!(algo::MAlgoABCPT,ch::Int,shock::Dict)
    eval_old = getLastEval(algo.MChains[ch])
    for k in keys(eval_old.params)
        algo.current_param[ch][k] = eval_old.params[k] + shock[k]
    end
end 

# With boundaries
function jumpParamsBnd!(algo::MAlgoABCPT,ch::Int,shock::Dict)
    eval_old = getLastEval(algo.MChains[ch])
    for k in keys(eval_old.params)
        algo.current_param[ch][k] = max(algo.m.params_to_sample[k][:lb],
                                         min(eval_old.params[k] + shock[k], 
                                               algo.m.params_to_sample[k][:ub]))
    end
end                                                

get_eigs(W::Matrix{Float64}) = svdvals(W).^2

function draw_from_ppca(M::MultivariateStats.PPCA{Float64}) 
    W = MultivariateStats.loadings(M)     # Loadings of PPCA model
    s = svdvals(W)
    ρ = softmax(s.^2)    # U,S,V = svd(W) -> U are PC, S are sqrt of eigenvalues since derived from centered data
    l = rand(Distributions.Categorical(ρ))
    return ρ, l, s[l], W[:,l]
end


# function getNewCandidates!
function getNewCandidatesPPCA!(algo::MAlgoABCPT, method::Symbol)

    # Number of parameters
    D = size(algo.MChains[1].F,1)

    # update chain by chain
    for ch in 1:algo["N"]

        S = A_mul_Bt(algo.MChains[ch].F,algo.MChains[ch].F)
        if method==:em
             M = MultivariateStats.ppcaem(S, algo.MChains[ch].mu, D; maxoutdim = algo["maxoutdim"]) 
        elseif method==:bayes
             M = MultivariateStats.bayespca(S, algo.MChains[ch].mu, D; maxoutdim = algo["maxoutdim"])
        else 
            println("Error: not an option for online PPCA")
        end

        # shock parameters on chain index ch
        (ρbar,l_id,σ,w) = draw_from_ppca(M)
        λ = exp(algo.MChains[ch].shock_sd[l_id])                # Scaling
        shock = λ*σ*randn()*w                                   # Vector - shock
        shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
        jumpParamsBnd!(algo,ch,shockd)                              # Add to parameters

        # Records which principal component changes
        algo.MChains[ch].shock_id = l_id
        fill!(algo.MChains[ch].shock_wgts,0.)
        for (n,k) in enumerate(ρbar)
            algo.MChains[ch].shock_wgts[n] = k
        end
    end
end


# function getNewCandidates!
function getNewCandidatesCompWise!(algo::MAlgoABCPT)

   # Number of parameters
    D = size(algo.MChains[1].F,1)

    # update chain by chain
    for ch in 1:algo["N"]

        # shock parameters on chain index ch
        l_id = rand(1:D)                            # Which component changing
        λ = exp(algo.MChains[ch].shock_sd[l_id])    # Scaling of chosen component
        S = A_mul_Bt(algo.MChains[ch].F,algo.MChains[ch].F)
        σ = sqrt(S[l_id,l_id])

        eval_old = getLastEval(algo.MChains[ch])
        for (n,k) in enumerate(keys(eval_old.params))
            if n==l_id 
                algo.current_param[ch][k] = max(algo.m.params_to_sample[k][:lb], 
                                                min(eval_old.params[k] + λ*σ*randn(),
                                                    algo.m.params_to_sample[k][:ub]))
            else 
                algo.current_param[ch][k] =  eval_old.params[k]
            end
        end

        # Records which component changes
        algo.MChains[ch].shock_id = l_id
        fill!(algo.MChains[ch].shock_wgts,1/D)
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
