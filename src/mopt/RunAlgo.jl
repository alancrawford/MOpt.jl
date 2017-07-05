
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
            getNewCandidates!(algo)               # Return a vector of parameters dicts in algo.current_param
        elseif algo["mc_update"]==:COMPWISE
            ρbar = getNewCandidatesCompWise!(algo)       # Return a vector of parameters dicts in algo.current_param
        elseif algo["mc_update"]==:PPCA
            ρbar = getNewCandidatesPPCA!(algo,algo["ppca_method"])       # Return a vector of parameters dicts in algo.current_param
        end
    end

    # evaluate objective on all chains
    # --------------------------------
    EV = pmap( x -> evaluateObjective(algo.m,x), algo.current_param)

    # Part 1) LOCAL MOVES ABC-MCMC for i={1,...,N}. accept/reject
    # -----------------------------------------------------------
    pvec = doAcceptReject!(algo,EV)
    do_rwAdapt!(algo,pvec)

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

function doAcceptReject!(algo::MAlgoABCPT,EV::Array{Eval})
    
    pvec = Float64[]
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
        push!(pvec,prob)
    end

    return pvec
end

function do_rwAdapt!(algo::MAlgoABCPT,pvec::Vector{Float64})
    if algo["mc_update"]==:GLOBAL 
        if algo["CommonCov"]            
            rwAdapt!(algo,pvec)
        else
            @inbounds for ch in 1:algo["N"]        
                rwAdapt!(algo,pvec[ch],ch)
            end
        end
    else
        if algo["CommonCov"]            
            rwAdaptLocal!(algo,pvec)
        else
            @inbounds for ch in 1:algo["N"]        
                rwAdaptLocal!(algo,pvec[ch],ch)
            end
        end
    end
end

