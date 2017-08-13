
# No boundaries
function jumpParams!(algo::MAlgoABCPT,ch::Int,shock::Dict)
    eval_old = getLastEval(algo.MChains[ch])
    for k in keys(eval_old.params)
        algo.current_param[ch][k] = eval_old.params[k] + shock[k]
    end
end 

# function getNewCandidates!
function getNewCandidates!(algo::MAlgoABCPT)

    # Number of parameters
    D = length(ps2s_names(algo.m))

    # update chain by chain
    for ch in 1:algo["N"]
        # shock parameters on chain index ch
        shock = exp(algo.MChains[ch].shock_sd[1]).*tril(algo.MChains[ch].F)*randn(D)    # Draw shocks scaled to ensure acceptance rate targeted at 0.234 (See Lacki and Meas)
        shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
        jumpParams!(algo,ch,shockd)                              # Add to parameters
    end

end

function getNewCandidatesBnd!(algo::MAlgoABCPT)

    # Number of parameters
    D = length(ps2s_names(algo.m))

    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 
    lb = [v[:lb] for (k,v) in algo.m.params_to_sample]
    ub = [v[:ub] for (k,v) in algo.m.params_to_sample]

    # update chain by chain
    for ch in 1:algo["N"]
        for zz in 1:algo["maxdrawiters"]
            # shock parameters on chain index ch
            shock = exp(algo.MChains[ch].shock_sd).*tril(algo.MChains[ch].F)*randn(D)    # Draw shocks scaled to ensure acceptance rate targeted at 0.234 (See Lacki and Meas)
            shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
            jumpParams!(algo,ch,shockd)                              # Add to parameters
            draw = [algo.current_param[ch][k] for (k,v) in algo.m.params_to_sample]
            all(draw.>lb) && all(draw.<ub) && break
            if mod(zz,100)==0 
                println("Reduced Shock SD on Chain $(ch) because not in bounds ")
                algo.MChains[ch].shock_sd = log((1.-step)*exp(algo.MChains[ch].shock_sd))
            elseif zz==algo["maxdrawiters"] 
                println("Exceeded Maximum Number Draws on Chain $(ch) - Defaulted to previous draw")
                shockd = Dict(zip(ps2s_names(algo) , zeros(D)))  
                jumpParams!(algo,ch,shockd)
            end
        end
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
function getNewCandidatesPPCABnd!(algo::MAlgoABCPT, method::Symbol)

    # Number of parameters
    D =  length(ps2s_names(algo.m))

    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 
    lb = [v[:lb] for (k,v) in algo.m.params_to_sample]
    ub = [v[:ub] for (k,v) in algo.m.params_to_sample]
    if algo["CommonCov"]
        S = A_mul_Bt(algo.MChains[1].F,algo.MChains[1].F)
        # shock parameters on chain index ch
        for ch in 1:algo["N"]
            if method==:em
                 M = MultivariateStats.ppcaem(S, algo.MChains[ch].mu, D) 
            elseif method==:bayes
                 M = MultivariateStats.bayespca(S, algo.MChains[ch].mu, D)
            else 
                println("Error: not an option for online PPCA")
            end
            @inbounds for zz in 1:algo["maxdrawiters"]
                (ρbar,l_id,σ,w) = draw_from_ppca(M)
                # Records which principal component changes
                λ = exp(algo.MChains[ch].shock_sd[l_id])                # Scaling
                shock = λ*σ*randn()*w                                   # Vector - shock
                shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
                jumpParams!(algo,ch,shockd)                              # Add to parameters
                draw = [algo.current_param[ch][k] for (k,v) in algo.m.params_to_sample]
                all(draw.>lb) && all(draw.<ub) && break
                if mod(zz,100)==0 
                    println("Reduced Shock SD on Chain $(ch) because not in bounds ")
                    algo.MChains[ch].shock_sd = log((1.-step)*exp(algo.MChains[ch].shock_sd))
                elseif zz==algo["maxdrawiters"] 
                    println("Exceeded Maximum Number Draws on Chain $(ch) - Defaulted to previous draw")
                    shockd = Dict(zip(ps2s_names(algo) , zeros(D)))  
                    jumpParams!(algo,ch,shockd)
                end
                algo.MChains[ch].shock_id = l_id
                fill!(algo.MChains[ch].shock_wgts,0.)
                for (n,k) in enumerate(ρbar)
                    algo.MChains[ch].shock_wgts[n] = k
                end
 
            end
        end        

    else 
        # update chain by chain
        for ch in 1:algo["N"]
            S = A_mul_Bt(algo.MChains[ch].F,algo.MChains[ch].F)
            if method==:em
                 M = MultivariateStats.ppcaem(S, algo.MChains[ch].mu, D) 
            elseif method==:bayes
                 M = MultivariateStats.bayespca(S, algo.MChains[ch].mu, D)
            else 
                println("Error: not an option for online PPCA")
            end
            @inbounds for zz in 1:algo["maxdrawiters"]
                # shock parameters on chain index ch
                (ρbar,l_id,σ,w) = draw_from_ppca(M)
                λ = exp(algo.MChains[ch].shock_sd[l_id])                # Scaling
                shock = λ*σ*randn()*w                                   # Vector - shock
                shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
                jumpParams!(algo,ch,shockd)                              # Add to parameters
                draw = [algo.current_param[ch][k] for (k,v) in algo.m.params_to_sample]
                all(draw.>lb) && all(draw.<ub) && break
                if mod(zz,100)==0 
                    println("Reduced Shock SD on Chain $(ch) because not in bounds ")
                    algo.MChains[ch].shock_sd = log((1.-step)*exp(algo.MChains[ch].shock_sd))
                elseif zz==algo["maxdrawiters"] 
                    println("Exceeded Maximum Number Draws on Chain $(ch) - Defaulted to previous draw")
                    shockd = Dict(zip(ps2s_names(algo) , zeros(D)))  
                    jumpParams!(algo,ch,shockd)
                end
                # Records which principal component changes
                algo.MChains[ch].shock_id = l_id
                fill!(algo.MChains[ch].shock_wgts,0.)
                for (n,k) in enumerate(ρbar)
                    algo.MChains[ch].shock_wgts[n] = k
                end
            end

        end
    end
end

# function getNewCandidates!
function getNewCandidatesPPCA!(algo::MAlgoABCPT, method::Symbol)

    # Number of parameters
    D =  length(ps2s_names(algo.m))

    if algo["CommonCov"]
        S = A_mul_Bt(algo.MChains[1].F,algo.MChains[1].F)
        if method==:em
             M = MultivariateStats.ppcaem(S, algo.MChains[ch].mu, D) 
        elseif method==:bayes
             M = MultivariateStats.bayespca(S, algo.MChains[ch].mu, D)
        else 
            println("Error: not an option for online PPCA")
        end
        # shock parameters on chain index ch
        for ch in 1:algo["N"]
            (ρbar,l_id,σ,w) = draw_from_ppca(M)
            λ = exp(algo.MChains[ch].shock_sd[l_id])                # Scaling
            shock = λ*σ*randn()*w                                   # Vector - shock
            shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
            jumpParams!(algo,ch,shockd)                              # Add to parameters

            # Records which principal component changes
            algo.MChains[ch].shock_id = l_id
            fill!(algo.MChains[ch].shock_wgts,0.)
            for (n,k) in enumerate(ρbar)
                algo.MChains[ch].shock_wgts[n] = k
            end
        end        

    else 
        # update chain by chain
        for ch in 1:algo["N"]
            S = A_mul_Bt(algo.MChains[ch].F,algo.MChains[ch].F)
            if method==:em
                 M = MultivariateStats.ppcaem(S, algo.MChains[ch].mu, D) 
            elseif method==:bayes
                 M = MultivariateStats.bayespca(S, algo.MChains[ch].mu, D)
            else 
                println("Error: not an option for online PPCA")
            end

            # shock parameters on chain index ch
            (ρbar,l_id,σ,w) = draw_from_ppca(M)
            λ = exp(algo.MChains[ch].shock_sd[l_id])                # Scaling
            shock = λ*σ*randn()*w                                   # Vector - shock
            shockd = Dict(zip(ps2s_names(algo) , shock))             # Put in a dictionary
            jumpParams!(algo,ch,shockd)                              # Add to parameters

            # Records which principal component changes
            algo.MChains[ch].shock_id = l_id
            fill!(algo.MChains[ch].shock_wgts,0.)
            for (n,k) in enumerate(ρbar)
                algo.MChains[ch].shock_wgts[n] = k
            end
        end
    end
end

# function getNewCandidates!
function getNewCandidatesCompWise!(algo::MAlgoABCPT)

   # Number of parameters
    D = length(ps2s_names(algo.m))

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
                #algo.current_param[ch][k] = eval_old.params[k] + λ*σ*randn()
                algo.current_param[ch][k] = eval_old.params[k] + λ*σ*randn()
            else 
                algo.current_param[ch][k] =  eval_old.params[k]
            end
        end

        # Records which component changes
        algo.MChains[ch].shock_id = l_id
        fill!(algo.MChains[ch].shock_wgts,1/D)
    end
end

# function getNewCandidates!
function getNewCandidatesCompWiseBnd!(algo::MAlgoABCPT)

   # Number of parameters
    D = length(ps2s_names(algo.m))
    lb = [v[:lb] for (k,v) in algo.m.params_to_sample]
    ub = [v[:ub] for (k,v) in algo.m.params_to_sample]
    # update chain by chain
    for ch in 1:algo["N"]
        @inbounds for zz in 1:algo["maxdrawiters"]
            # shock parameters on chain index ch
            l_id = rand(1:D)                            # Which component changing
            λ = exp(algo.MChains[ch].shock_sd[l_id])    # Scaling of chosen component
            S = A_mul_Bt(algo.MChains[ch].F,algo.MChains[ch].F)
            σ = sqrt(S[l_id,l_id])

            eval_old = getLastEval(algo.MChains[ch])
            for (n,k) in enumerate(keys(eval_old.params))
                if n==l_id
                    algo.current_param[ch][k] = eval_old.params[k] + λ*σ*randn()
                else 
                    algo.current_param[ch][k] =  eval_old.params[k]
                end
            end
            draw = [algo.current_param[ch][k] for (k,v) in algo.m.params_to_sample]
            all(draw.>lb) && all(draw.<ub) && break
            if zz.==algo["maxdrawiters"] 
               println("Exceeded Maximum Number Draws on Chain $(ch) - Defaulted to previous draw")
               for (n,k) in enumerate(keys(eval_old.params))
                    algo.current_param[ch][k] =  eval_old.params[k]    
               end
            end   
        end

        # Records which component changes
        algo.MChains[ch].shock_id = l_id
        fill!(algo.MChains[ch].shock_wgts,1/D)
    end
end
