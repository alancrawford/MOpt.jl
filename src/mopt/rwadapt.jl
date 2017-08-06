# This is development file

# Rank-one-update with stochastic approximation : See p358 of Andrieu and Thoms (2008) 'A tutorial on adaptive MCMC', Journal of Statistical Computing
rank1update(F::Matrix{Float64}, x::Vector{Float64}, Nx::Int64) = tril(inv(tril(F))*A_mul_Bt(x,x)*inv(tril(F)') - eye(Nx))
rank1update(F::Matrix{Float64}, Σ::Matrix{Float64}, Nx::Int64) = tril(inv(tril(F))*Σ*inv(tril(F)') - eye(Nx))

rank1update(F::Matrix{Float64}, x::Vector{Float64}, Nx::Int64, ρ::Float64) = tril(inv(tril(F))*((1-ρ)*A_mul_Bt(x,x)+ρ*eye(Nx))*inv(tril(F)') - eye(Nx))
rank1update(F::Matrix{Float64}, Σ::Matrix{Float64}, Nx::Int64, ρ::Float64) = tril(inv(tril(F))*((1-ρ)*Σ+ρ*eye(Nx))*inv(tril(F)') - eye(Nx))

# ----------- 1. Single Scaling for all parameters  ----------- #

# Random Walk adaptations: see Lacki and Miasojedow (2016) "State-dependent swap strategies ...."

# a.i)  Chain specific: Covariance and Scaling - NOT Regularised
function rwAdapt!(algo::MAlgoABCPT, prob_accept::Float64, ch::Int64)
    
    Nx = length(MOpt.ps2s_names(algo.m))
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
    dx = Xtilde - algo.MChains[ch].mu

    # Get Cholesky Factorisation of Covariance matrix (before update mu)
    #algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,dx,Nx)
    if algo.i > algo["TempAdapt"]
        lower_bound_index = maximum([1,algo.i-algo["past_iterations"]])
        Σ = cov(convert(Matrix,MOpt.parameters(algo.MChains[ch],lower_bound_index:algo.i)))
        algo.MChains[ch].F = convert(Matrix,cholfact(Σ)[:L])
    end
    # Update mu
    algo.MChains[ch].mu +=  step * dx

    # Update Sampling Variance (If acceptance above long run target, set net wider by increasing variance, otherwise reduce it)
    algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i
   
    # Update acceptance rate
    algo.MChains[ch].shock_sd += step * (prob_accept - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    #algo.MChains[ch].shock_sd += step * (algo.MChains[ch].infos[algo.i,:accept_rate]- 0.234)
    algo.MChains[ch].infos[algo.i,:shock_sd] = algo.MChains[ch].shock_sd

end

# a.ii)  Chain specific: Covariance and Scaling - Regularised
function rwAdapt!(algo::MAlgoABCPT, prob_accept::Float64, ch::Int64, ρ::Float64)
    
    Nx = length(MOpt.ps2s_names(algo.m))
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
    dx = Xtilde - algo.MChains[ch].mu

    # Get Cholesky Factorisation of Covariance matrix (before update mu)
    #algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,dx,Nx,ρ)
    if algo.i > algo["TempAdapt"]
        lower_bound_index = maximum([1,algo.i-algo["past_iterations"]])
        Σ = (1-ρ).*cov(convert(Matrix,MOpt.parameters(algo.MChains[ch],lower_bound_index:algo.i))) + ρ.*eye(Nx)
        algo.MChains[ch].F = convert(Matrix,cholfact(Σ)[:L])    
    end

    # Update mu
    algo.MChains[ch].mu +=  step * dx

    # Update Sampling Variance (If acceptance above long run target, set net wider by increasing variance, otherwise reduce it)
    algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i
   
    # Update acceptance rate
    algo.MChains[ch].shock_sd += step * (prob_accept - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    #algo.MChains[ch].shock_sd += step * (algo.MChains[ch].infos[algo.i,:accept_rate]- 0.234)
    algo.MChains[ch].infos[algo.i,:shock_sd] = algo.MChains[ch].shock_sd

end

# b.i) Common Covariance matrix across chains but separate scaling factor using prob of acceptance - NOT Regularised
function rwAdapt!(algo::MAlgoABCPT, pvec::Vector{Float64})
    
    Nx = length(MOpt.ps2s_names(algo.m))
    #Σ = zeros(Nx,Nx)     # Update for Covariance Matrix: pooling information across chains
    Δmu = zeros(Nx)      # Update for mu Matrix: pooling information across chains

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    @inbounds for ch in 1:algo["N"]
        Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
        dx = Xtilde - algo.MChains[ch].mu
        #Σ += A_mul_Bt(dx,dx)/algo["N"]      # Cov matrix update 
        Δmu += dx/algo["N"]                 # Mean update
    end

    # Read updates into chains
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 
    
    if algo.i > algo["TempAdapt"]
        lower_bound_index = maximum([1,algo.i-algo["past_iterations"]])
        Σ = (1-ρ).*cov(convert(Matrix,MOpt.parameters(algo.MChains[ch],lower_bound_index:algo.i))) + ρ.*eye(Nx)
        F = convert(Matrix,cholfact(Σ)[:L])
    end
    
    @inbounds for ch in 1:algo["N"]
        algo.MChains[ch].F = F
        algo.MChains[ch].mu +=  step * Δmu
        algo.MChains[ch].shock_sd += step * (pvec[ch] - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
        algo.MChains[ch].infos[algo.i,:shock_sd] = algo.MChains[ch].shock_sd
        algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i  # Update acceptance rate
    end
     
end

# b.ii) Common Covariance matrix across chains but separate scaling factor using prob of acceptance - Regularised
function rwAdapt!(algo::MAlgoABCPT, pvec::Vector{Float64}, ρ::Float64)
    
    Nx = length(MOpt.ps2s_names(algo.m))
    #Σ = zeros(Nx,Nx)     # Update for Covariance Matrix: pooling information across chains
    Δmu = zeros(Nx)      # Update for mu Matrix: pooling information across chains
    
    # Get value of accepted (i.e. old or new) parameters in chain after MH
    @inbounds for ch in 1:algo["N"]
        Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
        dx = Xtilde - algo.MChains[ch].mu
        Δmu += dx/algo["N"]                 # Mean update
    end

    # Read updates into chains
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 
    if algo.i > algo["TempAdapt"]
        lower_bound_index = maximum([1,algo.i-algo["past_iterations"]])
        Σ = (1-ρ).*cov(convert(Matrix,MOpt.parameters(algo.MChains[ch],lower_bound_index:algo.i))) + ρ.*eye(Nx)
        F = convert(Matrix,cholfact(Σ)[:L]) 
    end
    @inbounds for ch in 1:algo["N"]
        algo.MChains[ch].F = F
        algo.MChains[ch].mu +=  step * Δmu
        algo.MChains[ch].shock_sd += step * (pvec[ch] - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
        algo.MChains[ch].infos[algo.i,:shock_sd] = algo.MChains[ch].shock_sd
        #algo.MChains[ch].shock_sd += step * (algo.MChains[ch].infos[algo.i,:accept_rate]- 0.234)
        algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i  # Update acceptance rate
    end
     
end


# ----------- 2. Parameter specific scaling   ----------- #

#=
# a.i) Chain specific: Covariance and Scaling - NOT Regularised
function rwAdaptLocal!(algo::MAlgoABCPT, prob_accept::Float64, ch::Int64)
    
    Nx = length(ps2s_names(algo.m))
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
    dx = Xtilde - algo.MChains[ch].mu

    # Get Cholesky Factorisation of Covariance matrix (before update mu)
    #algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,dx,Nx)
    if algo.i > algo["TempAdapt"]
        lower_bound_index = maximum([1,algo.i-algo["past_iterations"]])
        Σ = cov(convert(Matrix,MOpt.parameters(algo.MChains[ch],lower_bound_index:algo.i)))
        algo.MChains[ch].F = convert(Matrix,cholfact(Σ)[:L])
    end
    # Update mu
    algo.MChains[ch].mu +=  step * dx

    # Update acceptance rate
    algo.MChains[ch].shock_sd[algo.MChains[ch].shock_id] += step * (prob_accept - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    #algo.MChains[ch].shock_sd[algo.MChains[ch].shock_id] += step * (algo.MChains[ch].infos[algo.i,:accept_rate] - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    algo.MChains[ch].infos[algo.i,:shock_sd] = dot(algo.MChains[ch].shock_sd,algo.MChains[ch].shock_wgts)
    algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i
    
end

# a.ii)  Chain specific: Covariance and Scaling - Regularised
function rwAdaptLocal!(algo::MAlgoABCPT, prob_accept::Float64, ch::Int64, ρ::Float64)
    
    Nx = length(MOpt.ps2s_names(algo.m))
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
    dx = Xtilde - algo.MChains[ch].mu

    # Get Cholesky Factorisation of Covariance matrix (before update mu)
    #algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,dx,Nx,ρ)
    if algo.i > algo["TempAdapt"]
        lower_bound_index = maximum([1,algo.i-algo["past_iterations"]])
        Σ = (1-ρ).*cov(convert(Matrix,MOpt.parameters(algo.MChains[ch],lower_bound_index:algo.i))) + ρ.*eye(Nx)
        algo.MChains[ch].F = convert(Matrix,cholfact(Σ)[:L]) 
    end
    # Update mu
    algo.MChains[ch].mu +=  step * dx

    # Update acceptance rate
    algo.MChains[ch].shock_sd[algo.MChains[ch].shock_id] += step * (prob_accept - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    #algo.MChains[ch].shock_sd[algo.MChains[ch].shock_id] += step * (algo.MChains[ch].infos[algo.i,:accept_rate] - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
    algo.MChains[ch].infos[algo.i,:shock_sd] = dot(algo.MChains[ch].shock_sd,algo.MChains[ch].shock_wgts)
    algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i

end

# b.i) Common Covariance matrix across chains but separate scaling factor using prob of acceptance - NOT Regularised
function rwAdaptLocal!(algo::MAlgoABCPT, pvec::Vector{Float64})
    Nx = length(MOpt.ps2s_names(algo.m))
    #Σ = zeros(Nx,Nx)     # Update for Covariance Matrix: pooling information across chains
    Δmu = zeros(Nx)      # Update for mu Matrix: pooling information across chains

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    @inbounds for ch in 1:algo["N"]
        Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
        dx = Xtilde - algo.MChains[ch].mu
        #Σ += A_mul_Bt(dx,dx)/algo["N"]      # Cov matrix update 
        Δmu += dx/algo["N"]                 # Mean update
    end

    # Read updates into chains
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations
    lower_bound_index = maximum([1,algo.i-algo["past_iterations"]])
    Σ = cov(convert(Matrix,MOpt.parameters(algo.MChains[1:algo["N"]],lower_bound_index:algo.i)))

    @inbounds for ch in 1:algo["N"]
        #algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,Σ,Nx)
        algo.MChains[ch].F = convert(Matrix,cholfact(Σ)[:L])
        algo.MChains[ch].mu +=  step * Δmu
        algo.MChains[ch].shock_sd[algo.MChains[ch].shock_id] += step * (pvec[ch] - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
        algo.MChains[ch].infos[algo.i,:shock_sd] = dot(algo.MChains[ch].shock_sd,algo.MChains[ch].shock_wgts)
        #algo.MChains[ch].shock_sd += step * (algo.MChains[ch].infos[algo.i,:accept_rate]- 0.234)
        algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i  # Update acceptance rate
    end

 end

# b.ii) Common Covariance matrix across chains but separate scaling factor using prob of acceptance - Regularised
function rwAdaptLocal!(algo::MAlgoABCPT, pvec::Vector{Float64},ρ::Float64)
    Nx = length(MOpt.ps2s_names(algo.m))
    #Σ = zeros(Nx,Nx)     # Update for Covariance Matrix: pooling information across chains
    Δmu = zeros(Nx)      # Update for mu Matrix: pooling information across chains

    # Get value of accepted (i.e. old or new) parameters in chain after MH
    @inbounds for ch in 1:algo["N"]
        Xtilde = convert(Array,parameters(algo.MChains[ch],algo.i)[:, ps2s_names(algo.m)])[:]
        dx = Xtilde - algo.MChains[ch].mu
        #Σ += A_mul_Bt(dx,dx)/algo["N"]      # Cov matrix update 
        Δmu += dx/algo["N"]                 # Mean update
    end

    # Read updates into chains
    step = (algo.i+1)^(-0.5)  # Declining step size over iterations 
    lower_bound_index = maximum([1,algo.i-algo["past_iterations"]])
    Σ = (1-ρ).*cov(convert(Matrix,MOpt.parameters(algo.MChains[1:algo["N"]],lower_bound_index:algo.i))) + ρ.*eye(Nx)

    @inbounds for ch in 1:algo["N"]
        #algo.MChains[ch].F += step*algo.MChains[ch].F*rank1update(algo.MChains[ch].F,Σ,Nx,ρ)
        algo.MChains[ch].F = convert(Matrix,cholfact(Σ)[:L])
        algo.MChains[ch].mu +=  step * Δmu
        algo.MChains[ch].shock_sd[algo.MChains[ch].shock_id] += step * (pvec[ch] - 0.234)   # Quite a simple update - maybe be slow. See AT 2008 sec 5.
        algo.MChains[ch].infos[algo.i,:shock_sd] = dot(algo.MChains[ch].shock_sd,algo.MChains[ch].shock_wgts)
        #algo.MChains[ch].shock_sd += step * (algo.MChains[ch].infos[algo.i,:accept_rate]- 0.234)
        algo.MChains[ch].infos[algo.i,:accept_rate] = sum(algo.MChains[ch].infos[1:algo.i,:accept])/algo.i  # Update acceptance rate
    end

 end
 =#