
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

function tempProj!(x::Real, Tmax::Real, N::Real) 
    x -= 1.
    x /= (N-1)
    x *= (Tmax-1.0)
    x += 1
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

    # Projection of temperature onto [1,algo["Tmax"]]
    if algo.MChains[algo["N"]].tempering > algo["Tmax"]
        for ch in 2:algo["N"]
            tempProj!(algo.MChains[ch].tempering, algo["Tmax"], algo.MChains[algo["N"]].tempering)
            algo.MChains[ch].reltemp = log(algo.MChains[ch].tempering) - log(algo.MChains[ch-1].tempering)
        end
    end
end
