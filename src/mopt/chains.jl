


# defining an abstract chain type in case 
# some algorithm need additional information
# on top of (eval, moment, parameter)
abstract AbstractChain


# the default chain type
# we create a dictionary with arrays
# for each parameters
type Chain <: AbstractChain
    i            :: Int              # current index
    infos        :: DataFrame
    parameters   :: DataFrame
    moments      :: DataFrame
    params_nms   :: Array{Symbol,1}  # names of parameters (i.e. exclusive of "id" or "iter", etc)
    params2s_nms :: Array{Symbol,1}  # names of parameters to sample
    moments_nms  :: Array{Symbol,1}  # names of moments
  
    function Chain(MProb,L)
        infos      = DataFrame(iter=1:L, evals =zeros(Float64,L), accept = zeros(Bool,L), status = zeros(Int,L), exhanged_with=zeros(Int,L), prob=zeros(Float64,L), eval_time=zeros(Float64,L))
        par_nms    = Symbol[ Symbol(x) for x in ps_names(MProb) ]
        par2s_nms  = Symbol[ Symbol(x) for x in ps2s_names(MProb) ]
        mom_nms    = Symbol[ Symbol(x) for x in ms_names(MProb) ]
        parameters = convert(DataFrame,zeros(L,length(par2s_nms)+1))
        moments    = convert(DataFrame,zeros(L,length(mom_nms)+1))
        names!(parameters,[:iter; par2s_nms])
        names!(moments   ,[:iter; mom_nms])
        return new(0,infos,parameters,moments,par_nms,par2s_nms,mom_nms)
    end
end

# methods for a single chain
# ==========================

# function parameters(c::AbstractChain, i::Union{Integer, UnitRange{Int}),all=false)
#     if all
#         c.parameters[i,:]
#     else
#         c.parameters[i,c.params2s_nms]
#     end
# end
function parameters(c::AbstractChain, i::Union{Integer, UnitRange{Int}})
    c.parameters[i,c.params2s_nms]
end

function parameters(c::AbstractChain, i::Vector{Int64})
    c.parameters[i,c.params2s_nms]
end

function parameters(c::AbstractChain)
    c.parameters[:,c.params2s_nms]
end

function parameters_ID(c::AbstractChain, i::Union{Integer, UnitRange{Int}})
    c.parameters[i,[:chain_id,:iter,c.params2s_nms]]
end
function parameters_ID(c::AbstractChain)
    c.parameters[:,[:chain_id,:iter,c.params2s_nms]]
end
moments(c::AbstractChain)                       = c.moments
moments(c::AbstractChain, i::UnitRange{Int})    = c.moments[i,:]
moments(c::AbstractChain, i::Int)               = moments(c, i:i)
infos(c::AbstractChain)                         = c.infos
infos(c::AbstractChain, i::UnitRange{Int})      = c.infos[i,:]
infos(c::AbstractChain, i::Int)                 = infos(c, i:i)
evals(c::AbstractChain, i::UnitRange{Int})      = c.infos[i,:evals]
evals(c::AbstractChain, i::Int)                 = evals(c, i:i)
allstats(c::AbstractChain,i::UnitRange{Int})    = hcat(c.infos[i,:],c.parameters[i,:],c.moments[i,:])
allstats(c::AbstractChain,i::Int)               = hcat(infos(c, i:i),parameters(c, i:i),moments(c, i:i))

# ---------------------------  CHAIN GETTERS / SETTERS ----------------------------
export getEval, getLastEval, appendEval

function getEval(chain::AbstractChain, i::Int64)
    ev = Eval()
    ev.value  = chain.infos[i,:evals]
    ev.time   = chain.infos[i,:eval_time]
    ev.status = chain.infos[i,:status]

    for k in names(chain.parameters)
        if !(k in [:chain_id,:iter])
            ev.params[k] = chain.parameters[i,k]
        end
    end
    for k in names(chain.moments)
        if !(k in [:chain_id,:iter])
            ev.simMoments[k] = chain.moments[i,k]
        end
    end

    return (ev)
end

getLastEval(chain :: AbstractChain) = getEval(chain::AbstractChain, chain.i - 1 )

function appendEval!(chain::AbstractChain, ev:: Eval, ACC::Bool, prob::Float64)
    chain.infos[chain.i,:evals]  = ev.value
    chain.infos[chain.i,:prob]   = prob
    chain.infos[chain.i,:accept] = ACC
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

# methods for an array of chains
# ==============================

# TODO ideally dispatch on MC::Array{AbstractChain}
# but that doesn't work. errors with
# no method for parameters(BGPChain)
#
# return an vcat of params from all chains
function parameters(MC::Array,i::Union{Integer, UnitRange{Int}})
    if !isa(MC[1],AbstractChain)
        error("must give array of AbstractChain") 
    end
    r = parameters(MC[1],i) 
    if length(MC)>1
        for ix=2:length(MC)
            r = vcat(r,parameters(MC[ix],i))
        end
    end
    return r
end

function parameters_ID(MC::Array,i::Union{Integer, UnitRange{Int}})
    if !isa(MC[1],AbstractChain)
        error("must give array of AbstractChain") 
    end
    r = parameters_ID(MC[1],i) 
    if length(MC)>1
        for ix=2:length(MC)
            r = vcat(r,parameters_ID(MC[ix],i))
        end
    end
    return r
end

function parameters(MC::Array)
    parameters(MC,1:nrow(MC[1].parameters))
end

function parameters_ID(MC::Array)
    parameters_ID(MC,1:nrow(MC[1].parameters))
end

function moments(MC::Array,i::Union{Integer, UnitRange{Int}})
    if !isa(MC[1],AbstractChain)
        error("must give array of AbstractChain") 
    end
    r = moments(MC[1],i) 
    if length(MC)>1
        for ix=2:length(MC)
            r = vcat(r,moments(MC[ix],i))
        end
    end
    return r
end

function moments(MC::Array)
    moments(MC,1:MC[1].i)
end

function infos(MC::Array,i::Int)
    infos(MC,i:i)
end

function infos(MC::Array,i::UnitRange{Int})
    if !isa(MC[1],AbstractChain)
        error("must give array of AbstractChain") 
    end
    r = infos(MC[1],i) 
    if length(MC)>1
        for ix=2:length(MC)
            r = vcat(r,infos(MC[ix],i))
        end
    end
    return r
end

function infos(MC::Array)
    infos(MC,1:MC[1].i)
end

function evals(MC::Array,i::UnitRange{Int})
    if !isa(MC[1],AbstractChain)
        error("must give array of AbstractChain") 
    end
    r = infos(MC[1],i) 
    if length(MC)>1
        for ix=2:length(MC)
            r = vcat(r,evals(MC[ix],i))
        end
    end
    return r
end

function evals(MC::Array)
    evals(MC,1:MC[1].i)
end



function allstats(MC::Array) 
    hcat(infos(MC),parameters(MC)[MC[1].params_nms],moments(MC)[MC[1].moments_nms])
end






# update the iteration count on each chain
function incrementChainIter!(MC::Array)
    for ix in 1:length(MC)
        MC[ix].i += 1
    end 
end


function saveChainToHDF5(chain::AbstractChain, ff5::HDF5File,path::AbstractString)
    simpleDataFrameSave(chain.parameters,ff5,joinpath(path,"parameters"))
    simpleDataFrameSave(chain.infos,ff5, joinpath(path,"infos"))
    simpleDataFrameSave(chain.moments,ff5, joinpath(path,"moments"))
end

function simpleDataFrameSave(dd::DataFrame,ff5::HDF5File, path::AbstractString)
    for nn in names(dd)
        col = dd[nn].data
        if eltype(col) == Bool
            col = convert(Array{Int64},col)
        end
        # if eltype(dd[nn]) <: Number
            write(ff5,joinpath(path,string(nn)),col) 
            # write(ff5,joinpath(path,string(nn)),convert(Array{Float64,1},dd[nn])) 
        # elseif eltype(dd[nn]) <: String
        #     write(ff5,joinpath(path,string(nn)),convert(Array{String,1},dd[nn])) 
        # end
    end
end

function simpleDataFrameRead(ff5::HDF5File, path::AbstractString)
    colnames = names(ff5[path])
    symnames = map(x->Symbol(x),colnames)
    n =  length(colnames)

    columns = Array(Any, n)
    for c in 1:n
        columns[c] = read(ff5[joinpath(path,colnames[c])])
    end
    return DataFrame(columns,symnames)
end




