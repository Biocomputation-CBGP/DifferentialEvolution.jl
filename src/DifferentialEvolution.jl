module DifferentialEvolution

using StatsBase

struct EvolutionThreads end
export EvolutionThreads

struct DiffEv{T<:Number, F<:Function}
    population::Matrix{T}
    losses::Vector{T}
    basefunc::F
    NP::Int
    n::Int
    F::T
    CR::T
end
export DiffEv

function DiffEv(NP, n, F::T, CR::T; strategy=:rand) where {T<:Number}
    population = rand(T, NP, n)
    losses = fill(typemax(T), NP)
    if strategy == :rand
        return DiffEv(population, losses, randbase, NP, n, F, CR)
    elseif strategy == :best
        return DiffEv(population, losses, bestbase, NP, n, F, CR)
    end
end

"""Pick a random base vector, plus a pair for the difference vector"""
@views function randbase(alg::DiffEv, i::Int)
    idxs = sample(deleteat!(collect(1:alg.NP), i), 3; replace=false)
    return alg.population[idxs, :]
end

"""Pick the best base vector, plus a pair for the difference vector"""
@views function bestbase(alg::DiffEv, i::Int)
    _, j = findmin(alg.losses)
    idxs = i == j ? [i] : sort([i, j])
    idxs = sample(deleteat!(collect(1:alg.NP), idxs), 2; replace=false)
    return alg.population[[j; idxs], :]
end

"""Generate a mutated vector from the base and difference vectors"""
@views function mutated(vectors, F)
    return vectors[1, :] .+ F .* (vectors[2, :] .- vectors[3, :])
end

"""Generate a proposal vector from a parent and mutated vector"""
@views function crossover(parent, mutated, CR)
    proposal = copy(parent)
    R = rand(1:length(parent))
    proposal[R] = mutated[R]
    r = rand(length(parent))
    proposal[r .< CR] .= mutated[r .< CR]
    return proposal
end


function evolve!(alg::DiffEv, f::F, i::Int) where {F<:Function}
    best = alg.population[i, :]
    mutant = mutated(alg.basefunc(alg, i), alg.F)
    proposal = crossover(best, mutant, alg.CR)
    alg.losses[i] = f(best) # this line is an extra function eval, but might be important for stochastic f
    loss = f(proposal)
    if loss <= alg.losses[i]
        alg.losses[i] = loss
        best = mutant
    end
    return best
end

function evolve!(alg::DiffEv, f::F) where {F<:Function}
    next_generation = copy(alg.population)
    for i in 1:alg.NP
        next_generation[i, :] .= evolve!(alg, f, i)
    end
    alg.population .= next_generation
    return alg.losses
end

function evolve!(alg::DiffEv, f::F, ::Type{EvolutionThreads}) where {F<:Function}
    next_generation = copy(alg.population)
    Threads.@threads for i in 1:alg.NP
        next_generation[i, :] .= evolve!(alg, f, i)
    end
    alg.population .= next_generation
    return alg.losses
end

export evolve!

end # module DifferentialEvolution
