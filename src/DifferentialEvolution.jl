module DifferentialEvolution

using StatsBase

"""Pick a random base vector, plus a pair for the difference vector"""
@views function randbase(population, i, NP)
    idxs = sample(vcat(1:i-1, i+1:NP), 3; replace=false)
    return population[idxs, :]
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
    proposal[r .< CR] = mutated[r .< CR]
    return proposal
end

struct EvolutionThreads end

struct DiffEv{T<:Number}
    population::Matrix{T}
    fitnesses::Vector{T}
    NP::Int
    n::Int
    F::T
    CR::T
end
export DiffEv

function DiffEv(NP, n, F::T, CR::T) where {T<:Number}
    population = rand(T, NP, n)
    fitnesses = fill(typemax(T), NP)
    DiffEv{T}(population, fitnesses, NP, n, F, CR)
end

function (X::DiffEv{T})(f) where {T<:Number}
    for x in 1:X.NP
        parent = @view X.population[x, :]
        mutant = mutated(randbase(X.population, x, X.NP), X.F)
        proposal = crossover(parent, mutant, X.CR)

        fitness = f(proposal)
        if fitness <= X.fitnesses[x]
            X.population[x, :] .= proposal
            X.fitnesses[x] = fitness
        end
    end
    return X.fitnesses
end

function (X::DiffEv{T})(f, ::Type{EvolutionThreads}) where {T<:Number}
    next_generation = copy(X.population)
    Threads.@threads for x in 1:X.NP
        parent = @view X.population[x, :]
        mutant = mutated(randbase(X.population, x, X.NP), X.F)
        proposal = crossover(parent, mutant, X.CR)
        
        fitness = f(proposal)
        if fitness <= X.fitnesses[x]
            next_generation[x, :] .= proposal
            X.fitnesses[x] = fitness
        end
    end
    X.population .= next_generation
    return X.fitnesses
end

end # module DifferentialEvolution
