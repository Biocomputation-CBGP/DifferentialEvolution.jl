using Test
using DifferentialEvolution

@testset "Rosenbrock test" begin
    function rosenbrock(p::AbstractVector{T}) where {T<:Real}
        x, y = p
        return (1 - x)^2 + 100 * (y - x^2)^2
    end

    DE = DiffEv(16, 2, 0.8, 0.9)
    for _ in 1:128
        DE(rosenbrock)
    end

    @test all(x -> isapprox(x, 0; atol=1e-3), DE.fitnesses)
    @test all(x -> isapprox(x, 1; atol=1e-3), DE.population)
end

@testset "Ackley test" begin
    function ackley(p::AbstractVector{T}) where {T<:Real}
        x, y = p
        return (
            -20 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2)))
            - exp(0.5 * (cos(2 * π * x) + cos(2 * π * y)))
            + exp(1) + 20
        )
    end

    DE = DiffEv(16, 2, 0.8, 0.9)
    for _ in 1:128
        DE(ackley)
    end

    @test all(x -> isapprox(x, 0; atol=1e-3), DE.fitnesses)
    @test all(x -> isapprox(x, 0; atol=1e-3), DE.population)
end
