


export  MonteCarloPolicyEvaluation,
        HookeJeevesPolicySearch,
        optimize,
        GeneticPolicySearch,
        CrossEntropyPolicySearch,
        optimize_dist,
        EvolutionStrategies,
        evolution_strategy_weights,
        IsotropicEvolutionStrategies


# --------------------------- POLICY SEARCH ---------------------------

# 1.  Monte Carlo Policy Evaluation
#
# Return the mean discounted reward of m random trajectories,
# each calculated by first sampling the initial state distribution,
# and then executing rollout from that state, using policy π.

"""
    struct MonteCarloPolicyEvaluation
        𝒫 # problem
        b # initial state distribution
        d # depth
        m # number of samples
    end
"""
struct MonteCarloPolicyEvaluation
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
end

@doc raw"""
    function (U::MonteCarloPolicyEvaluation)(π)

Return the mean discounted reward of m random trajectories, each calculated by first sampling the initial state distribution, and then executing rollout from that state, using policy π.

$U(\pi) = E_{\tau}[R(\tau)] = \sum_{\tau} p(\tau)R(\tau) \approx \frac{1}{m} \sum_{i=1}^{m} R(\tau^{(i)})$

Returns: Ũ(π)

Complexity: 𝒪(m × d)
"""
function (U::MonteCarloPolicyEvaluation)(π)
    R(π) = rollout(U.𝒫, rand(U.b), π, U.d) # 𝒪(d)
    return mean(R(π) for i = 1:U.m)
end


"""
    (U::MonteCarloPolicyEvaluation)(π, θ) = U(s->π(θ, s))

Parameterized version where policy adjusted by parameter vector `θ`.

Returns: Ũ(π₍θ₎)

Complexity: 𝒪(m × d)
"""
(U::MonteCarloPolicyEvaluation)(π, θ) = U(s->π(θ, s))



# 2.  Local Search (Hooke-Jeeves)
#
# Start w/initial parameterization and move from neighbor to neighbor
#   until convergence.
# θ: n-dimensional vector
# Takes step of ±α in each coordinate direction from θ.  If improvement found, moves to best neighbor.  Else stepsize reduced to α *= c for some c < 1.0.  Continue until α < ϵ for some ϵ > 0.

"""
    struct HookeJeevesPolicySearch
        θ # initial parameterization
        α # step size
        c # step size reduction factor
        ε # termination step size
    end
"""
struct HookeJeevesPolicySearch
    θ # initial parameterization
    α # step size
    c # step size reduction factor
    ε # termination step size
end

"""
    function optimize(M::HookeJeevesPolicySearch, π, U)
"""
function optimize(M::HookeJeevesPolicySearch, π, U)
    θ, θ′, α, c, ε = copy(M.θ), similar(M.θ), M.α, M.c, M.ε
    u, n = U(π, θ), length(θ)
    while α > ε
        copyto!(θ′, θ)
        best = (i=0, sgn=0, u=u)
        for i in 1:n
            for sgn in (-1,1)
                θ′[i] = θ[i] + sgn*α
                u′ = U(π, θ′)
                if u′ > best.u
                    best = (i=i, sgn=sgn, u=u′)
                end
            end
            θ′[i] = θ[i]
        end
        if best.i != 0
            θ[best.i] += best.sgn*α
            u = best.u
        else
            α *= c
        end
    end
    return θ
end


# 3.  Genetic Policy Search

"""
    struct GeneticPolicySearch
        θs # initial population
        σ # initial standard devidation
        m_elite # number of elite samples
        k_max # number of iterations
    end
"""
struct GeneticPolicySearch
    θs # initial population
    σ # initial standard devidation
    m_elite # number of elite samples
    k_max # number of iterations
end

"""
    function optimize(M::GeneticPolicySearch, π, U)
"""
function optimize(M::GeneticPolicySearch, π, U)
    θs, σ = M.θs, M.σ
    n, m = length(first(θs)), length(θs)
    for k in 1:M.k_max
        us = [U(π, θ) for θ in θs]
        sp = sortperm(us, rev=true)
        θ_best = θs[sp[1]]
        rand_elite() = θs[sp[rand(1:M.m_elite)]]
        θs = [rand_elite() + σ.*randn(n) for i in 1:(m-1)]
        push!(θs, θ_best)
    end
    return last(θs)
end


# 4.  Cross Entropy Method

"""
    struct CrossEntropyPolicySearch
        p # initial distribution
        m # number of samples
        m_elite # number of elite samples
        k_max # number of iterations
    end
"""
struct CrossEntropyPolicySearch
    p # initial distribution
    m # number of samples
    m_elite # number of elite samples
    k_max # number of iterations
end

"""
    function optimize_dist(M::CrossEntropyPolicySearch, π, U)
"""
function optimize_dist(M::CrossEntropyPolicySearch, π, U)
    p, m, m_elite, k_max = M.p, M.m, M.m_elite, M.k_max
    for k in 1:k_max
        θs = rand(p, m)
        us = [U(π, θs[:,i]) for i in 1:m]
        θ_elite = θs[:,sortperm(us)[(m-m_elite+1):m]]
        p = Distributions.fit(typeof(p), θ_elite)
    end
    return p
end

"""
    function optimize(M, π, U)
"""
function optimize(M, π, U)
    return Distributions.mode(optimize_dist(M, π, U))
end



# 5.  Evolutionary Strategies
#
# Update a search distribution parameterized by a vector ψ at
#   each iteration by taking a step in the direciton of the gradient of
#   the Expected Policy Utility function.
"""
    struct EvolutionStrategies
        D # distribution constructor
        ψ # initial distribution parameterization
        ∇logp # log search likelihood gradient
        m # number of samples
        α # step factor
        k_max # number of iterations
    end
"""
struct EvolutionStrategies
    D # distribution constructor
    ψ # initial distribution parameterization
    ∇logp # log search likelihood gradient
    m # number of samples
    α # step factor
    k_max # number of iterations
end

"""
    function evolution_strategy_weights(m)
"""
function evolution_strategy_weights(m)
    ws = [max(0, log(m/2+1) - log(i)) for i in 1:m]
    ws ./= sum(ws)
    ws .-= 1/m

    return ws
end

"""
    function optimize_dist(M::EvolutionStrategies, π, U)
"""
function optimize_dist(M::EvolutionStrategies, π, U)
    D, ψ, m, ∇logp, α = M.D, M.ψ, M.m, M.∇logp, M.α
    ws = evolution_strategy_weights(m)
    for k in 1:M.k_max
        θs = rand(D(ψ), m)
        us = [U(π, θs[:,i]) for i in 1:m]
        sp = sortperm(us, rev=true)
        ∇ = sum(w.*∇logp(ψ, θs[:,i]) for (w,i) in zip(ws,sp))
        ψ += α.*∇
    end
    return D(ψ)
end


# 6.  Isotropic Evolutionary Strategies
#
# Same as Evolutionary Strategies, but use mirroring of samples to reduce
#   gradient variation.
"""
    struct IsotropicEvolutionStrategies
        ψ # initial mean
        σ # initial standard devidation
        m # number of samples
        α # step factor
        k_max # number of iterations
    end
"""
struct IsotropicEvolutionStrategies
    ψ # initial mean
    σ # initial standard devidation
    m # number of samples
    α # step factor
    k_max # number of iterations
end

"""
    function optimize_dist(M::IsotropicEvolutionStrategies, π, U)
"""
function optimize_dist(M::IsotropicEvolutionStrategies, π, U)
    ψ, σ, m, α, k_max = M.ψ, M.σ, M.m, M.α, M.k_max
    n = length(ψ)
    ws = evolution_strategy_weights(2*div(m,2))
    for k in 1:k_max
        εs = [randn(n) for i in 1:div(m,2)]
        append!(εs, -εs) # weight mirroring
        us = [U(π, ψ + σ.*ε) for ε in εs]
        sp = sortperm(us, rev=true)
        ∇ = sum(w.*εs[i] for (w,i) in zip(ws,sp)) / σ
        ψ += α.*∇
    end
    return MvNormal(ψ, σ)
end
