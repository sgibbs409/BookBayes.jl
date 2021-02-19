

# -------------- CH 11: POLICY GRADIENT ESTIMATION --------------


export  simulate,
        FiniteDifferenceGradient,
        gradient,
        RegressionGradient,
        RewardToGoGradient,
        BaselineSubtractionGradient

"""
    function simulate(𝒫::MDP, s, π, d)

Generate depth-'d' sequence (trajectory `τ`) of state-action-reward tuples following policy `π` starting from state `s`.
"""
function simulate(𝒫::MDP, s, π, d)
    τ = []
    # iterate to depth d
    for i = 1:d
        # follow policy from current state
        a = π(s)
        # simulate dynamics for current state-action
        s′, r = 𝒫.TR(s,a)
        # save trajectory step
        push!(τ, (s,a,r))
        # move to new state (and repeat)
        s = s′
    end
    # return record of state-action-reward tuples
    return τ
end






"""
    struct FiniteDifferenceGradient
        𝒫 # problem
        b # initial state distribution
        d # depth
        m # number of samples
        δ # step size
    end
"""
struct FiniteDifferenceGradient
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
    δ # step size
end


@doc raw"""
    function gradient(M::FiniteDifferenceGradient, π, θ)

Calculate policy gradient for policy `π` at parameterization `θ`, with respect to `θ`.

Returns: $\nabla_{\theta} U(\pi_{\theta})$
"""
function gradient(M::FiniteDifferenceGradient, π, θ)

    𝒫, b, d, m, δ, γ, n = M.𝒫, M.b, M.d, M.m, M.δ, M.𝒫.γ, length(θ)

    # helper function: generate δ⋅(one-hot vector length n, index i-hot)
    Δθ(i) = [i == k ? δ : 0.0 for k in 1:n]

    # get trajectory τ's discounted (depth-d) reward.
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))

    # calculate mean of discounted reward over m rollouts using policy π and parameterization θ, each starting from random state s ∼ b.
    U(θ) = mean(R(simulate(𝒫, rand(b), s->π(θ, s), d)) for i in 1:m)

    # Calculate finite difference of U(θ) for each dimension, using mean of rollouts to estimate policy value function.
    ΔU = [U(θ + Δθ(i)) - U(θ) for i in 1:n]

    return ΔU ./ δ
end


# ----- Regression Gradient ------
#
# Estimate gradient using m > 2n samples of finite differences in random directions.
"""
    struct RegressionGradient
        𝒫 # problem
        b # initial state distribution
        d # depth
        m # number of samples
        δ # step size
    end
"""
struct RegressionGradient
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
    δ # step size
end

"""
    function gradient(M::RegressionGradient, π, θ)

Estimate policy gradient centered at `θ` using `m` random perturbations and then using least-squares over estimated finite-differences of each.

# Example

```julia
julia> using Random

julia> using LinearAlgebra

julia> f(x) = x^2 + 1e-2*randn()

julia> m = 20

julia> δ = 1e-2

julia> ΔX = [δ.*randn() for i = 1:m]

julia> x0 = 2.0

julia> ΔF = [f(x0 + Δx) - f(x0) for Δx in ΔX]

julia> pinv(ΔX) * ΔF       # should be around f′(x==2) == 2×(x==2) == 4
```
"""
function gradient(M::RegressionGradient, π, θ)
    𝒫, b, d, m, δ, γ = M.𝒫, M.b, M.d, M.m, M.δ, M.𝒫.γ
    # m random perturbations of θ from uniform distriubtion over radius δ hypersphere.
    ΔΘ = [δ.*normalize(randn(length(θ)), 2) for i = 1:m]
    # calculate discounted reward from trajectory τ
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    # calculate discounted reward from rollout using policy π and parameterization θ, starting from random state s ∼ b.
    U(θ) = R(simulate(𝒫, rand(b), s->π(θ,s), d))
    # finite differenes
    ΔU = [U(θ + Δθ) - U(θ) for Δθ in ΔΘ]
    # direct solve least-squares solution of gradient using pseudoinverse
    return pinv(reduce(hcat, ΔΘ)') * ΔU
end



"""
    struct LikelihoodRatioGradient
        𝒫 # problem
        b # initial state distribution
        d # depth
        m # number of samples
        ∇logπ # gradient of log likelihood
    end
"""
struct LikelihoodRatioGradient
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ   # gradient of log likelihood
            #  ie: gradient w.r.t. θ of π(θ,a,s),
            #   where π(θ,a,s) = probability(a | s; θ)
end

"""
    function gradient(M::LikelihoodRatioGradient, π, θ)


"""
function gradient(M::LikelihoodRatioGradient, π, θ)
    𝒫, b, d, m, ∇logπ, γ = M.𝒫, M.b, M.d, M.m, M.∇logπ, M.𝒫.γ

    # πθ: stochastic/non-deterministic function of s
    πθ(s) = π(θ, s)

    # Calculate total discounted reward from trajectory τ
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))

    # ∇₀log[p₀(τ)] × R(τ) = ∑ₖ₌₁(∇₀log[π(θ,a⁽ᵏ⁾,s⁽ᵏ⁾)]) × R(τ)
    ∇U(τ) = sum(∇logπ(θ, a, s) for (s,a) in τ)*R(τ)

    # monte carlo estimate of ∇₀U(θ) = 𝔼ₜ[∇₀log(p₀(τ)R(τ))] ∼ τ
    return mean(∇U(simulate(𝒫, rand(b), πθ, d)) for i in 1:m)
end



"""
    struct RewardToGoGradient
        𝒫 # problem
        b # initial state distribution
        d # depth
        m # number of samples
        ∇logπ # gradient of log likelihood
    end
"""
struct RewardToGoGradient
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood
end


"""
    function gradient(M::RewardToGoGradient, π, θ)


"""
function gradient(M::RewardToGoGradient, π, θ)
    𝒫, b, d, m, ∇logπ, γ = M.𝒫, M.b, M.d, M.m, M.∇logπ, M.𝒫.γ
    πθ(s) = π(θ, s)

    # see eq. 11.24: == γᵏ⁻¹ × r_to-go⁽ᵏ⁾
    R(τ, k) = sum(r*γ^(l-1) for (l,(s,a,r)) in zip(k:d, τ[k:end]))
    # see eq. 11.24
    ∇U(τ) = sum(∇logπ(θ, a, s)*R(τ,k) for (k, (s,a,r)) in enumerate(τ))

    # monte carlo approximation to 𝔼xpectation over τ
    return mean(∇U(simulate(𝒫, rand(b), πθ, d)) for i in 1:m)
end




"""
    struct BaselineSubtractionGradient
        𝒫 # problem
        b # initial state distribution
        d # depth
        m # number of samples
        ∇logπ # gradient of log likelihood
    end
"""
struct BaselineSubtractionGradient
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood
end


"""
    function gradient(M::BaselineSubtractionGradient, π, θ)


"""
function gradient(M::BaselineSubtractionGradient, π, θ)
    𝒫, b, d, m, ∇logπ, γ = M.𝒫, M.b, M.d, M.m, M.∇logπ, M.𝒫.γ
    πθ(s) = π(θ, s)

    l(a, s, k) = ∇logπ(θ, a, s)*γ^(k-1)
    # r_to-go
    R(τ, k) = sum(r*γ^(j-1) for (j,(s,a,r)) in enumerate(τ[k:end]))

    numer(τ) = sum(l(a,s,k).^2*R(τ,k) for (k,(s,a,r)) in enumerate(τ))

    denom(τ) = sum(l(a,s,k).^2 for (k,(s,a)) in enumerate(τ))
    # 11.43
    base(τ) = numer(τ) ./ denom(τ)

    trajs = [simulate(𝒫, rand(b), πθ, d) for i in 1:m]
    # 11.43
    rbase = mean(base(τ) for τ in trajs)
    # 11.28
    ∇U(τ) = sum(l(a,s,k).*(R(τ,k).-rbase) for (k,(s,a,r)) in enumerate(τ))
    # 11.28
    return mean(∇U(τ) for τ in trajs)
end
