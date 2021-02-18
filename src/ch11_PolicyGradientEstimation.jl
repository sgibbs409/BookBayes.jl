

# -------------- CH 11: POLICY GRADIENT ESTIMATION --------------


export  simulate,
        FiniteDifferenceGradient,
        gradient,
        RegressionGradient

"""
    function simulate(𝒫::MDP, s, π, d)

Generate depth-'d' sequence of state-action-reward tuples following policy `π` starting from state `s`.
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
