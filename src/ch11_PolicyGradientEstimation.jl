

# -------------- CH 11: POLICY GRADIENT ESTIMATION --------------


export  simulate,
        FiniteDifferenceGradient,
        gradient

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

Returns: $\del_{\theta} U(\pi_{\theta})$
"""
function gradient(M::FiniteDifferenceGradient, π, θ)

    𝒫, b, d, m, δ, γ, n = M.𝒫, M.b, M.d, M.m, M.δ, M.𝒫.γ, length(θ)

    Δθ(i) = [i == k ? δ : 0.0 for k in 1:n]

    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))

    U(θ) = mean(R(simulate(𝒫, rand(b), s->π(θ, s), d)) for i in 1:m)

    ΔU = [U(θ + Δθ(i)) - U(θ) for i in 1:n]

    return ΔU ./ δ
end
