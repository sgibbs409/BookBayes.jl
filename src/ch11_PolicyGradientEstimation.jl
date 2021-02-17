

# -------------- CH 11: POLICY GRADIENT ESTIMATION --------------


export  simulate

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
