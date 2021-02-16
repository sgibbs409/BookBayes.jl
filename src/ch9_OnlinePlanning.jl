
# ------------------- Ch. 9: Online Planning --------------------

# Reachable state space usually much smaller than than full state space.
#
# Algorithms to compute approximately optimal policies starting
#   from the current state.


export  RolloutLookahead,
        randstep,
        rollout,
        ForwardSearch,
        forward_search,
        BranchAndBound,
        branch_and_bound,
        SparseSampling,
        sparse_sampling,
        MonteCarloTreeSearch,
        simulate!,
        bonus,
        explore,
        HeuristicSearch,
        LabeledHeuristicSearch,
        expand,
        label!




# -------- Rollout with Lookahead ---------

"""
    struct RolloutLookahead
        𝒫 # problem
        π # rollout policy
        d # depth
    end
"""
struct RolloutLookahead
    𝒫 # problem
    π # rollout policy
    d # depth
end

"""
    randstep(𝒫::MDP, s, a) = 𝒫.TR(s, a)
"""
randstep(𝒫::MDP, s, a) = 𝒫.TR(s, a)

"""
    function rollout(𝒫, s, π, d)

Complexity: 𝒪(d)
"""
function rollout(𝒫, s, π, d)
    if d ≤ 0
        return 0.0
    end
    a = π(s)
    s′, r = randstep(𝒫, s, a)
    return r + 𝒫.γ*rollout(𝒫, s′, π, d-1)
end


"""
    function (π::RolloutLookahead)(s)

Policy function: similar to ValueFunctionPolicy but using a single call to rollout for value function.

Complexity: O(|𝒜|×|𝒮| × d)

Consider variation that uses average of m rollouts to calculate U(s) (vs just 1 with this version).
"""
function (π::RolloutLookahead)(s)
    U(s) = rollout(π.𝒫, s, π.π, π.d)
    return greedy(π.𝒫, U, s).a
end


# -------- Forward Search ---------
"""
    struct ForwardSearch
        𝒫 # problem
        d # depth
        U # value function at depth d
    end
"""
struct ForwardSearch
    𝒫 # problem
    d # depth
    U # value function at depth d
end

"""
    function forward_search(𝒫, s, d, U)

Determine optimal action (and its value) to take from state `s` by expanding all possible transitions up to a depth `d` using depth-first search. `U(s)` used to evaluate the terminal (depth 0) value function.

Complexity: 𝒪( (|𝒜|×|𝒮|)ᵈ )
"""
function forward_search(𝒫, s, d, U)
    if d ≤ 0
        return (a=nothing, u=U(s))
    end

    best = (a=nothing, u=-Inf)
    U′(s) = forward_search(𝒫, s, d-1, U).u

    for a in 𝒫.𝒜
        u = lookahead(𝒫, U′, s, a)
        if u > best.u
            best = (a=a, u=u)
        end
    end

    return best
end

(π::ForwardSearch)(s) = forward_search(π.𝒫, s, π.d, π.U).a

# --------- Branch and Bound ---------

"""
    struct BranchAndBound
        𝒫 # problem
        d # depth
        Ulo # lower bound on value function at depth d Qhi # upper bound on action value function
    end
"""
struct BranchAndBound
    𝒫 # problem
    d # depth
    Ulo # lower bound on value function at depth d
    Qhi # upper bound on action value function
end


"""
    function branch_and_bound(𝒫, s, d, Ulo, Qhi)
"""
function branch_and_bound(𝒫, s, d, Ulo, Qhi)
    if d ≤ 0
        return (a=nothing, u=Ulo(s))
    end

    U′(s) = branch_and_bound(𝒫, s, d-1, Ulo, Qhi).u
    best = (a=nothing, u=-Inf)
    for a in sort(𝒫.𝒜, by=a->Qhi(s,a), rev=true)
        if Qhi(s, a) < best.u
            return best # safe to prune
        end
        u = lookahead(𝒫, U′, s, a)
        if u > best.u
            best = (a=a, u=u)
        end
    end
    return best
end

(π::BranchAndBound)(s) = branch_and_bound(π.𝒫, s, π.d, π.Ulo, π.Qhi).a




# --------- Sparse Sampling ---------

"""
    struct SparseSampling
        𝒫 # problem
        d # depth
        m # number of samples
        U # value function at depth d
    end
"""
struct SparseSampling
    𝒫 # problem
    d # depth
    m # number of samples
    U # value function at depth d
end


"""
    function sparse_sampling(𝒫, s, d, m, U)

Complexity: 𝒪( (|𝒜| × m)ᵈ )
"""
function sparse_sampling(𝒫, s, d, m, U)
    if d ≤ 0
        return (a=nothing, u=U(s))
    end

    best = (a=nothing, u=-Inf)

    for a in 𝒫.𝒜
        u = 0.0
        for i in 1:m
            s′, r = randstep(𝒫, s, a)
            a′, u′ = sparse_sampling(𝒫, s′, d-1, m, U)
            u += (r + 𝒫.γ*u′) / m  # update mean
        end

        if u > best.u
            best = (a=a, u=u)
        end
    end

    return best
end

(π::SparseSampling)(s) = sparse_sampling(π.𝒫, s, π.d, π.m, π.U).a




# --------- Monte Carlo Tree Search ---------

"""
    struct MonteCarloTreeSearch
        𝒫 # problem
        N # visit counts
        Q # action value estimates
        d # depth
        m # number of simulations
        c # exploration constant
        π # rollout policy
    end
"""
struct MonteCarloTreeSearch
    𝒫 # problem
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    π # rollout policy
end


"""
    function (π::MonteCarloTreeSearch)(s)

Estimate optimal next action by first simulating m random trajectories.  Each simulation improves estimated action-value function.  The returned action is the optimal action over the estimated action-value function from the given state `s`.

Complexity: 𝒪(|𝒜| × d × m)
"""
function (π::MonteCarloTreeSearch)(s)
    # simulate m times (m trajectories)
    for k in 1:π.m
        simulate!(π, s)
    end
    # Return action with highest action-value
    return _argmax(a->π.Q[(s,a)], π.𝒫.𝒜)
end


"""
    function simulate!(π::MonteCarloTreeSearch, s, d=π.d)

Execute exploration-bonus policy for 1 random trajectory, updating system  counters, and return sample trajectory's value.

Complexity: `𝒪(|𝒜| × d)`
"""
function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return 0.0
    end

    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, TR, γ = 𝒫.𝒜, 𝒫.TR, 𝒫.γ

    # if never visited state s, init visit count and action-value estimate.
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        # first time to state s: return sample trajectory value as utility est.
        return rollout(𝒫, s, π.π, d)
    end

    # find next action to try
    a = explore(π, s) # 𝒪(|𝒜|)
    # simulate 1 step
    s′, r = TR(s,a)
    # recurse remaining stochastic trajectory
    q = r + γ*simulate!(π, s′, d-1)
    # update visit count
    N[(s,a)] += 1
    # update running average of action-value
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]

    return q
end


"""
    bonus(Nsa, Ns)

Monte Carlo exploration bonus term helper function.

Complexity: 𝒪(1)
"""
bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)


"""
    function explore(π::MonteCarloTreeSearch, s)

Use monte carlo exploration heuristic to find next action to simulate.  Balances need to explore state-action space with
Complexity: 𝒪(|𝒜|)
"""
function explore(π::MonteCarloTreeSearch, s)

    𝒜, N, Q, c = π.𝒫.𝒜, π.N, π.Q, π.c

    Ns = sum(N[(s,a)] for a in 𝒜)

    return _argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end


# --------- Heuristic Search ---------

# Use m simulations of a greedy policy with respect to value function U from state s.  U is initialized to an upperbound Ū (referred to as a heuristic).  Updates U with each lookahead step during simulation.  After simulations and value function estimate U improvement, return greedy action.
"""
    struct HeuristicSearch
        𝒫 # problem
        Uhi # upper bound on value function
        d # depth
        m # number of simulations
    end
"""
struct HeuristicSearch
    𝒫 # problem
    Uhi # upper bound on value function
    d # depth
    m # number of simulations
end


"""
    function simulate!(π::HeuristicSearch, U, s)

Simulate depth-d trajectory following greedy policy and updating U along the way.

Complexity: 𝒪(d × |𝒜|×|𝒮|)
"""
function simulate!(π::HeuristicSearch, U, s)
    𝒫, d = π.𝒫, π.d
    for d in 1:d                # 𝒪( d × ... (|𝒜|×|𝒮|))
        a, u = greedy(𝒫, U, s) # 𝒪(|𝒜|×|𝒮|)
        U[s] = u
        s = rand(𝒫.T(s, a))
    end
end


"""
    function (π::HeuristicSearch)(s)

Guranteed to converge to optimal value function iff `Uhi` is indeed an upperbound on `U`.

Complexity: 𝒪(m × d × |𝒮|×|𝒜|)
"""
function (π::HeuristicSearch)(s)
    # initialize U with upper bound
    U = [π.Uhi(s) for s in π.𝒫.𝒮]
    # run m random trajectory simulations to improve estimate on U.
    for i in 1:m
        simulate!(π, U, s)
    end
    # Return greedy action using improved U.
    return greedy(π.𝒫, U, s).a
end

# --------- Labled Heuristic Search ---------


"""
    struct LabeledHeuristicSearch
        𝒫 # problem
        Uhi # upper bound on value function
        d # depth
        δ # gap threshold
    end
"""
struct LabeledHeuristicSearch
    𝒫 # problem
    Uhi # upper bound on value function
    d # depth
    δ # gap threshold
end


"""
    function (π::LabeledHeuristicSearch)(s)
"""
function (π::LabeledHeuristicSearch)(s)
    U, solved = [π.Uhi(s) for s in 𝒫.𝒮], Set()
    while s ∉ solved
        simulate!(π, U, solved, s)
    end
    return greedy(π.𝒫, U, s).a
end


"""
    function simulate!(π::LabeledHeuristicSearch, U, solved, s)
"""
function simulate!(π::LabeledHeuristicSearch, U, solved, s)
    visited = []
    for d in 1:π.d
        if s ∈ solved
            break
        end

        push!(visited, s)
        a, u = greedy(π.𝒫, U, s)
        U[s] = u
        s = rand(π.𝒫.T(s, a))
    end

    while !isempty(visited)
        if label!(π, U, solved, pop!(visited))
            break
        end
    end
end


"""
    function expand(π::LabeledHeuristicSearch, U, solved, s)
"""
function expand(π::LabeledHeuristicSearch, U, solved, s)
    𝒫, δ = π.𝒫, π.δ
    𝒮, 𝒜, T = 𝒫.𝒮, 𝒫.𝒜, 𝒫.T

    found, toexpand, envelope = false, Set(s), []
    while !isempty(toexpand)
        s = pop!(toexpand)
        push!(envelope, s)
        a, u = greedy(𝒫, U, s)
        if abs(U[s] - u) > δ
            found = true
        else
            for s′ in 𝒮
                if T(s,a,s′) > 0 && s′ ∉ (solved ∪ envelope)
                    push!(toexpand, s′)
                end
            end
        end
    end

    return (found, envelope)
end


"""
    function label!(π::LabeledHeuristicSearch, U, solved, s)
"""
function label!(π::LabeledHeuristicSearch, U, solved, s)
    if s ∈ solved
        return false
    end
    found, envelope = expand(π, U, solved, s)
    if found
        for s ∈ reverse(envelope)
            U[s] = greedy(π.𝒫, U, s).u
        end
    else
        union!(solved, envelope)
    end

    return found
end
