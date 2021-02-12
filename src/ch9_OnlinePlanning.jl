

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
struct RolloutLookahead
    𝒫 # problem
    π # rollout policy
    d # depth
end

randstep(𝒫::MDP, s, a) = 𝒫.TR(s, a)

"""
    function rollout(𝒫, s, π, d)
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
            u += (r + 𝒫.γ*u′) / m
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
"""
function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    return _argmax(a->π.Q[(s,a)], π.𝒫.𝒜)
end


"""
    function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
"""
function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return 0.0
    end

    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, TR, γ = 𝒫.𝒜, 𝒫.TR, 𝒫.γ

    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return rollout(𝒫, s, π.π, d)
    end

    a = explore(π, s)
    s′, r = TR(s,a)
    q = r + γ*simulate!(π, s′, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]

    return q
end

"""
    bonus(Nsa, Ns)
"""
bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)


"""
    function explore(π::MonteCarloTreeSearch, s)
"""
function explore(π::MonteCarloTreeSearch, s)

    𝒜, N, Q, c = π.𝒫.𝒜, π.N, π.Q, π.c

    Ns = sum(N[(s,a)] for a in 𝒜)

    return _argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end


# --------- Heuristic Search ---------

struct HeuristicSearch
    𝒫 # problem
    Uhi # upper bound on value function
    d # depth
    m # number of simulations
end

"""
    function simulate!(π::HeuristicSearch, U, s)
"""
function simulate!(π::HeuristicSearch, U, s)
    𝒫, d = π.𝒫, π.d
    for d in 1:d
        a, u = greedy(𝒫, U, s)
        U[s] = u
        s = rand(𝒫.T(s, a))
    end
end

"""
    function (π::HeuristicSearch)(s)
"""
function (π::HeuristicSearch)(s)
    U = [π.Uhi(s) for s in π.𝒫.𝒮]
    for i in 1:m
        simulate!(π, U, s)
    end
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
