

export SimpleProblem,
        solve,
        value_of_information

"""
    struct SimpleProblem
        bn::BayesianNetwork
        chance_vars::Vector{Variable}
        decision_vars::Vector{Variable}
        utility_vars::Vector{Variable}
        utilities::Dict{Symbol, Vector{Float64}}
    end

Data structure representing a decision network.

# Fields

    bn::BayesianNetwork

    chance_vars::Vector{Variable}

    decision_vars::Vector{Variable}

    utility_vars::Vector{Variable}

    utilities::Dict{Symbol, Vector{Float64}}

"""
struct SimpleProblem
    bn::BayesianNetwork
    chance_vars::Vector{Variable}
    decision_vars::Vector{Variable}
    utility_vars::Vector{Variable}
    utilities::Dict{Symbol, Vector{Float64}}
end

"""
    function solve(𝒫::SimpleProblem, evidence, M)

Given a decision network `𝒫` and observed variable assignment `evidence`, determine the optimal action variable assignment and associated expected utility, using inference method M.

# Arguments

    𝒫::SimpleProblem
        Decision network
    evidence
        Vector of NamedTuples mapping Symbol name of Variable to assignment of observed values of condition vars.
    M
        Method to use for inference over utility vars given evidence
"""
function solve(𝒫::SimpleProblem, evidence, M)
    query = [var.name for var in 𝒫.utility_vars]
    U(a) = sum(𝒫.utilities[uname][a[uname]] for uname in query)
    best = (a=nothing, u=-Inf)
    for assignment in assignments(𝒫.decision_vars)
        evidence = merge(evidence, assignment)
        φ = infer(M, 𝒫.bn, query, evidence)
        u = sum(p*U(a) for (a, p) in φ.table)
        if u > best.u
            best = (a=assignment, u=u)
        end
    end
    return best
end

"""
    function value_of_information(𝒫, query, evidence, M)

Determine the Value of Information (VOI) of `query` variables, given `evidence` assignment of (specific) values to already observed variables, using inference method `M`.
"""
function value_of_information(𝒫, query, evidence, M)

    φ = infer(M, 𝒫.bn, query, evidence)
    voi = -solve(𝒫, evidence, M).u
    query_vars = filter(v->v.name ∈ query, 𝒫.chance_vars)

    for o′ in assignments(query_vars)
        oo′ = merge(evidence, o′)
        p = φ.table[o′]
        voi += p*solve(𝒫, oo′, M).u
    end

    return voi
end
