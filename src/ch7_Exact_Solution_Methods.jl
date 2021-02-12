



export  MDF,
        lookahead,
        iterative_policy_evaluation,
        policy_evaluation,
        ValueFunctionPolicy,
        greedy,
        PolicyIteration,
        solve,
        backup,
        ValueIteration,
        GaussSeidelValueIteration,
        LinearProgramFormulation,
        tensorform,
        LinearQuadraticProblem


"""
    struct MDP
        γ # discount factor
        𝒮 # state space
        𝒜 # action space
        T # transition function
        R # reward function
        TR # sample transition and reward
    end
"""
struct MDP
    γ # discount factor
    𝒮 # state space
    𝒜 # action space
    T # transition function
    R # reward function
    TR # sample transition and reward
end

"""
    function lookahead(𝒫::MDP, U, s, a)

Calculate state-action value function Q(s,a) from U and valid state transition distribution T:

`Returns`
    Q(s,a) = R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)

"""
function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
end

"""
    function lookahead(𝒫::MDP, U::Vector, s, a)

Calculate state-action value function Q(s,a) from U and valid state transition distribution T:

    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))

"""
function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
end



"""
    function iterative_policy_evaluation(𝒫::MDP, π, k_max)

 Approximate policy evaluation for policy `π`, using k_max iterations of 1-step lookahead, starting from U(s) = 0.
"""
function iterative_policy_evaluation(𝒫::MDP, π, k_max)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    U = [0.0 for s in 𝒮]
    for k in 1:k_max
        U = [lookahead(𝒫, U, s, π(s)) for s in 𝒮]
    end
    return U
end



"""
    function policy_evaluation(𝒫::MDP, π)

Exact policy evaluation using matrix-based Moore-Penrose pseudoinverse equation.
Requires O(|𝒮|³) time.

`Returns:` [ U(s) ∀ s ∈ {𝒮} ]
"""
function policy_evaluation(𝒫::MDP, π)
    𝒮, R, T, γ = 𝒫.𝒮, 𝒫.R, 𝒫.T, 𝒫.γ

    # R(s, π(s)) ∀ s ∈ {𝒮}
    R′ = [R(s, π(s)) for s in 𝒮] # nx1 = |𝒮|x1
    T′ = [T(s, π(s), s′) for s in 𝒮, s′ in 𝒮] # nxm = |𝒮|x|𝒮|
    return (I - γ*T′)\R′
end



"""
    struct ValueFunctionPolicy
        𝒫 # problem
        U # utility function
    end
"""
struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

"""
    function greedy(𝒫::MDP, U, s)

Find the greedy action and its expected utility starting from state `s` and using estimated optimal (greedy) state value function `U`.  Calculates estimate as maximum over action space of 1-step lookahead:

    u, a = _findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
"""
function greedy(𝒫::MDP, U, s)
    u, a = _findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end

"""
    function (π::ValueFunctionPolicy)(s)

Greedy policy function mapping states to greedy action.
"""
function (π::ValueFunctionPolicy)(s)
    return greedy(π.𝒫, π.U, s).a
end


# ---------------- Policy Iteration ----------------

"""
    struct PolicyIteration
        π # initial policy
        k_max # maximum number of iterations
    end
"""
struct PolicyIteration
    π # initial policy
    k_max # maximum number of iterations
end

"""
    function solve(M::PolicyIteration, 𝒫::MDP)

Iteratively improves an initial policy `π` to obtain an optimal policy for an MDP `𝒫` with discrete state and action spaces.  Returns improved policy.
"""
function solve(M::PolicyIteration, 𝒫::MDP)
    π, 𝒮 = M.π, 𝒫.𝒮
    converged = false

    for k = 1:M.k_max
        # exact solution for U_π(s)
        U = policy_evaluation(𝒫, π)
        # Use new utility function for finding greedy action.
        π′ = ValueFunctionPolicy(𝒫, U)
        # same action given for every state, => convergence (current π is best)
        if all(π(s) == π′(s) for s in 𝒮)
            break
        end
        # Not converged: update policy with new greedy policy.
        π = π′
    end
    return π
end



# -------------------------- Value Iteration --------------------------

"""
    function backup(𝒫::MDP, U, s)

Calculate maxₐ Q(s,a).
Complexity: O(|𝒮|×|𝒜|).

    return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
"""
function backup(𝒫::MDP, U, s)
    return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
end


"""
    struct ValueIteration
        k_max # maximum number of iterations
    end
"""
struct ValueIteration
    k_max # maximum number of iterations
end

"""
    function solve(M::ValueIteration, 𝒫::MDP)

Iteratively improves a value function `U` to obtain an optimal policy for an MDP `𝒫` with discrete state and action spaces.  Terminates after `k_max` iterations.

Order O(k_max × |𝒮| × |𝒜| × |𝒮|)
# Returns

Optimal policy `π`
"""
function solve(M::ValueIteration, 𝒫::MDP)
    U = [0.0 for s in 𝒫.𝒮]
    for k = 1:M.k_max
        U = [backup(𝒫, U, s) for s in 𝒫.𝒮]
    end
    return ValueFunctionPolicy(𝒫, U)
end

#
# @doc raw"""
#     function solve(M::ValueIteration, 𝒫::MDP, δ::Float64)
#
# Iteratively improves a value function `U` to obtain an optimal policy for an MDP `𝒫` with discrete state and action spaces.  Terminates after Bellman residual falls below threshold δ:
#
# $\|\|U\_{k+1} - U\_k\|\|\_{\infty} < \delta$
#
# # Returns
#
# Optimal policy `π`
# """
# function solve(M::ValueIteration, 𝒫::MDP, δ::Float64)
#     U = [0.0 for s in 𝒫.𝒮]
#     for k = 1:M.k_max
#         U = [backup(𝒫, U, s) for s in 𝒫.𝒮]
#     end
#     return ValueFunctionPolicy(𝒫, U)
# end



# ----------------- Asyncronous Value Iteration --------------------
#
# Similar to Value Iteration, but only subset of states updated every iteration.
#
# Common asyncronous value iteration method: Gauss-Seidel.
#   Sweeps through an ordering of states and applies Bellman update in place.
"""
    struct GaussSeidelValueIteration
        k_max # maximum number of iterations
    end
"""
struct GaussSeidelValueIteration
    k_max # maximum number of iterations
end

"""
    function solve(M::GaussSeidelValueIteration, 𝒫::MDP)

Same as Value Iteration but update U one state at a time.  Convergence speed depends on ordering of states in `𝒮`.
"""
function solve(M::GaussSeidelValueIteration, 𝒫::MDP)
    𝒮, 𝒜, T, R, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ
    U = [0.0 for s in 𝒮]
    for k = 1:M.k_max
        for (i, s) in enumerate(𝒮)
            u = backup(𝒫, U, s)
            U[i] = u
        end
    end
    return ValueFunctionPolicy(𝒫, U)
end


# ------------- Linear Programming --------------

"""
    struct LinearProgramFormulation end
"""
struct LinearProgramFormulation end

"""
    function tensorform(𝒫::MDP)

Convert an MDP into its tensor form, where the states and actions consist of integer indicies, the reward function is a matrix, and the transition function is a three-dimensional tensor.
"""
function tensorform(𝒫::MDP)
    𝒮, 𝒜, R, T = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T
    𝒮′ = eachindex(𝒮)
    𝒜′ = eachindex(𝒜)
    R′ = [R(s,a) for s in 𝒮, a in 𝒜]
    T′ = [T(s,a,s′) for s in 𝒮, a in 𝒜, s′ in 𝒮]
    return 𝒮′, 𝒜′, R′, T′
end


"""
    solve(𝒫::MDP) = solve(LinearProgramFormulation(), 𝒫)

Default MDP solver is the LinearProgramFormulation solver.
"""
solve(𝒫::MDP) = solve(LinearProgramFormulation(), 𝒫)


"""
    function solve(M::LinearProgramFormulation, 𝒫::MDP)
"""
function solve(M::LinearProgramFormulation, 𝒫::MDP)
    𝒮, 𝒜, R, T = tensorform(𝒫)
    model = Model(GLPK.Optimizer)
    @variable(model, U[𝒮])
    @objective(model, Min, sum(U))
    @constraint(model, [s=𝒮,a=𝒜], U[s] ≥ R[s,a] + 𝒫.γ*T[s,a,:]⋅U)
    optimize!(model)
    return ValueFunctionPolicy(𝒫, value.(U))
end


# ---------- Linear Quadratic Problem -----------


"""
    struct LinearQuadraticProblem
        Ts # transition matrix with respect to state
        Ta # transition matrix with respect to action
        Rs # reward matrix with respect to state (negative semidefinite)
        Ra # reward matrix with respect to action (negative definite)
        h_max # horizon
    end
"""
struct LinearQuadraticProblem
    Ts # transition matrix with respect to state
    Ta # transition matrix with respect to action
    Rs # reward matrix with respect to state (negative semidefinite)
    Ra # reward matrix with respect to action (negative definite)
    h_max # horizon
end

"""
    function solve(𝒫::LinearQuadraticProblem)
"""
function solve(𝒫::LinearQuadraticProblem)
    Ts, Ta, Rs, Ra, h_max = 𝒫.Ts, 𝒫.Ta, 𝒫.Rs, 𝒫.Ra, 𝒫.h_max
    V = zeros(size(Rs))
    πs = Any[s -> zeros(size(Ta, 2))]
    for h in 2:h_max
        V = Ts'*(V - V*Ta*((Ta'*V*Ta + Ra) \ Ta'*V))*Ts + Rs
        L = -(Ta'*V*Ta + Ra) \ Ta' * V * Ts
        push!(πs, s -> L*s)
    end
    return πs
end
