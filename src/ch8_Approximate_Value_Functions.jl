


export  ApproximateValueIteration,
        solve,
        NearestNeighborValueFunction,
        fit!,
        LocallyWeightedValueFunction,
        MultilinearValueFunction,
        SimplexValueFunction,
        LinearRegressionValueFunction


"""
    struct ApproximateValueIteration
        Uθ # initial parameterized value function that supports fit!
        S # set of discrete states for performing backups
        k_max # maximum number of iterations
    end
"""
struct ApproximateValueIteration
    Uθ # initial parameterized value function that supports fit!
    S # set of discrete states for performing backups
    k_max # maximum number of iterations
end


"""
    function solve(M::ApproximateValueIteration, 𝒫::MDP)
"""
function solve(M::ApproximateValueIteration, 𝒫::MDP)
    Uθ, S, k_max = M.Uθ, M.S, M.k_max
    for k in 1:k_max
        U = [backup(𝒫, Uθ, s) for s in S]
        fit!(Uθ, S, U)
    end

    return ValueFunctionPolicy(𝒫, Uθ)
end


"""
    mutable struct NearestNeighborValueFunction
        k # number of neighbors
        d # distance function d(s, s′)
        S # set of discrete states
        θ # vector of values at states in S
    end
"""
mutable struct NearestNeighborValueFunction
    k # number of neighbors
    d # distance function d(s, s′)
    S # set of discrete states
    θ # vector of values at states in S
end

"""
    function (Uθ::NearestNeighborValueFunction)(s)

Approximate value function at s as average of value function over k nearest states (whose value function value is already known).
"""
function (Uθ::NearestNeighborValueFunction)(s)
    # distances from s to each state s′ in state subset S
    dists = [Uθ.d(s,s′) for s′ in Uθ.S]
    # indicies of shortest 1:k distances (ind[1] == index of closest neighbor)
    ind = sortperm(dists)[1:Uθ.k]
    # average value of k nearest neighbors
    return mean(Uθ.θ[i] for i in ind)
end

"""
    function fit!(Uθ::NearestNeighborValueFunction, S, U)
"""
function fit!(Uθ::NearestNeighborValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end


"""
    mutable struct LocallyWeightedValueFunction
        k # kernel function k(s, s′)
        S # set of discrete states
        θ # vector of values at states in S
    end
"""
mutable struct LocallyWeightedValueFunction
    k # kernel function k(s, s′)
    S # set of discrete states
    θ # vector of values at states in S
end

"""
    function (Uθ::LocallyWeightedValueFunction)(s)
"""
function (Uθ::LocallyWeightedValueFunction)(s)
    w = normalize([Uθ.k(s,s′) for s′ in Uθ.S], 1)
    return Uθ.θ ⋅ w
end

"""
    function fit!(Uθ::LocallyWeightedValueFunction, S, U)
"""
function fit!(Uθ::LocallyWeightedValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end



# ------------------ MultilinearValueFunction ---------------------

# Also implemented in Interpolations.jl

"""
    mutable struct MultilinearValueFunction
        o # position of lower-left corner
        δ # vector of widths
        θ # vector of values at states in S
    end
"""
mutable struct MultilinearValueFunction
    o # position of lower-left corner
    δ # vector of widths
    θ # vector of values at states in S
end

"""
    function (Uθ::MultilinearValueFunction)(s)

Use multilinear interpolation to estimate the value of state vector `s` for known state values `θ` over a grid defined by lower-left vertex `o` and vector of widths `δ`.  Verticies of the grid can all be written as `o + δ.*i` for some non-negative integer vector `i`.
"""
function (Uθ::MultilinearValueFunction)(s)
    o, δ, θ = Uθ.o, Uθ.δ, Uθ.θ
    Δ = (s - o)./δ
    # Multidimensional index of lower-left cell
    i = min.(floor.(Int, Δ) .+ 1, size(θ) .- 1)
    vertex_index = similar(i)
    d = length(s)
    u = 0.0
    for vertex in 0:2^d-1
        weight = 1.0
        for j in 1:d
            # Check whether jth bit is set
            if vertex & (1 << (j-1)) > 0
                vertex_index[j] = i[j] + 1
                weight *= Δ[j] - i[j] + 1
            else
                weight *= i[j] - Δ[j]
            end
        end
        u += θ[vertex_index...]*weight
    end

    return u
end


"""
    function fit!(Uθ::MultilinearValueFunction, S, U)
"""
function fit!(Uθ::MultilinearValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end


# --------------- SimplexValueFunction ------------------
#
# Also implemented in GridInterpolations.jl

"""
    mutable struct SimplexValueFunction
        o # position of lower-left corner
        δ # vector of widths
        θ # vector of values at states in S
    end
"""
mutable struct SimplexValueFunction
    o # position of lower-left corner
    δ # vector of widths
    θ # vector of values at states in S
end

"""
    function (Uθ::SimplexValueFunction)(s)

Estimate value of state vector `s` for known values `θ` over a grid defined by a lower-left vertex `o` and vector of widths `δ`.  Vertices of the grid can all be written `o + δ.*i` for some non-negative integral vector `i`.
"""
function (Uθ::SimplexValueFunction)(s)
    Δ = (s - Uθ.o)./Uθ.δ
    # Multidimensional index of upper-right cell
    i = min.(floor.(Int, Δ) .+ 1, size(Uθ.θ) .- 1) .+ 1
    u = 0.0
    s′ = (s - (Uθ.o + Uθ.δ.*(i.-2))) ./ Uθ.δ
    p = sortperm(s′) # increasing order
    w_tot = 0.0
    for j in p
        w = s′[j] - w_tot
        u += w*Uθ.θ[i...]
        i[j] -= 1
        w_tot += w
    end

    u += (1 - w_tot)*Uθ.θ[i...]
    return u
end


"""
    function fit!(Uθ::SimplexValueFunction, S, U)
"""
function fit!(Uθ::SimplexValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end


# -------------- Linear Regression Value Function ---------------



"""
    mutable struct LinearRegressionValueFunction
        β # basis vector function
        θ # vector of paramters
    end
"""
mutable struct LinearRegressionValueFunction
    β # basis vector function
    θ # vector of paramters
end


"""
    function (Uθ::LinearRegressionValueFunction)(s)
"""
function (Uθ::LinearRegressionValueFunction)(s)
    return Uθ.β(s) ⋅ Uθ.θ
end

"""
    function fit!(Uθ::LinearRegressionValueFunction, S, U)
"""
function fit!(Uθ::LinearRegressionValueFunction, S, U)
    X = hcat([Uθ.β(s) for s in S]...)'
    Uθ.θ = pinv(X)*U
    return Uθ
end
