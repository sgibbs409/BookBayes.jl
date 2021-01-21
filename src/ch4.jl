

using LinearAlgebra

export sub2ind,
        statistics,
        prior,
        gaussian_kernel,
        kernel_density_estimate


"""
    function sub2ind(siz, x)

Convert a length n cartesian coordinate for indexing an n dimensional array into its linear (scalar) equivalent.

# Arguments

* siz::NTuple{Int}
  * Result of calling size(A) on the matrix of interest A.

* x::NTuple{Int}
  * length(siz)-length cartesian index to convert from
"""
function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

"""
    function statistics(vars, G, D::Matrix{Int})

Extract the statistics (ie counts) from a discrete dataset D assuming a Bayesian network with variables vars and structure G.  The dataset is an _n x m_ matrix, where n is the number of variables, and m is the number of data points.

# Returns

Returns an array *M* of length _n_.  The _i_th component consists of a _qᵢ x rᵢ_ matrix of counts.
"""
function statistics(vars, G, D::Matrix{Int})
    # n::number of variables
    n = size(D, 1)
    # rᵢ::for each variable i, number of discrete values it can take
    r = [vars[i].m for i in 1:n]

    # qᵢ: number of unique value combinations variable i's direct parents can take
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]

    # initialize counts matrix
    M = [zeros(q[i], r[i]) for i in 1:n]

    # for each data point (one per column of D)
    for o in eachcol(D)
        # for each variable in each sample
        for i in 1:n
            # actual category observed in this data point
            k = o[i]
            # parents of variable i
            parents = inneighbors(G,i)
            j=1

            # Find linear index of parents
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
            #increment mᵢⱼₖ
            M[i][j,k] += 1.0
        end
    end
    return M
end


"""
    function prior(vars, G)

"""
function prior(vars, G)

    n = length(vars)

    r = [vars[i].m for i in 1:n]

    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]

    return [ones(q[i], r[i]) for i in 1:n]
end


"""
    gaussian_kernel(b)

"""
gaussian_kernel(b) = x->pdf(Normal(0,b), x)

"""
    function kernel_density_estimate(φ, O)

"""
function kernel_density_estimate(φ, O)
    return x -> sum([φ(x - o) for o in O])/length(O)
end
