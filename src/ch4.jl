


export sub2ind,
        statistics,
        prior,
        gaussian_kernel,
        kernel_density_estimate


"""
    function sub2ind(siz, x)

"""
function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

"""
    function statistics(vars, G, D::Matrix{Int})

"""
function statistics(vars, G, D::Matrix{Int})
    n = size(D, 1)
    r = [vars[i].m for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    M = [zeros(q[i], r[i]) for i in 1:n]

    for o in eachcol(D)
        for i in 1:n k = o[i]
            parents = inneighbors(G,i) j=1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
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
