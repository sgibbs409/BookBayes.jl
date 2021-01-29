


export bayesian_score_component,
        bayesian_score,
        K2Search,
        LocalDirectedGraphSearch,
        rand_graph_neighbor,
        fit,
        are_markov_equivalent
"""
    function bayesian_score_component(M, α)

Compute the component of the Bayesian score for a single Variable.

# Arguments

    * M::Matrix{Integer}
        Count matrix of size qᵢ x rᵢ of data.

    * α::Matrix{Union{Float, Integer}}
        Matrix (of same size as M) of Dirichlet priors of a single Variable's parameter.
"""
function bayesian_score_component(M, α)

    p = sum(loggamma.(α + M))

    p -= sum(loggamma.(α))

    p += sum(loggamma.(sum(α,dims=2)))

    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2)))

    return p
end

"""
    function bayesian_score(vars, G, D)

Compute the Bayesian score for a list of variables `vars` and a graph `G` given data `D`.  Uses a uniform prior `αᵢⱼₖ = 1` for all `i`, `j`, and `k`.
"""
function bayesian_score(vars, G, D)

    n = length(vars)

    # Vector of length n of counts from Data evidence
    M = statistics(vars, G, D)

    # Generate parameters for a uniform dirichlet prior of the parameters of G
    # where size(α) == size(M)
    α = prior(vars, G)

    # total bayesian score is sum of individual variable's scores
    return sum(bayesian_score_component(M[i], α[i]) for i in 1:n)
end

"""
    struct K2Search
        ordering::Vector{Int} # variable ordering
    end

Data structure with parameters to inform acyclic directed graph fitting following K2 Search algorithm.

# Fields

    * ordering::Vector{Int}
        Enforces a topological sort on the potential graph search space.
"""
struct K2Search
    ordering::Vector{Int} # variable ordering
end

"""
    function fit(method::K2Search, vars, D)

Find a semi-optimal graph structure over the Variables `vars` using the K2 Search algorithm given the dataset D.

# Arguments

    * method::K2Search
        Parameters to contol fit operation
    * vars::Vector{Variables}
    * D::Matrix{Integer}
        Data array of size n x m, where n == length(vars) (ie number of Variables) and m == number of data points.
"""
function fit(method::K2Search, vars, D)

    # initial (unconnected) directed acyclic graph
    G = SimpleDiGraph(length(vars))

    # iterate over each variable according to topological ordering
    #  (first variable is by definition without any parents)
    for (k,i) in enumerate(method.ordering[2:end])
        # initial score on this sweep
        y = bayesian_score(vars, G, D)
        while true
            y_best, j_best = -Inf, 0
            # for each possible parent (those already iterated)
            for j in method.ordering[1:k]
                # if no edge from potential parent to target
                if !has_edge(G, j, i)
                    # try adding edge
                    add_edge!(G, j, i)
                    # if new edge results in better score, record that
                    y′ = bayesian_score(vars, G, D)
                    if y′ > y_best
                        y_best, j_best = y′, j
                    end
                    # for now undo new edge
                    rem_edge!(G, j, i)
                end
            end
            # if the best new score was better than before, re-add that edge
            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else
            # otherwise no edge helped this time so move on to the next targer variable
                break
            end
        end
    end
    return G
end


"""
    struct LocalDirectedGraphSearch
        G::DAG          # initial graph
        k_max::Integer  # number of iterations
    end

Data structure with parameters to inform graph fit operation.
"""
struct LocalDirectedGraphSearch
    G::DAG          # initial graph
    k_max::Integer  # number of iterations
end

"""
    function rand_graph_neighbor(G::DAG)

Generate a new acyclic graph that is a random neighbor of G.
"""
function rand_graph_neighbor(G::DAG)
    n = nv(G)       # number of Variables in graph
    i = rand(1:n)   # out-node of new random edge (all equally likely)
    j = mod1(i + rand(2:n)-1, n) # in-node of new random edge (all equally likely except node picked to be out-node)

    G′ = copy(G)
    # if edge exists, remove, else add
    has_edge(G, i, j) ? rem_edge!(G′, i, j) : add_edge!(G′, i, j)
    return G′
end

"""
    function fit(method::LocalDirectedGraphSearch, vars, D)

Fit a graph structure to data set D of samples of Variables vars, following Local Directed Graph Search algorithm.
"""
function fit(method::LocalDirectedGraphSearch, vars, D)
    G = method.G
    # initial best score
    y = bayesian_score(vars, G, D)
    # iterate k_max times
    for k in 1:method.k_max
        # get a random graph neighbor
        G′ = rand_graph_neighbor(G)
        # If random neighbor is cyclic => invalid graph, so reject by forcing score to minimum possible.  Else get neighbor's score
        y′ = is_cyclic(G′) ? -Inf : bayesian_score(vars, G′, D)
        # update best score and graph if better than previous
        if y′ > y
            y, G = y′, G′
        end
    end
    return G
end

"""
    function are_markov_equivalent(G::DAG, H::DAG)

Determine whether the directed acyclic graphs `G` and `H` are Markov equivalent.
"""
function are_markov_equivalent(G::DAG, H::DAG)

    # to be equivalent, both must have same set of node pairs that do not have any edge.
    if nv(G) != nv(H) || ne(G) != ne(H) ||
        !all(has_edge(H, e) || has_edge(H, reverse(e)) for e in edges(G))
            return false
    end

    # must have same v-structures
    for c in 1:nv(G)
        parents = inneighbors(G, c)

        for (a, b) in subsets(parents, 2)

            if !has_edge(G, a, b) && !has_edge(G, b, a) && !(has_edge(H, a, c) && has_edge(H, b, c))

                return false
            end
        end
    end

    return true
end
