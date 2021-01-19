import Base.:*

"""
    function Base.:*(ϕ::Factor, ψ::Factor)

Multiply two (joint and conditional) Factors to form a new (joint) Factor.
"""
function Base.:*(ϕ::Factor, ψ::Factor)

    ϕnames = variablenames(ϕ)
    ψnames = variablenames(ψ)
    shared = intersect(ϕnames, ψnames) # set of variables (names) in both
    ψonly = setdiff(ψ.vars, ϕ.vars)    # Variables unique to ψ ( ψ & !ϕ)
    table = FactorTable()

    for (ϕa, ϕp) in ϕ.table            # for assignment => probability in ϕ
        for a in assignments(ψonly)    # for each poss. assign. tuple in ψonly

            # Union of assignments
            a = merge(ϕa, a)

            # if ψ is empty, empty assignment, else extract name=>val tuple pair of ψ's variable assignments
            ψa = isempty(ψ.vars) ? NamedTuple() : select(a, ψnames)

            # add to new table mapping from a => p_ϕ(ϕa) * p_ψ(ψa | ϕa) = p_ϕψ(ϕa,ψa).  If ψa has been removed from ψ.table (for example, because ψ is conditioned on ψa being false) return 0.0.
            table[a] = ϕp * get(ψ.table, ψa, 0.0)

        end
    end

    # union of variables
    vars = vcat(ϕ.vars, ψonly)

    # return new (unnormalized) Factor
    return Factor(vars, table)
end




"""
    function marginalize(ϕ::Factor, name)

Integrate `ϕ::Factor` by `Variable` with name `name` to get new Factor with marginal distribution over remaining `Variable`s.
"""
function marginalize(ϕ::Factor, name)

    # new FactorTable
    table = FactorTable()

    # for each assignment=>probability pair in ϕ
    for (a,p) in ϕ.table
        # remove (if present) name=>val(name) from a
        a′ = delete(a,name)
        # update running sum of vals for the remining assignment terms
        table[a′] = get(table, a′, 0.0) + p
    end

    vars = filter(v->v.name != name, ϕ.vars)
    return Factor(vars, table)
end


# predicate function: true if any Variable v in ϕ has v.name == name
#   ie if name is a variable in ϕ
function in_scope(name, ϕ)

    return any(name == v.name for v in ϕ.vars)
end



function condition(φ::Factor, name, value)
    if !in_scope(name, φ)
        return φ
    end
    table = FactorTable()
    for (a, p) in φ.table
        if a[name] == value
            table[delete(a, name)] = p
        end
    end
    vars = filter(v -> v.name != name, φ.vars)
    return Factor(vars, table)
end

function condition(φ::Factor, evidence)
    for (name, value) in pairs(evidence)
        φ = condition(φ, name, value)
    end
    return φ
end




"""
    struct ExactInference

Singleton type to pass instances of as first arg of `infer` to trigger direct inference method.
"""
struct ExactInference end

"""
    function infer(M::ExactInference, bn, query, evidence)

Get a joint distribution over query variables, given evidence.  Uses direct inference.

Fields:

    M::ExactInference
Singleton value to use this version of infer

    bn
BayesianNetwork specifying original distribution

    query
Tuple of Symbols of Variable names to get joint distribution over.

    evidence
Dictionary of Variable Symbol name => Variable value pairs specifying known values.
"""
function infer(M::ExactInference, bn, query, evidence)
    # get joint distribution over all variables
    φ = prod(bn.factors)

    # Convert to conditional joint distribution
    φ = condition(φ, evidence)

    # Integrate out any remaining variables not in query
    for name in setdiff(variablenames(φ), query)
        φ = marginalize(φ, name)
    end

    # Normalize to 1.0 total probability
    return normalize!(φ)
end



"""
    struct VariableElimination

Singleton type to pass instances of as first arg of `infer` to trigger variable elimination method.

    ordering
Iterable over variable indices indicating the order over the factors to use.
"""
struct VariableElimination
    ordering # array of variable indices
end


"""
    function infer(M::VariableElimination, bn, query, evidence)

Get a joint distribution over query variables, given evidence.  Uses the sum-product variable elimination algorithm.

Fields:

    M::VariableElimination
Singleton value to dispatch this version of infer

    bn
BayesianNetwork specifying original distribution

    query
Tuple of Symbols of Variable names to get joint distribution over.

    evidence
Dictionary of Variable Symbol name => Variable value pairs specifying known values.
"""
function infer(M::VariableElimination, bn, query, evidence)

    # precondition each factor on evidence
    Φ = [condition(φ, evidence) for φ in bn.factors]

    # iterate over each Variable using given ordering
    for i in M.ordering
        name = bn.vars[i].name

        # if not a query variable (needs to be eliminated)
        if name ∉ query

            # find all factors using this Variable
            inds = findall(φ->in_scope(name, φ), Φ)

            if !isempty(inds)

                # join factors using this variable
                φ = prod(Φ[inds])

                # remove those factors from remaining factor list
                deleteat!(Φ, inds)

                # integrate the joined factor over this var (eliminate it)
                φ = marginalize(φ, name)

                # put reduced factor back in list
                push!(Φ, φ)
            end

            # assert: isempty(findall(φ->in_scope(name, φ), Φ))
        end
    end

    # assert: Φ now contains only factors with only query variables

    # finally, normalize remaining joint distribution to 1.0 total probability
    return normalize!(prod(Φ))
end



"""
    function Base.rand(φ::Factor)

Direct sample a discrete Factor  given by `φ`
"""
function Base.rand(φ::Factor)

    tot, p, w = 0.0, rand(), sum(values(φ.table))
    for (a,v) in φ.table
        tot += v/w
        if tot >= p
            return a
        end
    end

    return NamedTuple()
end


# import Base.isless
#
# function Base.isless()

"""
    function Base.rand(bn::BayesianNetwork)

Direct sample from a joint distribution given by `bn`.

**Returns:** NamedTuple of Variable name => value pairs sampled from bn.
"""
function Base.rand(bn::BayesianNetwork)

    # tuple to hold new random sample result
    a = NamedTuple()

    # requires iterating over factors following a topological sort
    for i in topological_sort_by_dfs(bn.graph)

        # get the var name and cooresonding factor
        name, φ = bn.vars[i].name, bn.factors[i]

        # condition factor on previously sampled components (which leaves only variable `name` unspecified) and sample
        value = rand(condition(φ, a))[name]

        # append new value to sample results
        a = merge(a, namedtuple(name)(value))
    end

    return a
end


struct DirectSampling
    m # number of samples
end


"""
    function infer(M::DirectSampling, bn, query, evidence)

Get a joint distribution over query variables, given evidence.  Uses the direct sampling inference method to draw m samples from network consistent with evidence.

Fields:

    M::DirectSampling
Singleton value to dispatch this version of infer

    bn
BayesianNetwork specifying original distribution

    query
Tuple of Symbols of Variable names to get joint distribution over.

    evidence
Dictionary of Variable Symbol name => Variable value pairs specifying known values.
"""
function infer(M::DirectSampling, bn, query, evidence)
    table = FactorTable()

    # take m samples
    for i in 1:(M.m)

        # sample a full joint assignment from network
        a = rand(bn)

        # if all values in sample consistent with evidence
        if all(a[k] == v for (k,v) in pairs(evidence))
            # select the query variable assignment subset
            b = select(a, query)

            # increment counter
            table[b] = get(table, b, 0) + 1
        end
    end

    # only vars in query
    vars = filter(v->v.name ∈ query, bn.vars)
    # normalize factor table before returning
    return normalize!(Factor(vars, table))
end


struct LikelihoodWeightedSampling
    m # number of samples
end


function infer(M::LikelihoodWeightedSampling, bn, query, evidence)
    table = FactorTable()
    ordering = topological_sort_by_dfs(bn.graph)

    for i in 1:(M.m)

        a, w = NamedTuple(), 1.0

        for j in ordering

            name, φ = bn.vars[j].name, bn.factors[j]

            if haskey(evidence, name)
                a = merge(a, namedtuple(name)(evidence[name]))
                w *= φ.table[select(a, variablenames(φ))]
            else
                val = rand(condition(φ, a))[name]
                a = merge(a, namedtuple(name)(val))
            end
        end

        b = select(a, query)
        table[b] = get(table, b, 0) + w
    end

    vars = filter(v->v.name ∈ query, bn.vars)
    return normalize!(Factor(vars, table))
end








function blanket(bn, a, i)
    name = bn.vars[i].name
    val = a[name]
    a = delete(a, name)
    Φ = filter(φ -> in_scope(name, φ), bn.factors)
    φ = prod(condition(φ, a) for φ in Φ)
    return normalize!(φ)
end


function update_gibbs_sample(a, bn, evidence, ordering)
    for i in ordering
        name = bn.vars[i].name
        if !haskey(evidence, name)
            b = blanket(bn, a, i)
            a = merge(a, namedtuple(name)(rand(b)[name]))
        end
    end
    return a
end


function gibbs_sample(a, bn, evidence, ordering, m)
    for j in 1:m
        a = update_gibbs_sample(a, bn, evidence, ordering)
    end
    return a
end

struct GibbsSampling
    m_samples # number of samples to use
    m_burnin # number of samples to discard during burn-in
    m_skip # number of samples to skip for thinning
    ordering # array of variable indices
end

function infer(M::GibbsSampling, bn, query, evidence)
    table = FactorTable()
    a = merge(rand(bn), evidence)
    a = gibbs_sample(a, bn, evidence, M.ordering, M.m_burnin)

    for i in 1:(M.m_samples)
        a = gibbs_sample(a, bn, evidence, M.ordering, M.m_skip)
        b = select(a, query)
        table[b] = get(table, b, 0) + 1
    end

    vars = filter(v->v.name ∈ query, bn.vars)
    return normalize!(Factor(vars, table))
end










@doc raw"""

    function infer(D::MvNormal, query, evidencevars, evidence)

Get the conditional normal distribution over the query variables, given the evidence variables

$\left[
\begin{array}{c}
   \bf a \\
   \bf b
\end{array}
\right] \sim \mathcal{N} \left( \left[
\begin{array}{c}
  \bf \mu_a \\
  \bf \mu_b
\end{array}
\right] , \left[
\begin{array}{cc}
  \bf A & \bf C \\
  \bf C^T & \bf B
\end{array}
\right] \right)$

The conditional distribution is:

$p(\textbf{a} | \textbf{b}) = \mathcal{N}( \bf a\ |\  \mu_{a|b},\ \Sigma_{a|b})$

$\boxed{\bf \mu_{a|b} = \mu_a + CB^{-1}(b - \mu_b)}$

$\boxed{\Sigma_{a|b} = A - CB^{-1}C^T}$


# Examples

Consider:

$\left[
\begin{array}{c}
   x_1 \\
   x_2
\end{array}
\right] \sim \mathcal{N} \left( \left[
\begin{array}{c}
  0 \\
  1
\end{array}
\right] , \left[
\begin{array}{cc}
  3 & 1 \\
  1 & 2
\end{array}
\right] \right)$

Then the conditional distribution for x₁ given x₂ = 2 is:

$\mu_{x_1|x_2=2} = 0 + 1 \cdot 2^{-1} \cdot (2 - 1) = 0.5$

$\Sigma_{x_1|x_2=2} = 3 - 1 \cdot 2^{-1}\cdot 1 = 2.5$


    julia>  d = MvNormal([0.0,1.0], [3.0 1.0; 1.0 2.0])
	d = FullNormal(
		dim: 2
		μ: [0.0, 1.0]
		Σ: [3.0 1.0; 1.0 2.0]
		)

    # get conditional dist for x₁ given x₂ = 2.0
    julia> infer(d, [1], [2], [2.0])
	FullNormal(
	dim: 1
	μ: [0.5]
	Σ: [2.5]
	)
"""
function infer(D::MvNormal, query, evidencevars, evidence)

	#  extract full distribution params
    μ, Σ = D.μ, D.Σ.mat

    b = evidence
	μa = μ[query]
	μb = μ[evidencevars]

    A = Σ[query,query]
    B = Σ[evidencevars,evidencevars]
    C = Σ[query,evidencevars]

    μ = μ[query] + C * (B\(b - μb))
    Σ = A - C * (B \ C')
    return MvNormal(μ, Σ)
end
