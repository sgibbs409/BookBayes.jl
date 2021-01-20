import Base.:*


export LikelihoodWeightedSampling,
        DirectSampling,
        GibbsSampling,
        ExactInference,
        marginalize,
        in_scope,
        condition,
        infer,
        VariableElimination,
        blanket,
        update_gibbs_sample,
        gibbs_sample


"""
    struct LikelihoodWeightedSampling
        m # number of samples
    end
"""
struct LikelihoodWeightedSampling
    m # number of samples
end

"""
    struct DirectSampling
        m # number of samples
    end
"""
struct DirectSampling
    m # number of samples
end

"""
    struct GibbsSampling
        m_samples # number of samples to use
        m_burnin # number of samples to discard during burn-in
        m_skip # number of samples to skip for thinning
        ordering # array of variable indices
    end

Data structure specifying parameters to use when Gibbs Sampling a BayesianNetwork
"""
struct GibbsSampling
    m_samples # number of samples to use
    m_burnin # number of samples to discard during burn-in
    m_skip # number of samples to skip for thinning
    ordering # array of variable indices
end


"""
    struct ExactInference

Singleton type to pass instances of as first arg of `infer` to trigger direct inference method.
"""
struct ExactInference end

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


"""
    function in_scope(name, ϕ)

predicate function: true if any Variable v in ϕ has v.name == name (ie if name is a variable in ϕ)

# Arguments
* name::Symbol: Name of variable to test for presense of in Factor ϕ.
* ϕ::Factor: Factor to test if variable named <name> is present in.
"""
function in_scope(name, ϕ)

    return any(name == v.name for v in ϕ.vars)
end


"""
    function condition(φ::Factor, name, value)

Method for factor conditioning.  Take a factor **ϕ** and return a new factor whose table entries are consistent with the variable named **name** having value **value**.
"""
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


"""
    function condition(φ::Factor, evidence)

Method for factor conditioning.  Take a factor **ϕ** and apply **evidence** in the form of a named tuple.
"""
function condition(φ::Factor, evidence)
    for (name, value) in pairs(evidence)
        φ = condition(φ, name, value)
    end
    return φ
end






"""
    function infer(M::ExactInference, bn, query, evidence)

Get a joint distribution over query variables, given evidence.  Uses direct inference.

# Arguments

    M::ExactInference
Singleton value to use this version of infer

    bn::BayesianNetwork
BayesianNetwork specifying original distribution

    query::Tuple
Tuple of Symbols of Variable names to get joint distribution over.

    evidence::NamedTuple
Dictionary of Variable Symbol name => Variable value pairs specifying known values.
"""
function infer(M::ExactInference, bn::BayesianNetwork, query::Tuple, evidence::NamedTuple)
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

    bn::BayesianNetwork
BayesianNetwork specifying original distribution

    query::Tuple{Symbol}
Tuple of Symbols of Variable names to get joint distribution over.

    evidence::NamedTuple
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




"""
    function infer(M::DirectSampling, bn, query, evidence)

Get a joint distribution over query variables, given evidence.  Uses the direct sampling inference method to draw m samples from network consistent with evidence.

### Returns

Returns estimated joint distribution over query variables conditioned on evidence variable assignment.

# Arguments

* M::DirectSampling
  Singleton value to dispatch this version of infer

* bn
  BayesianNetwork specifying original distribution

* query
  Tuple of Symbols of Variable names to get joint distribution over.

* evidence
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





"""
    function infer(M::LikelihoodWeightedSampling, bn, query, evidence)

Get a joint distribution over query variables, given evidence.  Uses the Likelihood Weighted Sampling inference method to draw m samples from network consistent with evidence.

### Returns

Returns estimated joint distribution over query variables conditioned on evidence variable assignment.

# Arguments

* M::LikelihoodWeightedSampling
  Singleton value to dispatch this version of infer

* bn
  BayesianNetwork specifying original distribution

* query
  Tuple of Symbols of Variable names to get joint distribution over.

* evidence
  Dictionary of Variable Symbol name => Variable value pairs specifying known values.
"""
function infer(M::LikelihoodWeightedSampling, bn, query, evidence)
    table = FactorTable()
    ordering = topological_sort_by_dfs(bn.graph)

    # Take M.m samples
    for i in 1:(M.m)
        #initial assignment, weight
        a, w = NamedTuple(), 1.0

        # for each variable in network
        for j in ordering

            # name and factor of current variable
            name, φ = bn.vars[j].name, bn.factors[j]

            # if this variable is one of the evidence variables
            if haskey(evidence, name)
                # automatically add (name=>evidence_value_of_name) to this sample
                a = merge(a, namedtuple(name)(evidence[name]))

                # but the weight of this sample is downgraded by probability of evidence given corresponding assignment values already determined.
                w *= φ.table[select(a, variablenames(φ))]
            else
                # Not an evidence variable.  Randomly sample this factor, given prior determined values.
                val = rand(condition(φ, a))[name]
                # and add to this sample assignment
                a = merge(a, namedtuple(name)(val))
            end
        end
        #  a is now a complete assignment (all variables in bn assigned a value)

        # extract the query variable portion of assignment
        b = select(a, query)

        # add weight of this sample to corresponding entry in factor table
        table[b] = get(table, b, 0) + w

    end # end of sampling loop

    # variable list of query variables
    vars = filter(v->v.name ∈ query, bn.vars)

    # return new Factor over query variables
    return normalize!(Factor(vars, table))
end


@doc raw"""
    function blanket(bn::BayesianNetwork, a, i)

Calculate $P(X\_{i} | x\_{-i})$ from a Bayesian Network bn.

# Arguments

* bn:: BayesianNetwork

* a::NamedTuple:
  Complete assignment of values to every variable in bn.
* i::Integer
  Index of leave-one-out variable Xᵢ in network (according to some valid topological sort)
"""
function blanket(bn, a, i)
    # name of ith variable
    name = bn.vars[i].name
    # val of ith variable
    val = a[name]
    # remove variable i from assignment a
    a = delete(a, name)

    # get factors that contain variable i
    Φ = filter(φ -> in_scope(name, φ), bn.factors)
    #
    φ = prod(condition(φ, a) for φ in Φ)
    return normalize!(φ)
end


@doc raw"""
    function update_gibbs_sample(a, bn, evidence, ordering)

Single Gibbs Sampling sample loop.  Starting from complete assignment a of values to all variable in bn, sample each distribution $P(X\\_{i} | x\\_{-i})$ one at a time, replacing its entry in a with sampled value after each iteration.

# Arguments
* a::NamedTuple

* bn::BayesianNetwork

* evidence::NamedTuple

* ordering::Vector{Integer}:
  Vector of indices of each variable in bayesian network according to valid topological sort.

# Returns

Returns NamedTuple assignment updated with new sampled values
"""
function update_gibbs_sample(a, bn, evidence, ordering)
    # for each variable in bn according to topological sort
    for i in ordering
        # name of current variable
        name = bn.vars[i].name
        # If this variable is not an evidence variable
        if !haskey(evidence, name)
            # Calculate marginal distribution of variable i given assignment a
            b = blanket(bn, a, i)
            # sample marginal distribuion of i and update a with it.
            a = merge(a, namedtuple(name)(rand(b)[name]))
        end
    end
    return a
end


"""
    function gibbs_sample(a, bn, evidence, ordering, m)

Sample BayesianNetwork bn using Gibbs Sampling.  Start from NamedTuple assignment a, given evidence, with ordering giving valid topological sort to follow, and take m samples.

# Returns

Returns a new NamedTuple assignment approximately randomly sampled from bn given evidence.
"""
function gibbs_sample(a, bn, evidence, ordering, m)
    # Sample distribution m times, starting from a
    for j in 1:m
        # replace a with result of previous iteration
        a = update_gibbs_sample(a, bn, evidence, ordering)
    end
    return a
end




"""
    function infer(M::GibbsSampling, bn, query, evidence)

Estimate the conditional distribution over the query variables given the evidence values, using Gibbs Sampling
"""
function infer(M::GibbsSampling, bn, query, evidence)
    # New FactorTable to hold return Factor probabilities
    table = FactorTable()

    # Randomly sample bn (using direct sampling), and replace evidence vars with given evidence values.
    a = merge(rand(bn), evidence)

    # Gibbs Sample bn M.m_burnin times to initialize starting point
    a = gibbs_sample(a, bn, evidence, M.ordering, M.m_burnin)

    # Sample bn M.m_samples times
    for i in 1:(M.m_samples)
        # Keep every M.m_skipᵗʰ sample only
        a = gibbs_sample(a, bn, evidence, M.ordering, M.m_skip)
        # extract query variable values from sample
        b = select(a, query)
        # increment the sampled query value counter
        table[b] = get(table, b, 0) + 1
    end

    # Extract the query Variables from bn
    vars = filter(v->v.name ∈ query, bn.vars)
    # Return estimated query variable distribution
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
