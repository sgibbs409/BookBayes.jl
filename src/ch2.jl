



"""
    const FactorTable = Dict{NamedTuple,Float64}

Data structure representing a conditional probability distribution of a single variable given zero or more conditional variables.

Keys: named tuples of variable names to values.
Values: conditional probability
"""
const FactorTable = Dict{NamedTuple,Float64}



"""
    struct Variable
        name::Symbol
        m::Int # number of possible values
    end

Data structure representing a single node in a discrete bayesian network
"""
struct Variable
    name::Symbol
    m::Int # number of possible values
end



"""
    struct Factor
        vars::Vector{Variable}
        table::FactorTable
    end

Data structure representing a discrete conditional probability distribution
"""
struct Factor
    vars::Vector{Variable}
    table::FactorTable
end


"""
    variablenames(ϕ::Factor) = [var.name for var in ϕ.vars]

Get Vector of the variable names of all variables in ϕ
"""
variablenames(ϕ::Factor) = [var.name for var in ϕ.vars]



"""
    function namedtuple(n::Array{Symbol,1}, v::Tuple)

Create a NamedTuple of pairs of ::Symbol => ::Any

### ARGS

- n:  Array of Symbols giving names of assignment variables
- v:  Tuple of values, one per name Symbol
"""
function namedtuple(n::Array{Symbol,1}, v::Tuple)
    return NamedTuple{Tuple(n)}(v)
end



"""
    function assignments(vars::AbstractVector{Variable})

Get a vector of every possible valid namedtuple corresponding to a possible valid (Integer (Categorical)) value assignment of the Variables in vars.

### ARGS

### EXAMPLES

(:a => 3, :b => 1, :c => 7)
"""
function assignments(vars::AbstractVector{Variable})
    n = [var.name for var in vars]
    #n = Tuple([var.name for var in vars])
    #return [NamedTuple{n}(v) for v in product((1:v.m for v in vars)...)]
    return [namedtuple(n, v) for v in product((1:v.m for v in vars)...)]
end




"""
    normalize!(ϕ::Factor)
Normalize a factor ϕ, which divides all the entries in the factor by the
same scalar so they sum to 1.
"""
function normalize!(ϕ::Factor)

    # total sum of probabilities in table
    z = sum(p for (a,p) in ϕ.table)
    for (a,p) in ϕ.table
        ϕ.table[a] = p/z
    end
    return ϕ
end




"""
    struct BayesianNetwork
        vars::Vector{Variable}
        factors::Vector{Factor}
        graph::SimpleDiGraph
    end

Data structure representing a discrete bayesian network.
"""
struct BayesianNetwork
    vars::Vector{Variable}
    factors::Vector{Factor}
    graph::SimpleDiGraph
end

"""
    function probability(bn::BayesianNetwork, assignment)

Calculate the probability of a given variable assignment given the bayesian network bn.

# Arguments

* bn: The BayesianNetwork describing the joint distribution
* assignment: a NamedTuple of :Symbol=>Integer pairs giving the
            categorical index value of each variable.

# Examples
```julia
julia> X = Variable(:x, 2);

julia> Y = Variable(:y, 2);

julia> Z = Variable(:z, 2);

julia> ϕx = Factor([X, Y, Z], FactorTable(
           (x=1, y=1, z=1) => 0.08,
           (x=1, y=1, z=2) => 0.31,
           (x=1, y=2, z=1) => 0.09,
           (x=1, y=2, z=2) => 0.37,
           (x=2, y=1, z=1) => 0.01,
           (x=2, y=1, z=2) => 0.05,
           (x=2, y=2, z=1) => 0.02,
           (x=2, y=2, z=2) => 0.07,
       ));

julia> ϕy = Factor([Y], FactorTable(
           (y=1,) => 0.20,
           (y=2,) => 0.80,
       ));

julia> ϕz = Factor([Z], FactorTable(
           (z=1,) => 0.40,
           (z=2,) => 0.60,
       ));

julia> g = SimpleDiGraph(3);

julia> add_edge!(g,1,3);

julia> add_edge!(g,2,3);

julia> bn = BayesianNetwork([Y, Z, X], [ϕy, ϕz, ϕ], g);

julia> assignment = (x=1, y=2, z=1);

julia> probability(bn, assignment)
0.028800000000000006
```

"""
function probability(bn::BayesianNetwork, assignment)
    select(ϕ) = NamedTupleTools.select(assignment, variablenames(ϕ))
    probability(ϕ) = ϕ.table[select(ϕ)]
    return prod(probability(ϕ) for ϕ in bn.factors)
end
