





"""
    const FactorTable = Dict{NamedTuple,Float64}

Type alias of a Dictionary mapping tuples of Variable name=>value assignments to probabilities.
"""
const FactorTable = Dict{NamedTuple,Float64}


"""
    struct Variable

A Type representing a single discrete variable.

Params:

    name::Symbol
name of Variable

    m::Int:
number of possible (discrete) values this Variable can take
"""
struct Variable
    name::Symbol
    m::Int # number of possible values
end





"""
    struct Factor

A Type representing a joint or conditional probability distribution over the variables in vars.

Fields:

    vars:
The variables represented by this distribution

    table:
Probability table mapping Variable assignments (NamedTuple) => probability (Float64)
"""
struct Factor
    vars::Vector{Variable}
    table::FactorTable
end






"""
    function variablenames(φ::Factor)

Get a vector of variable names (Symbols) for the Variables in a Factor
"""
variablenames(φ::Factor) = [var.name for var in φ.vars]





"""
    function assignments(vars::AbstractVector{Variable})

Get an array of all possible variable=>value assignment tuples for Variables in vars.
"""
function assignments(vars::AbstractVector{Variable})

    # get a vector of variable names
    n = [var.name for var in vars]

    # product(1:v1.m, 1:v2.m, 1:v3.m) -> cartesian product of integer indicies
    #                                 -> iterator over tuples of indicies
    return [namedtuple(n, v) for v in product((1:v.m for v in vars)...)]
end






"""
    function normalize!(φ::Factor)

Update φ::Factor so all values sum to 1.0 while maintaining relative value (so values represent probabilities).
"""
function normalize!(ϕ::Factor)

    # sum all probabilities in Factor table
    z = sum(p for (a,p) in ϕ.table)

    # divide each value by total sum
    for (a,p) in ϕ.table
            ϕ.table[a] = p/z
    end

    return ϕ
end




"""
    struct BayesianNetwork

Represents a Bayesian Network.

Fields:

    vars::Vector{Variable}:
Vector of Variables contained in network

    factors::Vector{Factor}:
Vector of Factors, one for each Variable in vars, representing conditional probability distribution over that variable given parents of var in network.

    graph::SimpleDiGraph{Int64}:
Represents the structure of the network variables and implicitly defines independence relationships.

"""
struct BayesianNetwork

    """Vector of every Variable in the network"""
    vars::Vector{Variable}

    """
        Network conditional independence specified by a `Vector` of `Factor`s.
        Each Factor speifies the conditional joint PMF over its subset of variables, given the
    """
    factors::Vector{Factor}

    """
        DAG that specifies the independence relationships among variables.
        Nodes
    """
    graph::SimpleDiGraph{Int64}
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

        # Function to map Factor to a NamedTuple containing only the name=>value pairs for names specified in the (conditionally independent) Factor φ and values for them (and all other variables in bn) specified in `assignment`
        __select(φ) = select(assignment, variablenames(φ))

        # Function mapping Factor to probability of factor assignment
        probability(φ) = φ.table[__select(φ)]

        # Final probability is product of individual factor's probabilities (since they are implicitly independent)
        return prod(probability(φ) for φ in bn.factors)

end
