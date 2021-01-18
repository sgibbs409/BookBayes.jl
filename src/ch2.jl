



"""
    const FactorTable = Dict{NamedTuple,Float64}


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

    z = sum(p for (a,p) in ϕ.table)
    for (a,p) in ϕ.table
        ϕ.table[a] = p/z
    end
    return ϕ
end
