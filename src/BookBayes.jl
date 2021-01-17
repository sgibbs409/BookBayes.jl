module BookBayes

import Base.Iterators.product
const FactorTable = Dict{NamedTuple,Float64}

struct Variable
    name::Symbol
    m::Int # number of possible values
end


struct Factor
    vars::Vector{Variable}
    table::FactorTable
end

variablenames(ϕ::Factor) = [var.name for var in ϕ.vars]

function namedtuple(n::Array{Symbol,1}, v::Tuple)
    return NamedTuple{Tuple(n)}(v)
end

function assignments(vars::AbstractVector{Variable})
    n = [var.name for var in vars]
    #n = Tuple([var.name for var in vars])
    #return [NamedTuple{n}(v) for v in product((1:v.m for v in vars)...)]
    return [namedtuple(n, v) for v in product((1:v.m for v in vars)...)]
end

X = Variable(:x, 2)
Y = Variable(:y, 2)
Z = Variable(:z, 2)

ϕ = Factor([X, Y, Z], FactorTable(
    (x=1, y=1, z=1) => 0.08,
    (x=1, y=1, z=2) => 0.31,
    (x=1, y=2, z=1) => 0.09,
    (x=1, y=2, z=2) => 0.37,
    (x=2, y=1, z=1) => 0.01,
    (x=2, y=1, z=2) => 0.05,
    (x=2, y=2, z=1) => 0.02,
    (x=2, y=2, z=2) => 0.07,
))

# display(ϕ.table)
#
# display(ϕ.vars)
#
# display(variablenames(ϕ))


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

k = assignments(ϕ.vars)
@show k
@show k[2,1,1]
@show k[2,1,1].x
@show k[2,1,1].y
@show k[2,1,1].z

println(k)
println(typeof(k))
println(size(k))

sum([2,3,2])




end # module
