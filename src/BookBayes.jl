
"""
    module BookBayes

Library of functions given in Algorithms for Decision Making
"""
module BookBayes

using Reexport
using NamedTupleTools
using LightGraphs
using IterTools


export
        Variable,
        Factor,
        variablenames,
        namedtuple,
        assignments,
        normalize!,
        FactorTable


using Base.Iterators: product

include("ch2.jl")


end # module
