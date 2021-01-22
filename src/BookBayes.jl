
"""
    module BookBayes

Library of functions given in Algorithms for Decision Making
"""
module BookBayes

using Reexport
using NamedTupleTools
using LightGraphs
using IterTools
using Distributions
using SpecialFunctions
using LinearAlgebra

export
        Variable,
        Factor,
        variablenames,
        namedtuple,
        assignments,
        normalize!,
        FactorTable,
        BayesianNetwork,
        probability



using Base.Iterators: product

"""
    const DAG = SimpleDiGraph

Data structure representing a directed acyclic graph
"""
const DAG = SimpleDiGraph

include("ch2.jl")
include("ch3.jl")
include("ch4.jl")
include("ch5.jl")


end # module
