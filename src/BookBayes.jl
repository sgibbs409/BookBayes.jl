
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

export
        Variable,
        Factor,
        variablenames,
        namedtuple,
        assignments,
        normalize!,
        FactorTable,
        BayesianNetwork,
        probability,
        marginalize,
        in_scope,
        condition,
        ExactInference,
        infer,
        VariableElimination,
        DirectSampling,
        blanket,
        update_gibbs_sample,
        gibbs_sample,
        GibbsSampling


using Base.Iterators: product

include("ch2.jl")
include("ch3.jl")

end # module
