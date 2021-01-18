module BookBayes

using Reexport

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
