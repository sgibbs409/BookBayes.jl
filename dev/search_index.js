var documenterSearchIndex = {"docs":
[{"location":"anotherPage/#The-BookBayes-Module","page":"Another page","title":"The BookBayes Module","text":"","category":"section"},{"location":"anotherPage/","page":"Another page","title":"Another page","text":"BookBayes","category":"page"},{"location":"anotherPage/#BookBayes","page":"Another page","title":"BookBayes","text":"module BookBayes\n\nLibrary of functions given in Algorithms for Decision Making\n\n\n\n\n\n","category":"module"},{"location":"anotherPage/#Module-Index","page":"Another page","title":"Module Index","text":"","category":"section"},{"location":"anotherPage/","page":"Another page","title":"Another page","text":"Modules = [BookBayes]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"anotherPage/#Detailed-API","page":"Another page","title":"Detailed API","text":"","category":"section"},{"location":"anotherPage/","page":"Another page","title":"Another page","text":"Modules = [BookBayes]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"anotherPage/#BookBayes.BayesianNetwork","page":"Another page","title":"BookBayes.BayesianNetwork","text":"struct BayesianNetwork\n    vars::Vector{Variable}\n    factors::Vector{Factor}\n    graph::SimpleDiGraph\nend\n\nData structure representing a discrete bayesian network.\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.Factor","page":"Another page","title":"BookBayes.Factor","text":"struct Factor\n    vars::Vector{Variable}\n    table::FactorTable\nend\n\nData structure representing a discrete conditional probability distribution\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.FactorTable","page":"Another page","title":"BookBayes.FactorTable","text":"const FactorTable = Dict{NamedTuple,Float64}\n\nData structure representing a conditional probability distribution of a single variable given zero or more conditional variables.\n\nKeys: named tuples of variable names to values. Values: conditional probability\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.Variable","page":"Another page","title":"BookBayes.Variable","text":"struct Variable\n    name::Symbol\n    m::Int # number of possible values\nend\n\nData structure representing a single node in a discrete bayesian network\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.assignments-Tuple{AbstractArray{Variable,1}}","page":"Another page","title":"BookBayes.assignments","text":"function assignments(vars::AbstractVector{Variable})\n\nGet a vector of every possible valid namedtuple corresponding to a possible valid (Integer (Categorical)) value assignment of the Variables in vars.\n\nARGS\n\nEXAMPLES\n\n(:a => 3, :b => 1, :c => 7)\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.namedtuple-Tuple{Array{Symbol,1},Tuple}","page":"Another page","title":"BookBayes.namedtuple","text":"function namedtuple(n::Array{Symbol,1}, v::Tuple)\n\nCreate a NamedTuple of pairs of ::Symbol => ::Any\n\nARGS\n\nn:  Array of Symbols giving names of assignment variables\nv:  Tuple of values, one per name Symbol\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.normalize!-Tuple{Factor}","page":"Another page","title":"BookBayes.normalize!","text":"normalize!(ϕ::Factor)\n\nNormalize a factor ϕ, which divides all the entries in the factor by the same scalar so they sum to 1.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.probability-Tuple{BayesianNetwork,Any}","page":"Another page","title":"BookBayes.probability","text":"function probability(bn::BayesianNetwork, assignment)\n\nCalculate the probability of a given variable assignment given the bayesian network bn.\n\nArguments\n\nbn: The BayesianNetwork describing the joint distribution\nassignment: a NamedTuple of :Symbol=>Integer pairs giving the           categorical index value of each variable.\n\nExamples\n\njulia> X = Variable(:x, 2);\n\njulia> Y = Variable(:y, 2);\n\njulia> Z = Variable(:z, 2);\n\njulia> ϕx = Factor([X, Y, Z], FactorTable(\n           (x=1, y=1, z=1) => 0.08,\n           (x=1, y=1, z=2) => 0.31,\n           (x=1, y=2, z=1) => 0.09,\n           (x=1, y=2, z=2) => 0.37,\n           (x=2, y=1, z=1) => 0.01,\n           (x=2, y=1, z=2) => 0.05,\n           (x=2, y=2, z=1) => 0.02,\n           (x=2, y=2, z=2) => 0.07,\n       ));\n\njulia> ϕy = Factor([Y], FactorTable(\n           (y=1,) => 0.20,\n           (y=2,) => 0.80,\n       ));\n\njulia> ϕz = Factor([Z], FactorTable(\n           (z=1,) => 0.40,\n           (z=2,) => 0.60,\n       ));\n\njulia> g = SimpleDiGraph(3);\n\njulia> add_edge!(g,1,3);\n\njulia> add_edge!(g,2,3);\n\njulia> bn = BayesianNetwork([Y, Z, X], [ϕy, ϕz, ϕ], g);\n\njulia> assignment = (x=1, y=2, z=1);\n\njulia> probability(bn, assignment)\n0.028800000000000006\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.variablenames-Tuple{Factor}","page":"Another page","title":"BookBayes.variablenames","text":"variablenames(ϕ::Factor) = [var.name for var in ϕ.vars]\n\nGet Vector of the variable names of all variables in ϕ\n\n\n\n\n\n","category":"method"},{"location":"#BookBayes.jl","page":"Index","title":"BookBayes.jl","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"Documentation for BookBayes.jl","category":"page"}]
}
