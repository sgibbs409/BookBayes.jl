var documenterSearchIndex = {"docs":
[{"location":"anotherPage/#The-BookBayes-Module","page":"Another page","title":"The BookBayes Module","text":"","category":"section"},{"location":"anotherPage/","page":"Another page","title":"Another page","text":"BookBayes","category":"page"},{"location":"anotherPage/#BookBayes","page":"Another page","title":"BookBayes","text":"module BookBayes\n\nLibrary of functions given in Algorithms for Decision Making\n\n\n\n\n\n","category":"module"},{"location":"anotherPage/#Module-Index","page":"Another page","title":"Module Index","text":"","category":"section"},{"location":"anotherPage/","page":"Another page","title":"Another page","text":"Modules = [BookBayes]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"anotherPage/#Detailed-API","page":"Another page","title":"Detailed API","text":"","category":"section"},{"location":"anotherPage/","page":"Another page","title":"Another page","text":"Modules = [BookBayes]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"anotherPage/#BookBayes.BayesianNetwork","page":"Another page","title":"BookBayes.BayesianNetwork","text":"struct BayesianNetwork\n\nRepresents a Bayesian Network.\n\nFields:\n\nvars::Vector{Variable}:\n\nVector of Variables contained in network\n\nfactors::Vector{Factor}:\n\nVector of Factors, one for each Variable in vars, representing conditional probability distribution over that variable given parents of var in network.\n\ngraph::SimpleDiGraph{Int64}:\n\nRepresents the structure of the network variables and implicitly defines independence relationships.\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.ExactInference","page":"Another page","title":"BookBayes.ExactInference","text":"struct ExactInference\n\nSingleton type to pass instances of as first arg of infer to trigger direct inference method.\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.Factor","page":"Another page","title":"BookBayes.Factor","text":"struct Factor\n\nA Type representing a joint or conditional probability distribution over the variables in vars.\n\nFields:\n\nvars:\n\nThe variables represented by this distribution\n\ntable:\n\nProbability table mapping Variable assignments (NamedTuple) => probability (Float64)\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.FactorTable","page":"Another page","title":"BookBayes.FactorTable","text":"const FactorTable = Dict{NamedTuple,Float64}\n\nType alias of a Dictionary mapping tuples of Variable name=>value assignments to probabilities.\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.Variable","page":"Another page","title":"BookBayes.Variable","text":"struct Variable\n\nA Type representing a single discrete variable.\n\nParams:\n\nname::Symbol\n\nname of Variable\n\nm::Int:\n\nnumber of possible (discrete) values this Variable can take\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#BookBayes.VariableElimination","page":"Another page","title":"BookBayes.VariableElimination","text":"struct VariableElimination\n\nSingleton type to pass instances of as first arg of infer to trigger variable elimination method.\n\nordering\n\nIterable over variable indices indicating the order over the factors to use.\n\n\n\n\n\n","category":"type"},{"location":"anotherPage/#Base.:*-Tuple{Factor,Factor}","page":"Another page","title":"Base.:*","text":"function Base.:*(ϕ::Factor, ψ::Factor)\n\nMultiply two (joint and conditional) Factors to form a new (joint) Factor.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#Base.rand-Tuple{BayesianNetwork}","page":"Another page","title":"Base.rand","text":"function Base.rand(bn::BayesianNetwork)\n\nDirect sample from a joint distribution given by bn.\n\nReturns: NamedTuple of Variable name => value pairs sampled from bn.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#Base.rand-Tuple{Factor}","page":"Another page","title":"Base.rand","text":"function Base.rand(φ::Factor)\n\nDirect sample a discrete Factor  given by φ\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.assignments-Tuple{AbstractArray{Variable,1}}","page":"Another page","title":"BookBayes.assignments","text":"function assignments(vars::AbstractVector{Variable})\n\nGet an array of all possible variable=>value assignment tuples for Variables in vars.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.in_scope-Tuple{Any,Any}","page":"Another page","title":"BookBayes.in_scope","text":"function in_scope(name, ϕ)\n\npredicate function: true if any Variable v in ϕ has v.name == name (ie if name is a variable in ϕ)\n\nArguments\n\nname::Symbol: Name of variable to test for presense of in Factor ϕ.\nϕ::Factor: Factor to test if variable named <name> is present in.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.infer-Tuple{DirectSampling,Any,Any,Any}","page":"Another page","title":"BookBayes.infer","text":"function infer(M::DirectSampling, bn, query, evidence)\n\nGet a joint distribution over query variables, given evidence.  Uses the direct sampling inference method to draw m samples from network consistent with evidence.\n\nFields:\n\nM::DirectSampling\n\nSingleton value to dispatch this version of infer\n\nbn\n\nBayesianNetwork specifying original distribution\n\nquery\n\nTuple of Symbols of Variable names to get joint distribution over.\n\nevidence\n\nDictionary of Variable Symbol name => Variable value pairs specifying known values.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.infer-Tuple{Distributions.MvNormal,Any,Any,Any}","page":"Another page","title":"BookBayes.infer","text":"function infer(D::MvNormal, query, evidencevars, evidence)\n\nGet the conditional normal distribution over the query variables, given the evidence variables\n\nleft\nbeginarrayc\n   bf a \n   bf b\nendarray\nright sim mathcalN left( left\nbeginarrayc\n  bf mu_a \n  bf mu_b\nendarray\nright  left\nbeginarraycc\n  bf A  bf C \n  bf C^T  bf B\nendarray\nright right)\n\nThe conditional distribution is:\n\np(textbfa  textbfb) = mathcalN( bf a   mu_ab Sigma_ab)\n\nboxedbf mu_ab = mu_a + CB^-1(b - mu_b)\n\nboxedSigma_ab = A - CB^-1C^T\n\nExamples\n\nConsider:\n\nleft\nbeginarrayc\n   x_1 \n   x_2\nendarray\nright sim mathcalN left( left\nbeginarrayc\n  0 \n  1\nendarray\nright  left\nbeginarraycc\n  3  1 \n  1  2\nendarray\nright right)\n\nThen the conditional distribution for x₁ given x₂ = 2 is:\n\nmu_x_1x_2=2 = 0 + 1 cdot 2^-1 cdot (2 - 1) = 05\n\nSigma_x_1x_2=2 = 3 - 1 cdot 2^-1cdot 1 = 25\n\njulia>  d = MvNormal([0.0,1.0], [3.0 1.0; 1.0 2.0])\nd = FullNormal(\n\tdim: 2\n\tμ: [0.0, 1.0]\n\tΣ: [3.0 1.0; 1.0 2.0]\n\t)\n\n# get conditional dist for x₁ given x₂ = 2.0\njulia> infer(d, [1], [2], [2.0])\nFullNormal(\ndim: 1\nμ: [0.5]\nΣ: [2.5]\n)\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.infer-Tuple{ExactInference,Any,Any,Any}","page":"Another page","title":"BookBayes.infer","text":"function infer(M::ExactInference, bn, query, evidence)\n\nGet a joint distribution over query variables, given evidence.  Uses direct inference.\n\nFields:\n\nM::ExactInference\n\nSingleton value to use this version of infer\n\nbn\n\nBayesianNetwork specifying original distribution\n\nquery\n\nTuple of Symbols of Variable names to get joint distribution over.\n\nevidence\n\nDictionary of Variable Symbol name => Variable value pairs specifying known values.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.infer-Tuple{VariableElimination,Any,Any,Any}","page":"Another page","title":"BookBayes.infer","text":"function infer(M::VariableElimination, bn, query, evidence)\n\nGet a joint distribution over query variables, given evidence.  Uses the sum-product variable elimination algorithm.\n\nFields:\n\nM::VariableElimination\n\nSingleton value to dispatch this version of infer\n\nbn\n\nBayesianNetwork specifying original distribution\n\nquery\n\nTuple of Symbols of Variable names to get joint distribution over.\n\nevidence\n\nDictionary of Variable Symbol name => Variable value pairs specifying known values.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.marginalize-Tuple{Factor,Any}","page":"Another page","title":"BookBayes.marginalize","text":"function marginalize(ϕ::Factor, name)\n\nIntegrate ϕ::Factor by Variable with name name to get new Factor with marginal distribution over remaining Variables.\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.normalize!-Tuple{Factor}","page":"Another page","title":"BookBayes.normalize!","text":"function normalize!(φ::Factor)\n\nUpdate φ::Factor so all values sum to 1.0 while maintaining relative value (so values represent probabilities).\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.probability-Tuple{BayesianNetwork,Any}","page":"Another page","title":"BookBayes.probability","text":"function probability(bn::BayesianNetwork, assignment)\n\nCalculate the probability of a given variable assignment given the bayesian network bn.\n\nArguments\n\nbn: The BayesianNetwork describing the joint distribution\nassignment: a NamedTuple of :Symbol=>Integer pairs giving the           categorical index value of each variable.\n\nExamples\n\njulia> X = Variable(:x, 2);\n\njulia> Y = Variable(:y, 2);\n\njulia> Z = Variable(:z, 2);\n\njulia> ϕx = Factor([X, Y, Z], FactorTable(\n           (x=1, y=1, z=1) => 0.08,\n           (x=1, y=1, z=2) => 0.31,\n           (x=1, y=2, z=1) => 0.09,\n           (x=1, y=2, z=2) => 0.37,\n           (x=2, y=1, z=1) => 0.01,\n           (x=2, y=1, z=2) => 0.05,\n           (x=2, y=2, z=1) => 0.02,\n           (x=2, y=2, z=2) => 0.07,\n       ));\n\njulia> ϕy = Factor([Y], FactorTable(\n           (y=1,) => 0.20,\n           (y=2,) => 0.80,\n       ));\n\njulia> ϕz = Factor([Z], FactorTable(\n           (z=1,) => 0.40,\n           (z=2,) => 0.60,\n       ));\n\njulia> g = SimpleDiGraph(3);\n\njulia> add_edge!(g,1,3);\n\njulia> add_edge!(g,2,3);\n\njulia> bn = BayesianNetwork([Y, Z, X], [ϕy, ϕz, ϕ], g);\n\njulia> assignment = (x=1, y=2, z=1);\n\njulia> probability(bn, assignment)\n0.028800000000000006\n\n\n\n\n\n","category":"method"},{"location":"anotherPage/#BookBayes.variablenames-Tuple{Factor}","page":"Another page","title":"BookBayes.variablenames","text":"function variablenames(φ::Factor)\n\nGet a vector of variable names (Symbols) for the Variables in a Factor\n\n\n\n\n\n","category":"method"},{"location":"#BookBayes.jl","page":"Index","title":"BookBayes.jl","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"Documentation for BookBayes.jl","category":"page"}]
}
