

using BookBayes
using LightGraphs
using LinearAlgebra
G = SimpleDiGraph(3);

add_edge!(G, 1,2);

add_edge!(G, 3,2);

vars = [Variable(:A, 2), Variable(:B, 2), Variable(:C, 2)];

D = [1 2 2 1; 1 2 2 1; 2 2 2 2];

M = statistics(vars, G, D)

# Compute the maximum likelihood estimate by normalizing the rows in the matricies in M:
θ = [mapslices(x->LinearAlgebra.normalize(x,1), Mi, dims=2)
        for Mi in M]
