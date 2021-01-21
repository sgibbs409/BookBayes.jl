

using BookBayes
using LightGraphs

G = SimpleDiGraph(3);

add_edge!(G, 1,2);

add_edge!(G, 3,2);

vars = [Variable(:A, 2), Variable(:B, 2), Variable(:C, 2)];

D = [1 2 2 1; 1 2 2 1; 2 2 2 2];

M = statistics(vars, G, D)
