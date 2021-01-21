

using BookBayes
using LightGraphs

C = Variable(:c, 3);
S = Variable(:s, 3);
V = Variable(:v, 3);

# P(C)
φC = Factor([C], FactorTable(
    (c=1,) => 0.80,
    (c=2,) => 0.19,
    (c=3,) => 0.01,
    ));
# P(S|C)
φSC = Factor([S,C], FactorTable(
    (s=1,c=1,) => 0.001,
    (s=1,c=2,) => 0.200,
    (s=1,c=3,) => 0.800,
    (s=2,c=1,) => 0.009,
    (s=2,c=2,) => 0.750,
    (s=2,c=3,) => 0.199,
    (s=3,c=1,) => 0.990,
    (s=3,c=2,) => 0.050,
    (s=3,c=3,) => 0.001,
    ));

# P(V|C)
φVC = Factor([V,C], FactorTable(
    (v=1,c=1,) => 0.2,
    (v=1,c=2,) => 0.5,
    (v=1,c=3,) => 0.4,
    (v=2,c=1,) => 0.2,
    (v=2,c=2,) => 0.4,
    (v=2,c=3,) => 0.4,
    (v=3,c=1,) => 0.6,
    (v=3,c=2,) => 0.1,
    (v=3,c=3,) => 0.2,
    ));

g = SimpleDiGraph(3)

add_edge!(g, 1, 2)
add_edge!(g, 1, 3);

bn = BayesianNetwork([C,S,V], [φC, φSC, φVC], g);

m_weighted = LikelihoodWeightedSampling(100000);
m_var_elim = VariableElimination(3:1);

r_exact= infer(ExactInference(), bn, (:c,), (s=2, v=1))
r_weighted = infer(m_weighted, bn, (:c,), (s=2,v=1,))
r_var_elim = infer(m_var_elim, bn, (:c,), (s=2,v=1,))
