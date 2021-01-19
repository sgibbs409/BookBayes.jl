using Test, BookBayes




X = Variable(:x, 2);
Y = Variable(:y, 2);
Z = Variable(:z, 2);
ϕx = Factor([X, Y, Z], FactorTable(
    (x=1, y=1, z=1) => 0.08,
    (x=1, y=1, z=2) => 0.31,
    (x=1, y=2, z=1) => 0.09,
    (x=1, y=2, z=2) => 0.37,
    (x=2, y=1, z=1) => 0.01,
    (x=2, y=1, z=2) => 0.05,
    (x=2, y=2, z=1) => 0.02,
    (x=2, y=2, z=2) => 0.07,
));
ϕy = Factor([Y], FactorTable(
    (y=1,) => 0.20,
    (y=2,) => 0.80,
));
ϕz = Factor([Z], FactorTable(
    (z=1,) => 0.40,
    (z=2,) => 0.60,
));
g = SimpleDiGraph(3);
add_edge!(g,1,3);
add_edge!(g,2,3);
bn = BayesianNetwork([Y, Z, X], [ϕy, ϕz, ϕ], g);
assignment = (x=1, y=2, z=1);
probability(bn, assignment)



k = assignments(ϕ.vars);
@show k



@show k[2,1,1]
@show k[2,1,1].x
@show k[2,1,1].y
@show k[2,1,1].z

display(ϕ.table);
display(ϕ.vars);
display(variablenames(ϕ));


A = Variable(:A, 3);
ϕA = Factor([A], FactorTable(
    (a=1,) => 0.40,
    (a=2,) => 0.50,
    (a=3,) => 0.1,
));

ϕAy = ϕA * ϕy
ϕyz = ϕy * ϕz;
ϕAyz = ϕAy * ϕyz












[v for (k,v) in ϕAyz.table]
sum([v for (k,v) in ϕAyz.table])

sum([v for (k,v) in ϕAy.table])
sum([v for (k,v) in ϕyz.table])






@test true
