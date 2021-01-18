using Test, BookBayes




X = Variable(:x, 2)
Y = Variable(:y, 2)
Z = Variable(:z, 2)

ϕ = Factor([X, Y, Z], FactorTable(
    (x=1, y=1, z=1) => 0.08,
    (x=1, y=1, z=2) => 0.31,
    (x=1, y=2, z=1) => 0.09,
    (x=1, y=2, z=2) => 0.37,
    (x=2, y=1, z=1) => 0.01,
    (x=2, y=1, z=2) => 0.05,
    (x=2, y=2, z=1) => 0.02,
    (x=2, y=2, z=2) => 0.07,
))





k = assignments(ϕ.vars)
@show k
@show k[2,1,1]
@show k[2,1,1].x
@show k[2,1,1].y
@show k[2,1,1].z


display(ϕ.table)

display(ϕ.vars)

display(variablenames(ϕ))


@test true
