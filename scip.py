from pyomo.environ import ConcreteModel, Param, Var, maximize, Objective, Constraint, \
    Set, Expression, Piecewise, Expr_if, Block,  PositiveIntegers, Binary, NonNegativeReals, Reals, Block
from pyomo.opt import SolverFactory

model = ConcreteModel()

index = ["male","female","baby"]
price = {index[0]:1, index[1]:2, index[2]:0.5}
budget = 100

model.i = Set(initialize=index)
model.chicken = Var(index, within=PositiveIntegers, initialize=1)

def objective_rule(m):
    return sum(m.chicken[index] for index in m.i)
model.obj = Objective(rule=objective_rule, sense=maximize)

def max_budget(m, i):
    return sum(m.chicken[index]*price[index] for index in m.i) <= budget
model.max_budget = Constraint(model.i, rule=max_budget)


solver = SolverFactory("ipopt")
solver.solve(model)
print(model)
#print(result.chicken)