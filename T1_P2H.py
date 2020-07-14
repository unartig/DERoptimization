from pyomo.environ import ConcreteModel, Param, Var, Boolean, value, maximize, Objective, Constraint, \
    Set, inequality, Expression
import matplotlib.pyplot as plt
import numpy as np
from pyomo.opt import SolverFactory

# T1: Jetzt Basismodell mit a und b ohne c!
# mit P2H ohne Risiko: Erlösoptimierung => Fahrplan des Betriebs der P2H
# Mathe/Grafik => Hendrik Obelöers Basisdokument (R)

"""
Simple Economical Model of a P2H Facility

Input: Power Production (DER) and Electricity Prices [t]

Output: Optimal scheduling for the P2H
"""


total_consumption = 24
max_consumption = 3
p2h_cost = 3

power_production = np.array([10, 9,  9,  8,  10, 11, 12, 13, 10, 9, 10, 9,  9,  8,  10, 11, 12, 13, 10, 9, 8, 8, 7, 8])
electricity_price = np.array([1,  2,  3,  4,  5,  5,  5,  5,  4,  4, 1,  2,  3,  4,  5,  5,  5,  5,  4,  4, 6, 7, 8, 1])

def t1_p2h_model():
    # Concrete Model
    model = ConcreteModel()

    '''Time steps'''
    # [h]
    model.time = Set(initialize=range(0, 24))

    '''Power production and electricity price'''
    # [MWh]
    model.power_production = Param(model.time, initialize=dict(enumerate(power_production)))
    # [€/MWh]
    model.electricity_price = Param(model.time, initialize=dict(enumerate(electricity_price)))

    '''Power to Heat Model'''
    # [€/MWh]
    model.p2h_cost = Param(initialize=p2h_cost)
    # [MWh]
    model.p2h_power_consumption = Var(model.time, bounds=(0, max_consumption))

    '''Revenue'''
    # [€]
    def revenue_rule(m, t):
        return (m.power_production[t] - m.p2h_power_consumption[t]) * m.electricity_price[t]
    model.revenue = Expression(model.time, rule=revenue_rule)

    '''Constraints'''
    # sum of power consumption = x
    def p2h_total_consumption_rule(m):
        return sum(m.p2h_power_consumption[t] for t in m.time) == total_consumption
    model.p2h_total_consumption_constr = Constraint(rule=p2h_total_consumption_rule)

    # consider max p2h power
    def p2h_max_consumption1(m, t):
        # 0 <= max <= power production
        return inequality(0, m.p2h_power_consumption[t], m.power_production[t])
    model.p2h_max_consumption_constr = Constraint(model.time, rule=p2h_max_consumption1)

    '''Objective'''
    # revenue = max
    def obj_rule(m):
        return sum(m.revenue[t] for t in m.time)
    model.obj = Objective(rule=obj_rule, sense=maximize)

    return model


m = model_()
solver = SolverFactory('ipopt')
results = solver.solve(m)


def plots():
    fig = plt.figure()

    ax11 = fig.add_subplot(2, 2, 1)
    ax12 = fig.add_subplot(2, 2, 2)
    ax21 = fig.add_subplot(2, 2, 3)
    ax22 = fig.add_subplot(2, 2, 4)

    p2h_power_consumption_arr = [m.p2h_power_consumption[t]() for t in m.time]
    print('Consumed Energy by p2h: ' + str(sum(p2h_power_consumption_arr)) + 'W')
    revenue_arr = [m.revenue[t]() for t in m.time]
    print('Earned revenue: ' + str(sum(revenue_arr)) + '€')
    print(revenue_arr)

    ax11.plot(p2h_power_consumption_arr)
    ax11.set_ylabel('Power Consumption [MWh]')
    ax12.plot(power_production)
    ax12.set_ylabel('Power Production [MWh]')
    ax21.plot(revenue_arr)
    ax21.set_ylabel('Earnings [€]')
    ax22.plot(electricity_price)
    ax22.set_ylabel('Electricity Price [€/MWh]')
    plt.show()


plots()
