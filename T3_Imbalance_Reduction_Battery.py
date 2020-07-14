from pyomo.core import value
from pyomo.environ import ConcreteModel, Param, Var, maximize, Objective, Constraint, \
    Set, Expression, Integers, Piecewise, Expr_if, PositiveIntegers, Binary, NonNegativeReals, Reals, minimize
import matplotlib.pyplot as plt
import numpy as np
from pyomo.opt import SolverFactory
import pandas as pd

'''
INITIAL PARAMETERS
capacity [MWh]
dis-/charge limits [MW]
soc min/max [MWh]/[%/soc] 
initial soc [MWh]
critical t [h]

TIME SERIES DATA
time [h]
power production(t) [MW] 
electricity price(t) [€/MWh]

MODEL
t                           time index [h]              can be None type -> T1 Battery Model
critical t                  critical time index
revenue(t)                  [€]
electricity price(t)        [€/MWh]
soc(t)                      [MWh]                       (min soc, max soc)
charging power(t)           [MW]                        (0, max charging power)
discharging power(t)        [MWh]                       (0, max discharging power)

BATTERY MODEL
battery efficiency(t) = f(charging power(t))
net output = power production(t) - charging power(t) + discharging power(t) 

soc(0) = initial soc
soc(t>0) = soc(t-1) + charging power(t)*efficiency + discharging power(t) 

soc(critical t) = soc min

charging power(t) <= power production(t)
discharging power(t) <= soc(t)

discharging(t) * charging(t) = 0
charging power(0) + discharging power(0) = 0

IMBALANCE REDUCTION MODEL
revenue = net output(t) * electricity price(t)

OBJECTIVE
max revenue
'''

"""Setting Initial Parameters"""
# [MWh]
capacity = 50
# [MW]
battery_max_charge = 48
battery_max_discharge = 48
# [MWh]
battery_SOC_min = 0.1 * capacity
battery_SOC_max = 0.9 * capacity
initial_SOC = 0.8 * capacity


def t2_rr_battery_model(bid, power_production, critical_t):
    t = len(power_production)
    # Concrete Model
    model = ConcreteModel()

    '''Time steps'''
    # [h]
    model.time = Set(initialize=range(0, t))

    '''
        Power production and electricity price

        Electricity price and (estimated) power production we take as given.

        Electricity price from the Epex Spot Market
        Power production is given by estimating the power production from weather data
    '''
    # [MW]
    model.plant_power_production = Param(model.time, initialize=dict(enumerate(power_production)), within=Reals)
    # [€/MWh]
    model.bid = Param(model.time, initialize=dict(enumerate(bid)), within=Reals)

    '''
    VARIABLES
    1. charge power(t)
    2. discharge power(t)
    3. battery soc(t)
    '''
    # [MW]
    model.charge_power = Var(model.time, bounds=(0, battery_max_charge))
    model.discharge_power = Var(model.time, bounds=(0, battery_max_discharge))

    # [MWh]
    model.battery_SOC = Var(model.time, bounds=(battery_SOC_min, battery_SOC_max))

    """EXPRESSIONS"""

    '''EFFICIENCY'''

    def efficiency_rule(m, t):  # y = R² = 0,9789
        return -3E-07 * m.charge_power[t] ** 4 + 4E-05 * m.charge_power[t] ** 3 \
               - 0.0017 * m.charge_power[t] ** 2 + 0.0291 * m.charge_power[t] + 0.7916

    model.efficiency = Expression(model.time, rule=efficiency_rule)

    '''NET OUTPUT'''

    def net_output_expression(m, t):
        # Net Output (t) [MW] = Power production (t) [MW] - Battery charge (t) [MW] + Battery discharge (t) [MW]
        return m.plant_power_production[t] - m.charge_power[t] + m.discharge_power[t]

    # [MW]
    model.net_output = Expression(model.time, rule=net_output_expression)

    '''IMBALANCE'''

    def imbalance_expression(m, t):
        # Imbalance [MW] = Net Output (t) [MW] * Bid (t) [€/MWh]
        return (m.net_output[t] - m.bid[t])**2

    # [MW]
    model.imbalance = Expression(model.time, rule=imbalance_expression)

    """CONSTRAINTS"""

    '''CALCULATES SOC'''

    def SOC_expression(m, t):
        if t == 0:
            return m.battery_SOC[t] == initial_SOC
        else:
            return m.battery_SOC[t] == m.battery_SOC[t - 1] + m.charge_power[t] * m.efficiency[t] - m.discharge_power[t]

    # [MWh]
    model.SOC_constraint = Constraint(model.time, rule=SOC_expression)

    '''GET SOC TO min AT CRITICAL t'''

    def critical_t_rule(m, t):
        if t == critical_t:
            return m.battery_SOC[t] == battery_SOC_min
        else:
            return Constraint.Skip

    model.critical_t_constr = Constraint(model.time, rule=critical_t_rule)

    '''
    CHARGING AND DISCHARGING LIMITS
    1. charge <= production
    2. discharge <= SOC
    3. Either Charge or Discharge has to be 0
    '''

    def charge_limit(m, t):
        return m.charge_power[t] <= m.plant_power_production[t]

    model.charge_limit_constr = Constraint(model.time, rule=charge_limit)

    def discharge_limit(m, t):
        return m.discharge_power[t] <= m.battery_SOC[t]

    model.discharge_limit_constr = Constraint(model.time, rule=discharge_limit)

    def charge_discharge(m, t):
        return m.discharge_power[t] * m.charge_power[t] == 0

    model.carge_discharge_constr = Constraint(model.time, rule=charge_discharge)

    '''"SETS INITIAL POWER FLOW TO 0'''

    def initial_power_flow(m, t):
        if t == 0:
            return m.discharge_power[t] + m.charge_power[t] == 0
        else:
            return Constraint.Skip

    # [MW]
    model.battery_power_flow_initial_constr = Constraint(model.time, rule=initial_power_flow)

    """Objective"""

    # revenue = max
    def obj_rule(m):
        return sum(m.imbalance[t] for t in m.time)

    model.obj = Objective(rule=obj_rule, sense=minimize)

    return model


def solve(model):
    solver = SolverFactory('ipopt')
    results= solver.solve(model, tee=True, keepfiles=True)
    model.display()

    results.write()
    return model


def output(model):
    fig = plt.figure()

    ax11 = fig.add_subplot(2, 2, 1)
    ax12 = fig.add_subplot(2, 2, 2)
    ax21 = fig.add_subplot(2, 2, 3)
    ax22 = fig.add_subplot(2, 2, 4)

    plots = [ax11, ax12, ax21, ax22]

    charge = [model.charge_power[t]() for t in model.time]
    discharge = [-model.discharge_power[t]() for t in model.time]
    eff = [model.efficiency[t]() for t in model.time]
    net = [model.net_output[t]() for t in model.time]
    print(f"charge: {charge}")
    print(f"discharge: {discharge}")
    print(f"efficiency: {eff}")

    battery_SOC = [model.battery_SOC[t]() for t in model.time]
    revenue_arr = [model.revenue[t]() for t in model.time]

    e_price_arr = [model.electricity_price[t] for t in model.time]
    pp_arr = [model.plant_power_production[t] for t in model.time]

    ax11.plot(charge, c='b')
    ax11.plot(discharge, c='orange')
    ax11.plot(eff, c='r')
    horiz_line_data = np.array([0 for i in charge])
    ax11.plot(horiz_line_data)
    ax11.legend(["charge", "discharge", "eff"])
    ax11.set_ylabel('Battery Power Flow [MW]')

    ax12.plot(pp_arr)
    ax12.set_ylabel('Power Production [MW]')

    ax21.plot(battery_SOC)
    ax21.plot(net)
    ax21.legend(["SOC", "net output"])
    ax21.set_ylabel('[MWh]')

    ax22.plot(e_price_arr)
    ax22.set_ylabel('Electricity Price [€/MWh]')

    for ax in plots:
        ax.grid()

    print(f"{sum(revenue_arr)}")

    plt.show()
