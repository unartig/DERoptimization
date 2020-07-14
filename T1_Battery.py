from pyomo.environ import ConcreteModel, Param, Var, maximize, Objective, Constraint, \
    Set, Expression, NonNegativeReals
import matplotlib.pyplot as plt
import numpy as np
from pyomo.opt import SolverFactory
import pandas as pd

'''
power losse indexed by (time) and (power loss set)
battery soc(t) = battery soc(t-1) + (battery power flow (t) - power loss(t)(battery power flow))  

MODEL

TIME SERIES DATA
time [h]
power production(t) [MW] 
electricity price(t) [€/MWh]

TIME SERIES VARIABLES
-max discharge < battery power flow(t) [MW] < max charge
min SOC < battery SOC(t) [MWh] < max SOC
net output(t) [MW] = power production(t) - battery power flow(t)
revenue(t) [€] = net output(t) * electricity price[t]

CONSTRAINTS

battery soc
if t is not 0
battery soc(t) = battery soc(t-1) + battery power flow (t)  
else
battery soc(0) = initial soc

battery power flow
if t is not 0
battery power flow(t) <= power production(t) 
else 
battery power flow(0) = 0

OBJECTIVE
max revenue
'''

# [MWh]
capacity = 50
# [MW]
battery_max_charge = 48
battery_max_discharge = 48
# [MWh]
battery_SOC_min = 0.1 * capacity
battery_SOC_max = 0.9 * capacity
initial_SOC = 0.1 * capacity
# [h]
t = 24

data = pd.read_excel('tagessatz.xlsx')

power_production = np.array([0.523, 0.92, 0.523, 1.471, 1.471, 1.471, 2.867, 3.481, 3.481, 3.903, 3.903, 3.481, 3.903,
                             2.867, 3.481, 2.867, 2.867, 3.481, 3.481, 2.151, 2.151, 3.481, 3.903, 3.903])
power_production = np.array([5 * p for p in power_production])
electricity_price = np.array(data['[€/MWh]'])


def t1_battery_model(power_production, electricity_price):
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
    model.plant_power_production = Param(model.time, initialize=dict(enumerate(power_production)))
    # [€/MWh]
    model.electricity_price = Param(model.time, initialize=dict(enumerate(electricity_price)))

    '''
        Battery Model
        
        Model variables battery flow [MW] and SOC [MWh] at a given time t  
    '''
    # [MW]
    model.battery_power_flow = Var(model.time, bounds=(-battery_max_discharge, battery_max_charge))

    # [MWh]
    model.battery_SOC = Var(model.time, bounds=(battery_SOC_min, battery_SOC_max))

    model.power_loss = Var(model.time, within=NonNegativeReals)

    # [MWh]
    def power_loss_expression(m, t):
        return m.power_loss[t] == 0.0022*m.battery_power_flow[t]**2 - 0.0233*m.battery_power_flow[t] + 0.4887
    model.power_loss_constr = Constraint(model.time, rule=power_loss_expression)

    '''
        Net Output
        
        Net output [MW] at a given time t
    '''
    def net_output_expression(m, t):
        # Net Output (t) [MW] = Power production (t) [MW] - Battery power flow (t) [MW]
        return m.plant_power_production[t] - m.battery_power_flow[t]
    # [MW]
    model.net_output = Expression(model.time, rule=net_output_expression)

    '''
        Revenue
    
        Revenue [€] at given time t
    '''
    def revenue_expression(m, t):
        # Revenue [€] = Net Output (t) [MW] * Electricity Price (t) [€/MWh]
        return m.net_output[t] * m.electricity_price[t]
    # [€]
    model.revenue = Expression(model.time, rule=revenue_expression)

    '''Constraints'''
    def SOC_expression(m, t):
        if t is not 0:
            return m.battery_SOC[t] == m.battery_SOC[t - 1] + m.battery_power_flow[t] - m.power_loss[t]
        else:
            return m.battery_SOC[t] == initial_SOC
    # [MWh]
    model.SOC_constraint = Constraint(model.time, rule=SOC_expression)

    def battery_power_flow_initial(m, t):
        if t is not 0:
            return m.battery_power_flow[t] <= m.plant_power_production[t]
        else:
            return m.battery_power_flow[t] == 0
    # [MW]
    model.battery_power_flow_initial_constr = Constraint(model.time, rule=battery_power_flow_initial)


    '''Objective'''
    # revenue = max
    def obj_rule(m):
        return sum(m.revenue[t] for t in m.time)
    model.obj = Objective(rule=obj_rule, sense=maximize)

    return model


m = t1_battery_model()
solver = SolverFactory('ipopt')
results = solver.solve(m)


def plots():
    fig = plt.figure()

    ax11 = fig.add_subplot(2, 2, 1)
    ax12 = fig.add_subplot(2, 2, 2)
    ax21 = fig.add_subplot(2, 2, 3)
    ax22 = fig.add_subplot(2, 2, 4)

    plots = [ax11, ax12, ax21, ax22]
    battery_power_flow = [m.battery_power_flow[t]() for t in m.time]
    print(battery_power_flow)
    battery_SOC = [m.battery_SOC[t]() for t in m.time]
    print(battery_SOC)
    revenue_arr = [m.revenue[t]() for t in m.time]
    print(sum(revenue_arr))
    loss_arr = [m.power_loss[t]() for t in m.time]

    ax11.plot(battery_power_flow)
    ax11.plot(loss_arr)
    horiz_line_data = np.array([0 for i in battery_power_flow])
    ax11.plot(horiz_line_data)
    ax11.set_ylabel('Battery Power Flow [MW]')

    ax12.plot(power_production)
    ax12.set_ylabel('Power Production [MW]')

    ax21.plot(battery_SOC)
    ax21.set_ylabel('Battery_SOC [MWh]')

    ax22.plot(electricity_price)
    ax22.set_ylabel('Electricity Price [€/MWh]')

    for plot in plots:
        plot.grid()

    plt.show()


plots()
