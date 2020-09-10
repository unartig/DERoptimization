from pyomo.core import value
from pyomo.environ import AbstractModel, Param, Var, maximize, Objective, Constraint, \
    Set, Expression, Integers, Piecewise, Expr_if, PositiveIntegers, Binary, NonNegativeReals, Reals, Block, \
    ConcreteModel
import matplotlib.pyplot as plt
import numpy as np
from pyomo.opt import SolverFactory
import pandas as pd


"""Setting Initial Parameters"""
# [MWh]
capacity = 50
# [MW]
battery_max_charge = 48
battery_max_discharge = 0
# [MWh]
# this will be changed to an time series containing the corrdior
battery_SOC_min = 0.1 * capacity
battery_SOC_max = 0.9 * capacity

initial_SOC = 0.8 * capacity


class DEROptimisation:

    def __init__(self, electricity_price, power_production, der_list):

        self.der_list = der_list
        self.model = self.create_model(electricity_price, power_production)

    def create_model(self, electricity_price, power_production):
        # create abstract model
        model = AbstractModel()

        # get length of the time interval
        t = len(power_production)

        '''VARIABLE INITIALISATION'''
        # [h]
        model.time = Set(initialize=range(0, t))
        model.ders = Set(initialize=range(0, len(self.der_list)))

        '''DATA INITIALISATION'''
        # [MW]
        model.plant_power_production = Param(model.time, initialize=dict(enumerate(power_production)), within=Reals)
        # [€/MWh]
        model.electricity_price = Param(model.time, initialize=dict(enumerate(electricity_price)), within=Reals)

        '''DER INITIALISATION'''
        model.blocks = Block(model.ders, rule=self.create_abstract_der)

        """CONSTRAINTS"""
        def charge_power_sum_rule(m, t):
            return sum(m.blocks[der].charge_power[t] for der in m.ders)
        model.charge_power_sum = Expression(model.time, rule=charge_power_sum_rule)

        def discharge_power_sum_rule(m, t):
            return sum(m.blocks[der].discharge_power[t] for der in m.ders)
        model.discharge_power_sum = Expression(model.time, rule=charge_power_sum_rule)

        def charge_limit(m, t):
            return m.charge_power_sum[t] <= m.plant_power_production[t]
        model.charge_limit_constr = Constraint(model.time, rule=charge_limit)

        def net_output_expression(m, t):
            # Net Output (t) [MW] = Power production (t) [MW] - Battery charge (t) [MW] + Battery discharge (t) [MW]
            return m.plant_power_production[t] - m.charge_power_sum[t] + m.discharge_power_sum[t]
        # [MW]
        model.net_output = Expression(model.time, rule=net_output_expression)

        def revenue_expression(m, t):
            # Revenue [€] = Net Output (t) [MW] * Electricity Price (t) [€/MWh]
            return m.net_output[t] * m.electricity_price[t]
        # [€]
        model.revenue = Expression(model.time, rule=revenue_expression)

        return model

    def create_abstract_der(self, i, ders):

        data = self.der_list[ders][None]

        i.time = Set(within=Reals, initialize=data['time'])
        # [MW]
        i.charge_power = Var(i.time, bounds=data['charge_power'], initialize=data['soc_initial'])
        i.discharge_power = Var(i.time, bounds=data['discharge_power'])

        # [MWh]
        i.soc = Var(i.time)

        i.corridor_lower_bound = Param(i.time, within=Reals, initialize=data['corridor_lower_bound'])
        i.corridor_upper_bound = Param(i.time, within=Reals, initialize=data['corridor_upper_bound'])

        def efficiency_rule(m, t):
            return 1
        i.efficiency = Expression(i.time, rule=efficiency_rule)

        def discharge_limit(i, t):
            return i.discharge_power[t] <= i.soc[t]
        i.discharge_limit_constr = Constraint(i.time, rule=discharge_limit)

        def charge_discharge(i, t):
            return i.discharge_power[t] * i.charge_power[t] == 0
        i.charge_discharge_constr = Constraint(i.time, rule=charge_discharge)

        def stay_in_corridor(i, t):
            return i.corridor_lower_bound[t], i.soc[t], i.corridor_upper_bound[t]
        i.corridor_constr = Constraint(i.time, rule=stay_in_corridor)

        def soc_expression(m, t):
            if t == i.time.first():
                return i.soc[t] == data['initial_soc'] * i.corridor_upper_bound[t]
            if t == i.time.last():
                return i.soc[t] == i.corridor_lower_bound[t] + data['initial_soc'] * (i.corridor_upper_bound[t] - i.corridor_lower_bound[t])
            else:
                return i.soc[t] == m.soc[t - 1] + i.charge_power[t] * i.efficiency[t] - i.discharge_power[t]
        # [MWh]
        i.soc_constraint = Constraint(i.time, rule=soc_expression)

        def initial_power_flow(m, t):
            if t == i.time.first():
                return i.discharge_power[t] + i.charge_power[t] == 0
            else:
                return Constraint.Skip
        # [MW]
        i.battery_power_flow_initial_constr = Constraint(i.time, rule=initial_power_flow)

    def solve(self):
        instance = self.model.create_instance()
        solver = SolverFactory('ipopt')
        #solver.set_options("max_iter=")
        solver.solve(instance)
        return instance

    def print_model(self):
        instance = self.model.create_instance()
        instance.pprint()
