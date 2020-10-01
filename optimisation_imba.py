from pyomo.environ import AbstractModel, Param, Var, maximize, Objective, Constraint, \
    Set, Expression, Block, Reals, Any, Integers
from pyomo.opt import SolverFactory


class IMBAOptimisation:

    def __init__(self, electricity_price, power_production, der_list, bid, der1charge, der1dcharge, der2charge, der2dcharge):

        self.der1 = der1charge, der1dcharge
        self.der2 = der2charge, der2dcharge
        self.der_list = der_list
        self.model = self.create_model(electricity_price, power_production, bid)

    def create_model(self, electricity_price, power_production, bid):
        # create abstract model
        model = AbstractModel()

        # get length of the time interval
        t = len(power_production)

        '''VARIABLE INITIALISATION'''
        # [h]
        model.time = Set(initialize=range(0, t))
        # number of ders
        model.ders = Set(initialize=range(0, len(self.der_list)))

        '''DATA INITIALISATION'''
        # [MW]
        model.plant_power_production = Param(model.time, initialize=dict(enumerate(power_production)), within=Reals)
        # [€/MWh]
        model.electricity_price = Param(model.time, initialize=dict(enumerate(electricity_price)), within=Reals)
        # [MW]
        model.bid = Param(model.time, initialize=dict(enumerate(bid)), within=Reals)

        '''DER INITIALISATION'''
        model.der = Block(model.ders, rule=self.create_abstract_der)

        """CONSTRAINTS"""

        def charge_power_sum_rule(m, t):
            return sum(m.der[der].charge_power[t] for der in m.ders)
        model.charge_power_sum = Expression(model.time, rule=charge_power_sum_rule)

        def discharge_power_sum_rule(m, t):
            return sum(m.der[der].discharge_power[t] for der in m.ders)
        model.discharge_power_sum = Expression(model.time, rule=discharge_power_sum_rule)

        def net_flow_expression(m, t):
            # Net Output (t) [MW] = Power production (t) [MW] - Battery charge (t) [MW] + Battery discharge (t) [MW]
            return m.plant_power_production[t] + m.discharge_power_sum[t] - m.charge_power_sum[t]
        # [MW]
        model.net_flow = Expression(model.time, rule=net_flow_expression)


        def obj_rule(m):
            return sum(abs(m.net_flow[t]-m.bid[t]) for t in m.time)
        model.obj = Objective(rule=obj_rule)

        return model

    def create_abstract_der(self, i, ders):

        data = self.der_list[ders][None]

        if ders == 0:
            p = self.der1
        else:
            p = self.der2

        i.time = Set(within=Reals, initialize=data['time'])
        # [MW]
        i.charge_power = Var(i.time, bounds=data['charge_power'], initialize=dict(enumerate(p[0])))
        i.discharge_power = Var(i.time, bounds=data['discharge_power'], initialize=dict(enumerate(p[1])))

        # [MWh]
        i.soc = Var(i.time)

        i.corridor_lower_bound = Param(i.time, within=Reals, initialize=data['corridor_lower_bound'])
        i.corridor_upper_bound = Param(i.time, within=Reals, initialize=data['corridor_upper_bound'])

        def efficiency_rule(i, t):
            return 1
        i.efficiency = Expression(i.time, rule=efficiency_rule)

        def discharge_limit(i, t):
            return i.discharge_power[t] <= i.soc[t] - i.corridor_lower_bound[t]
        i.discharge_limit_constr = Constraint(i.time, rule=discharge_limit)

        def charge_discharge(i, t):
            return i.discharge_power[t] * i.charge_power[t] == 0
        i.charge_discharge_constr = Constraint(i.time, rule=charge_discharge)

        def stay_in_corridor(i, t):
            return i.corridor_lower_bound[t], i.soc[t], i.corridor_upper_bound[t]
        i.corridor_constr = Constraint(i.time, rule=stay_in_corridor)

        def soc_inital(i, t):
            if t == i.time.first():
                return i.soc[t] == data['initial_soc'] * i.corridor_upper_bound[t]
            if t == i.time.last():
                return i.soc[t] == i.corridor_lower_bound[t] + data['initial_soc'] * (i.corridor_upper_bound[t] - i.corridor_lower_bound[t])
            else:
                return Constraint.Skip
        # [MWh]
        i.soc_inital_constraint = Constraint(i.time, rule=soc_inital)

        def soc_expression(i, t):
            if t != i.time.first():
                return i.soc[t] == i.soc[t - 1] + i.charge_power[t] * i.efficiency[t] - i.discharge_power[t]
            else:
                return Constraint.Skip
            # [MWh]
        i.soc_constraint = Constraint(i.time, rule=soc_expression)

        def initial_power_flow(i, t):
            if t == i.time.first():
                return i.discharge_power[t] + i.charge_power[t] == 0
            else:
                return Constraint.Skip
        # [MW]
        i.battery_power_flow_initial_constr = Constraint(i.time, rule=initial_power_flow)

    def solve(self):
        instance = self.model.create_instance()
        solver = SolverFactory('scip')
        solver.solve(instance, tee=True)
        return instance

    def print_model(self):
        instance = self.model.create_instance()
        instance.pprint()
