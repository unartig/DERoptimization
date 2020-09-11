from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimisation import DEROptimisation


def list_to_dict(list):

    dict = {i: list[i] for i in range(0, len(list))}

    return dict


def create_linear_corridor(slope, width, length):

    # slope = (upper, lower)
    lower = [0 + slope[0] * x for x in range(0, length)]
    upper = [width + slope[1] * x for x in range(0, length)]

    return lower, upper


def get_corridor_center(corridor):

    # center = lower + (upper - lower)
    corridor_center = [corridor[0][i] + (corridor[1][i] - corridor[0][i]) for i in range(0, len(corridor[0]))]

    return corridor_center


def plot_der_corridors(number_of_ders):
    fig = plt.figure()

    for der in range(0, number_of_ders):
        lower = [result.der[der].corridor_lower_bound[t] for t in result.time]
        upper = [result.der[der].corridor_upper_bound[t] for t in result.time]
        soc = [result.der[der].soc[t]() for t in result.time]

        plot = fig.add_subplot(number_of_ders, 1, der + 1)

        plot.plot(lower)
        plot.plot(upper)
        plot.plot(soc)

        plot.legend(["lower", "upper", "actual energy"])

def plot_power_flows():

    fig = plt.figure()

    plot = fig.add_subplot()
    net_output = [result.net_flow[t]() for t in result.time]
    power_production = [result.plant_power_production[t] for t in result.time]

    plot.plot(net_output)
    plot.plot(power_production)

    plot.legend(["net_flow", "power_production"])



eprice_data = pd.read_csv('datensatz_risikobehaftetes_wetter/eprice_2018-11-01-2.csv', sep=',')
eprice_data = eprice_data['Deutschland/Luxemburg[€/MWh]']
electricity_price = np.array(eprice_data)

rebap = pd.read_csv('datensatz_risikobehaftetes_wetter/rebap_2018-11-01.CSV')
rebap = rebap['reBAP [€/Mwh]']
rebap = np.array(rebap)

power_data = pd.read_csv('datensatz_risikobehaftetes_wetter/power_2018-11-01.csv')
power_production = np.array(power_data['p[kW]'])
power_production = np.true_divide(power_production, 100)# MWh

interval_len = len(power_production)

c1_list = create_linear_corridor(slope=(0.5, 0.5), width=50, length=interval_len)
c1_center_list = get_corridor_center(c1_list)
c1_center_dict = list_to_dict(c1_list)
corridor1_dict = list_to_dict(c1_list[0]), list_to_dict(c1_list[1])


der_dict1 = {None: {
    'time': range(0, 169),
    'charge_power': (0, 48),
    'discharge_power': (0, 0),
    'corridor_lower_bound': corridor1_dict[0],
    'corridor_upper_bound': corridor1_dict[1],
    'soc_guess': c1_list,
    'charge_guess': None,
    'initial_soc': 0.8
}}


c2_list = create_linear_corridor(slope=(0, 0), width=50, length=interval_len)
c2_center_list = get_corridor_center(c2_list)
c2_center_dict = list_to_dict(c2_list)
corridor2_dict = list_to_dict(c2_list[0]), list_to_dict(c2_list[1])

der_dict2 = {None: {
    'time': range(0, 169),
    'charge_power': (0, 40),
    'discharge_power': (0, 40),
    'corridor_lower_bound': corridor2_dict[0],
    'corridor_upper_bound': corridor2_dict[1],
    'soc_initial': None,
    'charge_inital': None,
    'initial_soc': 0.6
}}

liste = []
liste.append(der_dict1)
liste.append(der_dict2)

optimisation = DEROptimisation(electricity_price, power_production, liste)
result = optimisation.solve()

plot_der_corridors(2)
plot_power_flows()

plt.show()


