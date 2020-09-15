from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from optimisation import DEROptimisation

def show_power():
    fig, axs = plt.subplots(3, 1)

    n = [result.net_flow[t]() for t in result.time]

    soc = [[result.der[der].soc[t]() for t in result.time] for der in result.der]

    power = [[-result.der[der].charge_power[t]() + result.der[der].discharge_power[t]() for t in result.time] for
             der in result.der]

    price = np.array([result.electricity_price[t] for t in result.time])

    points = np.array([range(0, interval_len), n], dtype=object).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    min_price = min(price)
    max_price = max(price)

    norm = plt.Normalize(min_price, max_price)
    lc = LineCollection(segments, cmap=plt.get_cmap('jet'), norm=norm)# viridis

    # Set the values used for colormapping
    lc.set_array(np.array(price))
    lc.set_linewidth(3)
    line = axs[0].add_collection(lc)

    divider = make_axes_locatable(axs[0])
    cax = divider.new_vertical(size="5%", pad=0.6)
    fig.add_axes(cax)
    fig.colorbar(line, cax=cax, orientation="horizontal")

    x = range(0, len(power[0]))
    axs[0].stackplot(x, power_production, power[0], power[1], colors=["w", "c", "b"])

    axs[0].plot(power_production, "m")
    axs[0].set_ylabel("Power P")
    axs[0].set_title("Powers")
    axs[0].legend(["Generation", "DER1", "Netzfluss", "DER2"], loc="upper center", ncol=5)
    axs[0].grid(True)

    axs[0].set_xlim(0, 169)
    axs[0].set_ylim(min(n), max(n))

    lower = [result.der[0].corridor_lower_bound[t] for t in result.time]
    upper = [result.der[0].corridor_upper_bound[t] for t in result.time]

    lower1 = [result.der[1].corridor_lower_bound[t] for t in result.time]
    upper1 = [result.der[1].corridor_upper_bound[t] for t in result.time]

    axs[1].plot(lower, "k--")
    axs[1].plot(soc[0], "c")
    axs[1].plot(upper, "k--")
    axs[1].set_title("Corridor DER1")
    axs[1].set_ylabel("Energy E")
    axs[1].grid(True)

    axs[2].plot(lower1, "k--")
    axs[2].plot(soc[1], "b")
    axs[2].plot(upper1, "k--")
    axs[2].set_title("Corridor DER2")
    axs[2].set_xlabel("Time t")
    axs[2].set_ylabel("Energy E ")
    axs[2].grid(True)

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

#plot_der_corridors(2)
#plot_power_flows()

show_power()

plt.show()


