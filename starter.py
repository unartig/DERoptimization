from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from optimisation import DEROptimisation
from optimisation_imba import IMBAOptimisation


def show_power():
    fig, axs = plt.subplots(4, 1)

    net_flow = [result.net_flow[t]() for t in result.time]

    soc = [[result.der[der].soc[t]() for t in result.time] for der in result.der]

    power = [[-result.der[der].charge_power[t]() + result.der[der].discharge_power[t]() for t in result.time] for
             der in result.der]

    price = np.array([result.electricity_price[t] for t in result.time])

    points = np.array([range(0, interval_len), net_flow], dtype=object).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    min_price = min(price)
    max_price = max(price)

    norm = plt.Normalize(min_price, max_price)

    lc = LineCollection(segments, cmap='jet', norm=norm)# viridis

    # Set the values used for colormapping
    lc.set_array(np.array(price))
    lc.set_linewidth(0)
    line = axs[0].add_collection(lc)

    divider = make_axes_locatable(axs[0])
    cax = divider.new_vertical(size="50%", pad=0.7)
    fig.add_axes(cax)
    fig.colorbar(line, cax=cax, orientation="horizontal")

    pseudo_matrix = np.array(electricity_price)
    pseudo_matrix = np.expand_dims(pseudo_matrix, axis=0)
    im = axs[0].imshow(pseudo_matrix, cmap="jet")

    axs[0].set_title("Electricity Price")
    axs[0].get_yaxis().set_visible(False)


    x = range(0, len(power[0]))
    axs[1].stackplot(x, power_production, power[0], power[1], colors=["w", "c", "b"])

    axs[1].plot(power_production, "m")
    axs[1].plot(net_flow, "r--")
    axs[1].set_ylabel("Power P")
    axs[1].set_title("Powers")
    axs[1].legend(["Generation", "DER1", "Netzfluss", "DER2"], loc="upper center", ncol=5)
    axs[1].grid(True)

    axs[1].set_xlim(0, 169)
    axs[1].set_ylim(min(net_flow), max(net_flow))

    lower = [result.der[0].corridor_lower_bound[t] for t in result.time]
    upper = [result.der[0].corridor_upper_bound[t] for t in result.time]

    lower1 = [result.der[1].corridor_lower_bound[t] for t in result.time]
    upper1 = [result.der[1].corridor_upper_bound[t] for t in result.time]

    axs[2].plot(lower, "k--")
    axs[2].plot(soc[0], "c")
    axs[2].plot(upper, "k--")
    axs[2].set_title("Corridor DER1")
    axs[2].set_ylabel("Energy E")
    axs[2].grid(True)

    axs[3].plot(lower1, "k--")
    axs[3].plot(soc[1], "b")
    axs[3].plot(upper1, "k--")
    axs[3].set_title("Corridor DER2")
    axs[3].set_xlabel("Time t")
    axs[3].set_ylabel("Energy E ")
    axs[3].grid(True)


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

show_power()


power_production = list(power_production)
power_production = power_production[-5:] + power_production[:-5]

bid = [result.net_flow[t]() for t in result.time]

der1charge = [result.der[1].charge_power[t]() for t in result.time]
der1dcharge = [result.der[1].discharge_power[t]() for t in result.time]

der2charge = [result.der[1].charge_power[t]() for t in result.time]
der2dcharge = [result.der[1].discharge_power[t]() for t in result.time]

opti2 = IMBAOptimisation(electricity_price, power_production, liste, bid, der1charge, der1dcharge, der2charge, der2dcharge)
result = opti2.solve()

#show_power()


fig = plt.figure()
plot = fig.add_subplot()

bid3 = [result.net_flow[t]() for t in result.time]
bid5 = [result.bid[t] for t in result.time]

plot.plot(bid3)
plot.plot(bid5)
plot.grid(True)
plot.legend(["korrektur", "summending", "bid"])
plt.show()

