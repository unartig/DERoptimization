from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimisation import DEROptimisation

eprice_data = pd.read_csv('datensatz_risikobehaftetes_wetter/eprice_2018-11-01-2.csv', sep=',')
eprice_data = eprice_data['Deutschland/Luxemburg[€/MWh]']
electricity_price = np.array(eprice_data)

rebap = pd.read_csv('datensatz_risikobehaftetes_wetter/rebap_2018-11-01.CSV')
rebap = rebap['reBAP [€/Mwh]']
rebap = np.array(rebap)

power_data = pd.read_csv('datensatz_risikobehaftetes_wetter/power_2018-11-01.csv')
power_production = np.array(power_data['p[kW]'])
power_production = np.true_divide(power_production, 10000)# MWh

lower = [0 + 3*x for x in range(0, 169)]
upper = [50 + 3*x for x in range(0, 169)]
mid = [lower[i] + (upper[i] - lower[i]) for i in range(0, len(lower))]
char = [0.5 for i in range(0, len(lower))]

lower_dict = {i: lower[i] for i in range(0, len(lower))}
upper_dict = {i: upper[i] for i in range(0, len(upper))}
mid_dict = {i: mid[i] for i in range(0, len(mid))}
char_dict = {i: char[i] for i in range(0, len(mid))}

der_dict = {None: {
    'time': range(0, 169),
    'charge_power': (0, 48),
    'discharge_power': (0, 0),
    'corridor_lower_bound': lower_dict,
    'corridor_upper_bound': upper_dict,
    'soc_initial': mid_dict,
    'charge_inital': char_dict,
    'initial_soc': 0.8
}}


lower1 = [0 + 0.5*x for x in range(0, 169)]
upper1 = [5 + x for x in range(0, 169)]
mid1 = [lower1[i] + (upper1[i] - lower1[i]) for i in range(0, len(lower1))]

lower_dict1 = {i: lower1[i] for i in range(0, len(lower1))}
upper_dict1 = {i: upper1[i] for i in range(0, len(upper1))}
mid_dict1 = {i: mid1[i] for i in range(0, len(mid1))}

der_dict1 = {None: {
    'time': range(0, 169),
    'charge_power': (0, 111),
    'discharge_power': (0, 0),
    'corridor_lower_bound': lower_dict1,
    'corridor_upper_bound': upper_dict1,
    'soc_initial': mid_dict1,
    'charge_inital': char_dict,
    'initial_soc': 0.5
}}

liste = []
liste.append(der_dict)
#liste.append(der_dict1)

optimisation = DEROptimisation(electricity_price, power_production, liste)
result = optimisation.solve()

soc0 = [result.blocks[0].soc[t]() for t in result.time]
#r = [result.revenue[t]() for t in result.time]
n = [result.net_output[t]() for t in result.time]

print(n)

fig = plt.figure()
ax11 = fig.add_subplot(2, 1, 1)
ax12 = fig.add_subplot(2, 1, 2)

ax11.plot(lower)
ax11.plot(upper)
ax11.plot(soc0)

#ax12.plot(lower1)
#ax12.plot(upper1)
#ax12.plot(soc1)
ax12.plot(n)
ax12.plot(power_production)
ax12.legend(["netto_output", "production"])


plt.show()

