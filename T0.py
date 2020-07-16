import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# T0: Windpool ohne P2H-Anlage
# Erlös-max über Gesamtzeit = Summe_Time [ power(t) * erlös_pro_MWh(t) ] dt

"""
Very Simple Economical Model for a DER

Input: Power Production (DER) and Electricity Price [t]

Output: Earnings [t]
"""


eprice_data = pd.read_csv('datensatz_risikobehaftetes_wetter/eprice_2018-11-01.csv', sep=';')
eprice_data = eprice_data['Deutschland/Luxemburg[€/MWh]']
eprice_data = eprice_data.iloc[:].str.replace(',', '.').astype(float)# cant read X,Y as float
electricity_price = np.array(eprice_data)

power_data = pd.read_csv('datensatz_risikobehaftetes_wetter/power_2018-11-01.csv')
power_production = np.array(power_data['p[kW]'])
power_production = np.true_divide(power_production, 10000)# um auf MWh zu kommen

time = range(len(electricity_price))

revenue = [electricity_price[t] * power_production[t] for t in time]

fig = plt.figure()

ax11 = fig.add_subplot(2, 2, 1)
ax12 = fig.add_subplot(2, 2, 2)
ax21 = fig.add_subplot(2, 2, 3)

ax11.plot(time, power_production)
ax11.set_ylabel('Power [MWh]')
ax12.plot(time, electricity_price)
ax12.set_ylabel('Electricity Price [€/MWh]')
ax21.plot(time, revenue)
ax21.set_ylabel('Earnings [€]')

print(revenue)
print(sum(revenue))

plt.show()