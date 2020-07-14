import matplotlib.pyplot as plt

# T0: Windpool ohne P2H-Anlage
# Erlös-max über Gesamtzeit = Summe_Time [ power(t) * erlös_pro_MWh(t) ] dt

"""
Very Simple Economical Model for a DER

Input: Power Production (DER) and Electricity Price [t]

Output: Earnings [t]
"""


time            = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9]
power_in_mwh    = [10, 9,  9,  8,  10, 11, 12, 13, 10, 9]
erloes_pro_mwh  = [1,  2,  3,  4,  5,  5,  5,  5,  4,  4]
erloes_gesamt   = []

ind = 0
for mwh in power_in_mwh:
    erloes_gesamt.append(mwh * erloes_pro_mwh[ind])
    ind += 1

fig = plt.figure()

ax11 = fig.add_subplot(2, 2, 1)
ax12 = fig.add_subplot(2, 2, 2)
ax21 = fig.add_subplot(2, 2, 3)

ax11.plot(time, power_in_mwh)
ax11.set_ylabel('Power [MWh]')
ax12.plot(time, erloes_pro_mwh)
ax12.set_ylabel('Electricity Price [€/MWh]')
ax21.plot(time, erloes_gesamt)
ax21.set_ylabel('Earnings [€]')

plt.show()