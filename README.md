# DER optimization

This repository contains the algorithmic 'Tasks' 0 - 3, example data and a documentation (will be added soon).

## Task Introduction
Consider the three cost components:

    a. cost of operating device (Anlagenbetriebskosten), CHP different from P2H (only power cost)
	b. trading revenue (Markterlöse des Energiehändlers beim Agg.)
	c. penalty costs for imbalance
	
Algorithmic Challenge for Aggregator: aggregate corridor models for business value. 
The goal is to create time schedules for DER's.
 
    T0: only considering b., simply trading the produced energy on the market.
    Optimizations:
    T1: a.&b. todays VPP pool plus thermal devices/batteries offering flexiblity via their corridors without risk
    T2: a.,b.&c. risk weather front comes X hour earlier; the windmills might be allowed only reduced feed into the grid.
    T3: a.,b.&c. Operation with Riskoreduction 
---    
## Running Instructions

### Setup 

In order to run the optimization you need to have following Python modules installed:
 
 1. numpy
 2. matplotlib
 3. pandas
 4. pyomo
 5. xlrd
 
In addition you need a non linear solver in the project folder. 
Different solvers are capable to handle the optimization, 
but IPOPT will be the simplest choice here. 
A guide on how to set up the ipopt-solver properly can be found here: 
https://github.com/matthias-k/cyipopt. 
A faster way is to download a the solver binary from https://www.coin-or.org/download/binary/Ipopt/.

### Using the models

The code structure will be refactored in the near future (moving to object oriented). 
Right now the models can be used by importing.

```
import Economic_Battery as eco_model_battery
import Imbalance_Reduction_Battery as imbalance_red_battery
```

Following models can be used: 
```
eprice_data = pd.read_csv('datensatz_risikobehaftetes_wetter/eprice_2018-11-01.csv', sep=';')
eprice_data = eprice_data['Deutschland/Luxemburg[€/MWh]']
eprice_data = eprice_data.iloc[:].str.replace(',', '.').astype(float)# cant read X,Y as float
electricity_price = np.array(eprice_data)

power_data = pd.read_csv('datensatz_risikobehaftetes_wetter/power_2018-11-01.csv')
power_production = np.array(power_data['p[kW]'])
power_production = np.true_divide(power_production, 10000)# MWh

```
The parameters ``power_production``, ``electricity_price`` and ``bid`` have to be arrays of the same size. 
Example data is provided in the datensatz_risikobehaftetes_wetter file. 
``critical_t`` can be set to activate the "risk reducing" constraint.

 1. Create a model
 ``
 model = t2b.t2_battery_model(power_production, electricity_price, critical_t=None)
 ``
 2. solve the model
 ``
 model = t2b.solve(model)
 ``
The ``model`` now contains the solved  model. 
(Note this only works for T2 and T3. The models will be refactored, 
the structure will most likely be changed soon)
