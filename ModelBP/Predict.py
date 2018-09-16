import numpy as np
# from scipy_optimize import charge_discharge
import pandas as pd
from Black_Box import black_box_quantity

with open('Final_Parameters.txt') as f:
	content = f.readlines()

std_price = []
x_values_price = []
std_demand_quantity = []
x_values_charging = []
x_values_discharging = []
x_values_neutral = []

var = [std_demand_quantity, std_price, x_values_charging, x_values_discharging, x_values_neutral, x_values_price]
for i in range(6):
	for j in range(24):
		var[i].append(float(content[i].split()[j]))

print std_price
print x_values_price

'''
std_demand_quantity = (np.fromstring(content[0], count=24)).astype(float)
std_price = (np.fromstring(content[1], count=24)).astype(float)
x_values_charging = (np.fromstring(content[2], count=24)).astype(float)
x_values_discharging = (np.fromstring(content[3], count=24)).astype(float)
x_values_neutral = (np.fromstring(content[4], count=24)).astype(float)
x_values_price = (np.fromstring(content[5], count=24)).astype(float)

for i in range(len(content)):
	if i % 5 == 0:
		std_price.append(float(content[i].split()[0]))
		std_demand_quantity.append(float(content[i].split()[1]))
	if i % 5 == 2:
		#Charging
		x_values_charging.append(float(content[i].split()[-2]))
		x_values_price.append(float(content[i].split()[-1]))
	if i % 5 == 3:
		x_values_discharging.append(float(content[i].split()[-2]))
	if i % 5 == 4:
		x_values_neutral.append(float(content[i].split()[-2]))


std_price = np.array(std_price)
x_values_price = np.array(x_values_price)
std_demand_quantity = np.array(std_demand_quantity)
x_values_charging = np.array(x_values_charging)
x_values_discharging = np.array(x_values_discharging)
'''


demand_test_pred = pd.read_csv('Demand_PS_pred.csv', header=None).as_matrix()
solar_test_pred = pd.read_csv('Solar_PS_pred.csv', header=None).as_matrix()
price_test_pred = pd.read_csv('Price_PS_pred.csv', header=None).as_matrix()

bid_price = np.zeros(price_test_pred.shape)
bid_quantity = np.zeros(price_test_pred.shape)
charge_decision = black_box_quantity(price_test_pred.ravel(), (demand_test_pred - solar_test_pred).ravel())
#print charge_decision.shape
#print type(charge_decision)
charge_decision = charge_decision.reshape(price_test_pred.shape)
# charge_decision = np.zeros(price_test_pred.shape)

for hour in range(0, 24):
	bid_price[:, hour] = (price_test_pred[:, hour] + x_values_price[hour]*std_price[hour]).clip(max=7.)

for hour in range(0, 24):
	charge = (charge_decision[:, hour] > 0).astype(np.int) * (5 + x_values_charging[hour]*std_demand_quantity[hour])
	discharge = (charge_decision[:, hour] < 0).astype(np.int) * ((-4) + x_values_discharging[hour]*std_demand_quantity[hour])
	neutral = (charge_decision[:, hour] == 0).astype(np.int) * (x_values_neutral[hour]*std_demand_quantity[hour])
	bid_quantity[:, hour] = (demand_test_pred[:, hour] - solar_test_pred[:, hour] + charge + discharge + neutral).clip(min=0)




temp1 = np.reshape(bid_price, (np.product(bid_price.shape), 1))
temp2 = np.reshape(bid_quantity, (np.product(bid_quantity.shape), 1))
#temp2 = np.reshape(black_box(temp1.ravel(), temp2.ravel())[1], (np.product(bid_quantity.shape), 1))
final = np.concatenate((temp1, temp2), axis=1)
full_final = pd.DataFrame(final)
full_final.to_csv('PS_Bids.csv', index=False, header = False)
