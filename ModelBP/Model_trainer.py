import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pyswarm import pso

sigma_qty = []
sigma_price = []
x_values_qty_charging = []
x_values_qty_discharging = []
x_values_qty_neutral = []
x_values_price = []
lb = [-5, -5]
ub = [5, 5]

def cost_func(result_bidding, my_bid_quantity, my_bid_price, demand_train, hour, solar_train):
	cost_if_won = result_bidding * (my_bid_quantity * my_bid_price  + (demand_train[:, hour] - my_bid_quantity - solar_train[:, hour]).clip(min=0) * 7)
	cost_if_lost = (1 - result_bidding) * (7 * (demand_train[:, hour] - solar_train[:, hour]))
	total_cost = np.sum(cost_if_won) + np.sum(cost_if_lost)
	return total_cost

def obj_func_charging(x):
	x_value = x[0]
	y_value = x[1]

	my_bid_quantity_charging = (demand_train_pred_hour - solar_train_pred_hour + x_value*sigma_error_quantity + 5).clip(min=0)
	my_bid_price = price_train_pred_hour + y_value*sigma_error_price
	result_bidding = (my_bid_price >= price_train_hour).astype(np.int)

	cost_charging = cost_func(result_bidding, my_bid_quantity_charging, my_bid_price, demand_train, hour, solar_train)
	return cost_charging

def obj_func_discharging(x):
	x_value = x[0]
	y_value = x[1]

	my_bid_quantity_discharging = (demand_train_pred_hour - solar_train_pred_hour + x_value*sigma_error_quantity - 4).clip(min=0)
	my_bid_price = price_train_pred_hour + y_value*sigma_error_price
	result_bidding = (my_bid_price >= price_train_hour).astype(np.int)

	cost_discharging = cost_func(result_bidding, my_bid_quantity_discharging, my_bid_price, demand_train, hour, solar_train)
	return cost_discharging

def obj_func_neutral(x):
	x_value = x[0]
	y_value = x[1]

	my_bid_quantity_neutral = (demand_train_pred_hour - solar_train_pred_hour + x_value*sigma_error_quantity).clip(min=0)
	my_bid_price = price_train_pred_hour + y_value*sigma_error_price
	result_bidding = (my_bid_price >= price_train_hour).astype(np.int)

	cost_neutral = cost_func(result_bidding, my_bid_quantity_neutral, my_bid_price, demand_train, hour, solar_train)
	return cost_neutral

for block in range(1):
	demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[50*block:50*(block+18), :]
	demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+18), :]
	solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[50*block:50*(block+18), :]
	solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+18), :]
	price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()[50*block:50*(block+18), :]
	price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+18), :]

for block in range(1):
	temp_qty_sigma = []
	temp_price_sigma = []
	temp_x_value_qty_charging = []
	temp_x_value_qty_discharging = []
	temp_x_value_qty_neutral = []
	temp_x_value_price = []

	for hour in range(24):

		error_data_quantity = (demand_train[:, hour] - solar_train[:, hour]) - (demand_train_pred[:, hour] - solar_train_pred[:, hour])
		error_data_price = (price_train_pred[:, hour] - price_train[:, hour])

		(mu_error_quantity, sigma_error_quantity) = norm.fit(error_data_quantity)
		(mu_error_price, sigma_error_price) = norm.fit(error_data_price)

		print(sigma_error_price, sigma_error_quantity)

		demand_train_pred_hour = demand_train_pred[:, hour]
		solar_train_pred_hour = solar_train_pred[:, hour]
		price_train_pred_hour = price_train_pred[:, hour]
		price_train_hour = price_train[:, hour]


		params_charging, cost_charging = pso(obj_func_charging, lb, ub, swarmsize = 1000, maxiter = 10, debug = True)
		params_discharging, cost_discharging = pso(obj_func_discharging, lb, ub, swarmsize = 1000, maxiter = 10, debug = True)
		params_neutral, cost_neutral = pso(obj_func_neutral, lb, ub, swarmsize = 1000, maxiter = 10, debug = True)

		temp_qty_sigma.append(sigma_error_quantity)
		temp_price_sigma.append(sigma_error_price)
		temp_x_value_qty_charging.append(params_charging[0])
		temp_x_value_qty_discharging.append(params_discharging[0])
		temp_x_value_qty_neutral.append(params_neutral[0])
		temp_x_value_price.append(params_charging[1])

		'''
		print "Hour {}".format(hour)
		print np.min(cost_charging), np.unravel_index(np.argmin(cost_charging), cost_charging.shape), x_values[np.unravel_index(np.argmin(cost_charging), cost_charging.shape)[0]], y_values[np.unravel_index(np.argmin(cost_charging), cost_charging.shape)[1]]
		print np.min(cost_discharging), np.unravel_index(np.argmin(cost_discharging), cost_discharging.shape), x_values[np.unravel_index(np.argmin(cost_discharging), cost_discharging.shape)[0]], y_values[np.unravel_index(np.argmin(cost_discharging), cost_discharging.shape)[1]]
		print np.min(cost_neutral), np.unravel_index(np.argmin(cost_neutral), cost_neutral.shape), x_values[np.unravel_index(np.argmin(cost_neutral), cost_neutral.shape)[0]], y_values[np.unravel_index(np.argmin(cost_neutral), cost_neutral.shape)[1]]
		'''
	sigma_qty.append(temp_qty_sigma)
	sigma_price.append(temp_price_sigma)
	x_values_qty_charging.append(temp_x_value_qty_charging)
	x_values_qty_discharging.append(temp_x_value_qty_discharging)
	x_values_qty_neutral.append(temp_x_value_qty_neutral)
	x_values_price.append(temp_x_value_price)

sigma_qty = np.mean(np.asarray(sigma_qty), axis=0)
sigma_price = np.mean(np.asarray(sigma_price), axis=0)
x_values_qty_charging = np.mean(np.asarray(x_values_qty_charging), axis=0)
x_values_qty_discharging = np.mean(np.asarray(x_values_qty_discharging), axis=0)
x_values_qty_neutral = np.mean(np.asarray(x_values_qty_neutral), axis=0)
x_values_price = np.mean(np.asarray(x_values_price), axis=0)

final = np.vstack((sigma_qty, sigma_price, x_values_qty_charging, x_values_qty_discharging, x_values_qty_neutral, x_values_price))

np.savetxt('Model_parameters.txt', final, fmt='%.3e')
