# You are given Price and Quantity array
import numpy as np
import operator



def cost_one_battery_state_to_another(battery_state_left, battery_state_right, price, quantity):
	if battery_state_right > battery_state_left + 5:
		return np.inf
	if battery_state_right < battery_state_left - 5:
		return np.inf
	diff_states = battery_state_left - battery_state_right
	# If diff_states greater than 0 DISCHARGING
	# If diff_states less than 0 CHARGING
	if diff_states <= 0:
		return (np.abs(diff_states) + quantity) * price
	if diff_states > 0:
		return (-0.8*np.abs(diff_states) + quantity) * price



def black_box(Price, Quantity):
	cost_matrix = np.full((26, Price.size + 1), np.inf)
	quantity_matrix = np.full((26, Price.size + 1), np.inf)
	cost_matrix[0, -1] = 0
	quantity_matrix[0, -1] = 0
	for hour in range(Price.size-1, -1, -1):
		for left in range(26):
			temp = []
			for right in range(26):
				temp.append(cost_matrix[right, hour + 1] + cost_one_battery_state_to_another(left, right, Price[hour], Quantity[hour]))
				#print cost_one_battery_state_to_another(left, right, Price[hour], Quantity[hour])
			quantity_matrix[left, hour], cost_matrix[left, hour] = min(enumerate(temp), key=operator.itemgetter(1))
	demand_list = [0.] + quantity_matrix[0, :].tolist()
	demand_list.pop()
	qty = np.array(demand_list[1:]) - np.array(demand_list[:-1])
	return cost_matrix[0,0], Quantity + (qty > 0).astype(np.int) * qty + (qty <= 0).astype(np.int) * 0.8 * qty


print black_box(np.array([1,3]), np.array([4,4]))
print cost_one_battery_state_to_another(5,0,3, 4)

def black_box_quantity(Price, Quantity):
	cost_matrix = np.full((26, Price.size + 1), np.inf)
	quantity_matrix = np.full((26, Price.size + 1), np.inf)
	cost_matrix[0, -1] = 0
	quantity_matrix[0, -1] = 0
	for hour in range(Price.size-1, -1, -1):
		for left in range(26):
			temp = []
			for right in range(26):
				temp.append(cost_matrix[right, hour + 1] + cost_one_battery_state_to_another(left, right, Price[hour], Quantity[hour]))
				#print cost_one_battery_state_to_another(left, right, Price[hour], Quantity[hour])
			quantity_matrix[left, hour], cost_matrix[left, hour] = min(enumerate(temp), key=operator.itemgetter(1))
	demand_list = [0.] + quantity_matrix[0, :].tolist()
	demand_list.pop()
	qty = np.array(demand_list[1:]) - np.array(demand_list[:-1])
	ans = (qty > 0).astype(np.int) + (qty < 0).astype(np.int) * -1
	return ans