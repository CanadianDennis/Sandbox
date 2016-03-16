"""
"""

# C_c = cost of corrective replacement
# C_d = downtime or shortage cost per unit time
# C_h = spares holding cost, cost for holding one spare part in inventory for one unit of time
# C_o = spares order cost
# C_p = cost of preventive replacement
# h = time horizon
# i = suffix i indicates asset number
# j = suffix j indicates j-th event
# N = number of assets, fleet size
# O = equals S - St, and is the number of spare parts ordered at the order review
# p = mixing parameter
# R = every time units the stock of spare parts is reviewed
# S = every time units spare parts are ordered to replenish the inventory from the current stock level
#     up to a fixed level of stock units
# S_o = stock on hand
# S_t = current inventory position
# T = age of component when preventive replacement is performed
# T_i = initial age of asset at start of simulation
# t = current simulation time
# t_c[i][j] = time of j-th corrective replacement on asset i
# t_f[i][j] = time of j-th failure on asset i
# t_p[i][j] = time of j-th preventive replacement on asset i
# t_u = unit of time
# eta_1, eta_2 = characteristic life parameters of mixture time to failure distribution
# beta_1, beta_2 = shape parameters of mixture time to failure distribution
# tau = lead time
# tau_r = unplanned replacement downtime
# theta_opt = the joint optimal policy
# C_opt = the long-run total cost per unit time or cost-rate of the joint optimal policy
# C_p_opt = cost of preventive replacements in joint optimal policy
# C_c_opt = cost of corrective replacements in joint optimal policy
# C_i_opt = cost of inventory (holding cost) in joint optimal policy
# C_o_opt = cost of orders in joint optimal policy
# C_d_opt = cost of downtime in joint optimal policy the cost-rate per asset,
#           (i.e. the long-run total cost per unit time per asset)
# lambda = the cost-rate scale effect
# E_d = average demand for spare parts per unit time
# sigma_d = standard deviation of the demand for spare parts per unit time
#
# F = failure cumulative distribution function

import numpy as np


# Evaluate failure time based on a randomly generated number
def get_failure_time(cfd, time_cfd):
    # cfd = cumulative failure distribution (numpy array)
    # time_cfd = time vector associated with cfd
    rand = 1.0 - np.random.random()
    return time_cfd[np.searchsorted(cfd, rand) - 1]


# Input parameters
t_u = 1
tau = 1
tau_r = 0.5
C_d = 6000
C_p = 1000
C_c = 5000
h = 2000
C_o = 200
C_h = 50
eta_1 = 10
eta_2 = 70
beta_1 = 3
beta_2 = 3
p_list = (0, 0.1, 0.3)
N_list = (1, 2, 5, 10, 15, 20, 30)

num_iter = 10

time_arr = np.arange(0, h, t_u)
time_start = 100

# To Do: Replace with loops
N = 20
p = 0

t_0 = np.zeros(N)

cost_p = np.zeros([num_iter, time_arr.size])
cost_c = np.zeros([num_iter, time_arr.size])
cost_h = np.zeros([num_iter, time_arr.size])
cost_o = np.zeros([num_iter, time_arr.size])
cost_d = np.zeros([num_iter, time_arr.size])

# Compute failure cumulative distribution function
F_1 = 1 - np.exp(-1 * (time_arr / eta_1) ** beta_1)
F_2 = 1 - np.exp(-1 * (time_arr / eta_2) ** beta_2)
F = p * F_1 + (1 - p) * F_2
# For efficiency, truncate at a value very close to 1
F = F[:np.searchsorted(F, 0.99999)]

# To Do: Evaluate optimal parameters
T = 29
R = 1
S = 1

for i in range(0, num_iter):
    # Compute initial event times
    t_p = t_0 + np.ones(N) * T
    t_f = np.zeros(N)
    for n in range(0, N):
        t_f[n] = t_0[n] + get_failure_time(F, time_arr)
    t_r = R
    t_r_tau = t_r + tau

    # Set variables used to track spares and orders
    S_o = S
    S_t = S
    spare_ordered = [False] * N
    order_tracker = []

    debug_action = ["N"] * N

    for time_idx, t in np.ndenumerate(time_arr):
        debug_order = []

        # Incur holding cost for previous time period
        cost_h[i, time_idx] += C_h * S_o

        # Receive spares after lead time has expired
        if t >= t_r_tau and len(order_tracker) > 0:
            k = order_tracker.pop(0)
            S_o += k
            S_t += k
            t_r_tau = t_r + tau
            debug_order.append("Order Rx")

        for n in range(0, N):
            # Corrective maintenance is needed
            if t >= t_f[n] and t_p[n] != t_f[n]:
                # Corrective maintenance can be performed
                if S_o > 0:
                    S_o -= 1
                    if not spare_ordered[n]:
                        S_t -= 1
                    cost_c[i, time_idx] += C_c
                    cost_d[i, time_idx] += C_d * (t - t_f[n] + tau_r) / t_u
                    t_f[n] = t + get_failure_time(F, time_arr)
                    t_p[n] = t + T
                    spare_ordered[n] = False
                    debug_action[n] = "CC"
                # Corrective maintenance cannot be performed and inventory position has not been updated
                elif not spare_ordered[n]:
                    S_t -= 1
                    spare_ordered[n] = True
                    debug_action[n] = "XC"
                else:
                    debug_action[n] = "CX"

            # Preventive maintenance is needed
            elif t >= t_p[n]:
                # Preventive maintenance can be performed
                if S_o > 0:
                    S_o -= 1
                    if not spare_ordered[n]:
                        S_t -= 1
                    cost_p[i, time_idx] += C_p
                    t_f[n] = t + get_failure_time(F, time_arr)
                    t_p[n] = t + T
                    spare_ordered[n] = False
                    debug_action[n] = "PP"
                # Preventive maintenance cannot be performed and inventory position has not been updated
                elif not spare_ordered[n]:
                    S_t -= 1
                    spare_ordered[n] = True
                    debug_action[n] = "XP"
                else:
                    debug_action[n] = "PX"
            else:
                debug_action[n] = "NA"
        # Incur order cost if placing an order
        if t >= t_r and S > S_t:
            order_tracker.append(S - S_t)
            cost_o[i, time_idx] += C_o
            t_r += R
            debug_order.append("Order Tx")

        # print("t =", t, " ; S_o =", S_o, " ; S_t =", S_t)
        # print("debug_action =", debug_action)
        # print("debug_order =", debug_order, " ; order_tracker =", order_tracker)
        # print("t_p =", t_p)
        # print("t_f =", t_f)

cost_p = cost_p[:, time_start:]
cost_c = cost_c[:, time_start:]
cost_h = cost_h[:, time_start:]
cost_o = cost_o[:, time_start:]
cost_d = cost_d[:, time_start:]

C = np.average(cost_p) + np.average(cost_c) + np.average(cost_h) + np.average(cost_o) + np.average(cost_d)

print("Results:")
print("C   = $%10.2f"% C)
print("C_p = $%10.2f"% np.average(cost_p))
print("C_c = $%10.2f"% np.average(cost_c))
print("C_i = $%10.2f"% np.average(cost_h))
print("C_o = $%10.2f"% np.average(cost_o))
print("C_d = $%10.2f"% np.average(cost_d))
