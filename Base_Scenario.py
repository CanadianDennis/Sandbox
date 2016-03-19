""" Base scenario simulation used for validation against reference journal article

Author: Dennis McLean
Date: 2016-03-18
Course: EMP 5103 C

[1] A. Van Horenbeek, P. A. Scarf, C. A. V. Cavalcante and L. Pintelon, "The effect of maintenance quality on
spare parts inventory for a fleet of assets," IEEE Trans. Rel., vol. 62, no. 3, pp. 596-607, Sep. 2013.
"""

import numpy as np


def get_failure_time(cfd, time_cfd):
    """ Evaluate failure time based on a randomly generated number
    :param cfd: cumulative failure distribution (numpy array)
    :param time_cfd: time vector associated with cfd
    :return: random failure time based on cfd
    """
    rand = 1.0 - np.random.random()
    return time_cfd[np.searchsorted(cfd, rand) - 1]


# Input parameters
t_u = 1
tau = 1
tau_r = 0.5
C_d = 6000
C_p = 1000
C_c = 5000
h = 2500
C_o = 200
C_h = 50
eta_1 = 10
eta_2 = 70
beta_1 = 3
beta_2 = 3
p_list = (0, 0.1, 0.3)
N_list = (1, 2, 5, 10, 15, 20, 30)
T_list =  (26, 29, 30, 31, 36, 37, 38, 45, 46, 47, 48)
R = 1
S = 1

num_iter = 25

time_arr = np.arange(0, h, t_u)
time_start = 100

print("N\tp\tT\tC\tC_p\tC_c\tC_i\tC_o\tC_d")

for p in p_list:

    # Compute failure cumulative distribution function
    F_1 = 1 - np.exp(-1 * (time_arr / eta_1) ** beta_1)
    F_2 = 1 - np.exp(-1 * (time_arr / eta_2) ** beta_2)
    F = p * F_1 + (1 - p) * F_2
    # For efficiency, truncate at a value very close to 1
    F = F[:np.searchsorted(F, 0.99999)]

    for N in N_list:

        time_deploy = np.zeros(N)

        C_opt = None

        for T in T_list:

            cost_p = np.zeros([num_iter, time_arr.size])
            cost_c = np.zeros([num_iter, time_arr.size])
            cost_h = np.zeros([num_iter, time_arr.size])
            cost_o = np.zeros([num_iter, time_arr.size])
            cost_d = np.zeros([num_iter, time_arr.size])

            for iter_idx in range(0, num_iter):
                # Compute initial event times
                t_p = time_deploy + np.ones(N) * T
                t_f = np.zeros(N)
                for asst_idx in range(0, N):
                    t_f[asst_idx] = time_deploy[asst_idx] + get_failure_time(F, time_arr)
                t_r = R

                # Set variables used to track spares and orders
                S_o = S
                S_t = S
                back_ordered = [False] * N
                time_orders = []
                orders = []

                debug_action = ["N"] * N

                for time_idx, t in np.ndenumerate(time_arr):
                    debug_order = []

                    # Determine order to parse assets
                    asst_ord = []
                    # Outstanding corrective actions take precedence
                    for asst_idx in np.nditer(np.argsort(t_f)):
                        if t_f[asst_idx] <= t:
                            asst_ord.append(np.asscalar(asst_idx))
                    for asst_idx in np.nditer(np.argsort(t_p)):
                        if asst_idx not in asst_ord:
                            asst_ord.append(np.asscalar(asst_idx))

                    # asst_ord = range(0, N)

                    # Incur holding cost for previous time period
                    cost_h[iter_idx, time_idx] += C_h * S_o

                    # Receive spares after lead time has expired
                    if len(time_orders) > 0 and t >= time_orders[0]:
                        time_orders.pop(0)
                        order_rx = orders.pop(0)
                        S_o += order_rx
                        S_t += order_rx
                        debug_order.append("Order Rx")

                    for asst_idx in asst_ord:
                        # Corrective maintenance is needed
                        if t >= t_f[asst_idx] and t_p[asst_idx] != t_f[asst_idx]:
                            # Corrective maintenance can be performed
                            if S_o > 0:
                                S_o -= 1
                                if not back_ordered[asst_idx]:
                                    S_t -= 1
                                cost_c[iter_idx, time_idx] += C_c
                                cost_d[iter_idx, time_idx] += C_d * (t - t_f[asst_idx] + tau_r) / t_u
                                t_f[asst_idx] = t + get_failure_time(F, time_arr)
                                t_p[asst_idx] = t + T
                                back_ordered[asst_idx] = False
                                debug_action[asst_idx] = "CC"
                            # Corrective maintenance cannot be performed and inventory position has not been updated
                            elif not back_ordered[asst_idx]:
                                S_t -= 1
                                back_ordered[asst_idx] = True
                                debug_action[asst_idx] = "XC"
                            else:
                                debug_action[asst_idx] = "CX"

                        # Preventive maintenance is needed
                        elif t >= t_p[asst_idx]:
                            # Preventive maintenance can be performed
                            if S_o > 0:
                                S_o -= 1
                                if not back_ordered[asst_idx]:
                                    S_t -= 1
                                cost_p[iter_idx, time_idx] += C_p
                                t_f[asst_idx] = t + get_failure_time(F, time_arr)
                                t_p[asst_idx] = t + T
                                back_ordered[asst_idx] = False
                                debug_action[asst_idx] = "PP"
                            # Preventive maintenance cannot be performed and inventory position has not been updated
                            elif not back_ordered[asst_idx]:
                                S_t -= 1
                                back_ordered[asst_idx] = True
                                debug_action[asst_idx] = "XP"
                            else:
                                debug_action[asst_idx] = "PX"
                        else:
                            debug_action[asst_idx] = "NA"
                    # Check inventory levels
                    if t >= t_r:
                        t_r += R
                        # If needed, place an order
                        if S_t < S:
                            time_orders.append(t + tau_r)
                            orders.append(S - S_t)
                            cost_o[iter_idx, time_idx] += C_o
                            debug_order.append("Order Tx")

                    # print("t =", t, " ; S_o =", S_o, " ; S_t =", S_t)
                    # print("debug_action =", debug_action)
                    # print("debug_order =", debug_order, " ; orders =", orders)
                    # print("t_p =", t_p)
                    # print("t_f =", t_f)
                    # print("asst_ord =", asst_ord)
                    # print(" ")

            cost_p = cost_p[:, time_start:]
            cost_c = cost_c[:, time_start:]
            cost_h = cost_h[:, time_start:]
            cost_o = cost_o[:, time_start:]
            cost_d = cost_d[:, time_start:]

            C = np.average(cost_p) + np.average(cost_c) + np.average(cost_h) + np.average(cost_o) + np.average(cost_d)

            if C_opt is None or C < C_opt:
                T_opt = T
                C_opt = C
                C_p_opt = np.average(cost_p)
                C_c_opt = np.average(cost_c)
                C_i_opt = np.average(cost_h)
                C_o_opt = np.average(cost_o)
                C_d_opt = np.average(cost_d)

        print("%d\t%.1f\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"% (N, p, T_opt, C_opt, C_p_opt, C_c_opt, C_i_opt,
                                                                     C_o_opt, C_d_opt))

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
# N_list = list of asset numbers to loop through
# p_list = list of weak component proportions to loop through
# F_1 = failure cumulative distribution function for weak components
# F_2 = failure cumulative distribution function for standard components
# F = combined failure cumulative distribution function
# num_iter = number of iterations to run for each scenario
# time_arr = vector of time points to step through for each simulation run
# time_deploy = time each asset is deployed
# time_start = ignore time points before this time to mimic an already established scenario
# cost_p = preventative replacement cost for each time point for each iteration
# cost_c = corrective replacement cost for each time point for each iteration
# cost_h = holding cost for each time point for each iteration
# cost_o = order cost for each time point for each iteration
# cost_d = downtime cost for each time point for each iteration
# iter_idx = index of the current iteration
# time_idx = index of current time within time_arr
# asst_idx = index of the current asset
# asst_ord = order to parse assets (corrective actions takes priority)
# time_orders = list of arrival times for each order
# orders = list of currently placed orders
# back_ordered = whether or not there is a spare on back order for each asset
# order_rx = received order size

