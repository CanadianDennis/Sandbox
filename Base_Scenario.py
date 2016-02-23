'''

C_c = cost of corrective replacement
C_d = downtime or shortage cost per unit time
C_h = spares holding cost, cost for holding one spare part in inventory for one unit of time
C_o = spares order cost
C_p = cost of preventive replacement
h = time horizon
i = suffix i indicates asset number
j = suffix j indicates j-th event
N = number of assets, fleet size
O = equals S - St, and is the number of spare parts ordered at the order review
p = mixing parameter
R = every time units the stock of spare parts is reviewed
S = every time units spare parts are ordered to replenish the inventory from the current stock level up to a fixed level of stock units
S_o = stock on hand
S_t = current inventory position
T = age of component when preventive replacement is performed
T_i = initial age of asset at start of simulation
t = current simulation time
t_c[i][j] = time of j-th corrective replacement on asset i
t_f[i][j] = time of j-th failure on asset i
t_p[i][j] = time of j-th preventive replacement on asset i
t_u = unit of time
nu_1, nu_2 = characteristic life parameters of mixture time to failure distribution
beta_1, beta_2 = shape parameters of mixture time to failure distribution
tau = lead time
tau_r = unplanned replacement downtime
theta_opt = the joint optimal policy
C_opt = the long-run total cost per unit time or cost-rate of the joint optimal policy
C_p_opt = cost of preventive replacements in joint optimal policy
C_c_opt = cost of corrective replacements in joint optimal policy
C_i_opt = cost of inventory (holding cost) in joint optimal policy
C_o_opt = cost of orders in joint optimal policy
C_d_opt = cost of downtime in joint optimal policy the cost-rate per asset, (i.e. the long-run total cost per unit time per asset)
lambda = the cost-rate scale effect
E_d = average demand for spare parts per unit time
sigma_d = standard deviation of the demand for spare parts per unit time

F = failure cumulative distribution function

'''

import random
import math
import statistics

# Input parameters
t_u = 1
tau = 1
tau_r = 0.5
C_d = 6000
C_p = 1000
C_c = 5000
h = 25
C_o = 200
C_h = 50
nu_1 = 10
nu_2 = 70
beta_1 = 3
beta_2 = 3
p_list = (0, 0.1, 0.3)
N_list = (1, 2, 5, 10, 15, 20, 30)

# Temporary, to be removed and replaced with loops
N = 1
p = 0.3
T = 46
R = 1
S = 1

# Compute failure cumulative distribution function
F = []
for t in range(0, h+t_u, t_u):
    F_1 = 1 - math.exp(-1 * (t / nu_1) ** beta_1)
    F_2 = 1 - math.exp(-1 * (t / nu_2) ** beta_2)
    F.append(p * F_1 + (1 - p) * F_2)


def get_failure_time():
    for t in range(0, h+t_u, t_u):
        i = round(t / t_u)
        if F[i] > random.random():
            return t
    return h+t_u

n_i = 1

C_h_i = [0] * n_i
C_p_i = [0] * n_i
C_c_i = [0] * n_i
C_d_i = [0] * n_i
C_o_i = [0] * n_i

for n in range(0, n_i):
    # Compute initial times for preventive maintenance (t_p) and failures (t_f)
    t_p = []
    t_f = []
    for i in range(0, N):
        t_p.append(T)
        t_f.append(get_failure_time())
    t_r = R

    print("t_p =", t_p)
    print("t_f =", t_f)
    print("t_r =", t_r)

    S_o = S

    for t in range(t_u, h+t_u, t_u):
        # Incur holding cost for previous time period
        C_h_i[n] += C_h * S_o

        # Incur order cost if placing an order
        if t == t_r:
            C_o_i[n] += C_o

        # Receive spares after lead time has expired
        if t == (t_r + tau):
            S_o = S
            t_r += R

        for i in range(0, N):
            # Corrective maintenance is needed and can be performed
            if t >= t_f[i] and S_o > 0:
                S_o -= 1
                C_c_i[n] += C_c
                C_d_i[n] += C_d * (t - t_f[i]) / t_u
                t_f[i] = t + get_failure_time()
                t_p[i] = t + T
            # Preventive maintenance is needed and can be performed
            elif t >= t_p[i] and S_o > 0:
                S_o -= 1
                C_p_i[n] += C_p
                t_f[i] = t + get_failure_time()
                t_p[i] = t + T

C_h_i = statistics.mean(C_h_i) / h
C_p_i = statistics.mean(C_p_i) / h
C_c_i = statistics.mean(C_c_i) / h
C_d_i = statistics.mean(C_d_i) / h
C_o_i = statistics.mean(C_o_i) / h
C = C_h_i + C_p_i + C_c_i + C_d_i + C_o_i

print("Results:")
print("C     =", C)
print("C_h_i =", C_h_i)
print("C_p_i =", C_p_i)
print("C_c_i =", C_c_i)
print("C_d_i =", C_d_i)
print("C_o_i =", C_o_i)