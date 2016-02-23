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

# Input parameters
t_u = 1
tau = 1
tau_r = 0.5
C_d = 6000
C_p = 1000
C_c = 5000
h = 100
C_o = 200
C_h = 50
nu_1 = 10
nu_2 = 70
beta_1 = 3
beta_2 = 3
p_list = (0, 0.1, 0.3)
N_list = (1, 2, 5)

# Temporary, to be removed and replaced with loops
N = 5
p = 0.3
T = 51
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
    return h+1


# Compute initial times for preventive maintenance (t_p) and failures (t_f)
t_p = []
t_f = []
for i in range(0, N):
    t_p.append(T)
    t_f.append(get_failure_time())
t_r = R

S_o = S
C = 0

for t in range(t_u, h+t_u, t_u):
    # Incur holding cost for previous time period
    C += C_h * S_o

    # Incur order cost if placing an order
    if t == t_r:
        C += C_o

    # Receive spares after lead time has expired
    if t == (t_r + tau):
        S_o = S
        t_r += R

    for i in range(0, N):
        # Corrective maintenance is needed and can be performed
        if t >= t_f[i] and S_o > 0:
            S_o -= 1
            C += C_c + C_d * (t - t_f[i]) / t_u
            t_f[i] = t + get_failure_time()
            t_p[i] = t + T
        # Preventive maintenance is needed and can be performed
        elif t >= t_p[i] and S_o > 0:
            S_o -= 1
            S_t -= 1
            C += C_p
            t_f[i] = t + get_failure_time()
            t_p[i] = t + T

print("Result = ")
print(C / (h / t_u))