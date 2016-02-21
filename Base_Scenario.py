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
O = equals S-St, and is the number of spare parts ordered at the order review
p = mixing parameter
R = every time units the stock of spare parts is reviewed
S = every time units spare parts are ordered to replenish the inventory from the current stock level up to a fixed level of stock units
S_0 = stock on hand
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

'''

