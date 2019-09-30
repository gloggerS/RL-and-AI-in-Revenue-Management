"""
This script will calculate the Approximate Policy Iteration with plc value function and store it in a large dataframe
as specified in 0_settings.csv.
"""


from B_helper import *

#%%
# capacity chunk size
chunk_size = 2

# linear policy evaluations at start
K_lin = 1

#%%
# Setup of parameters
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("APIplMultiLeg"+str(chunk_size)+"-Klin"+str(K_lin)+"-")


capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]



#%%
# trick for storing pi_plc later
def intervals_capacities_num(capacities_thresholds):
    return max([len(i) for i in capacities_thresholds])

#%%
def update_parameters(v_samples, c_samples, thetas, pis, k, exponential_smoothing, silent=True):
    """
    Updates the parameter theta, pi given the sample path using least squares linear regression with constraints.
    Using gurobipy

    :param v_samples:
    :param c_samples:
    :return:
    """
    set_i = np.arange(len(v_samples))
    set_t = np.arange(len(thetas))
    set_h = np.arange(len(pis[0]))

    theta_multidict = {}
    for t in set_t:
        theta_multidict[t] = thetas[t]
    theta_indices, theta_tuples = multidict(theta_multidict)

    pi_multidict = {}
    for t in set_t:
        for h in set_h:
            pi_multidict[t, h] = pis[t, h]
    pi_indices, pi_tuples = multidict(pi_multidict)

    try:
        m = Model()

        # Variables
        m_theta = m.addVars(theta_indices, name="theta", lb=0.0)  # Constraint 10
        m_pi = m.addVars(pi_indices, name="pi", ub=max(revenues), lb=0.0)  # Constraint 11

        for t in set_t:
            m_theta[t].start = theta_tuples[t]
            for h in set_h:
                m_pi[t, h].start = pi_tuples[t, h]

        # Goal Function (14)
        lse = quicksum((v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h] * c_samples[i][t][h] for h in set_h)) *
                       (v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h] * c_samples[i][t][h] for h in set_h))
                       for t in set_t for i in set_i)
        m.setObjective(lse, GRB.MINIMIZE)

        # Constraints
        # C12 (not needed yet)
        for t in set_t[:-1]:
            m.addConstr(m_theta[t], GRB.GREATER_EQUAL, m_theta[t+1], name="C15")  # Constraint 15
            for h in set_h:
                m.addConstr(m_pi[t, h], GRB.GREATER_EQUAL, m_pi[t+1, h], name="C16")  # Constraint 16

        if silent:
            m.setParam("OutputFlag", 0)

        m.optimize()

        theta_new = deepcopy(thetas)
        pi_new = deepcopy(pis)

        for t in set_t:
            theta_new[t] = m.getVarByName("theta[" + str(t) + "]").X
            for h in set_h:
                pi_new[t, h] = m.getVarByName("pi[" + str(t) + "," + str(h) + "]").X

        # check exponential smoothing
        if exponential_smoothing:
            return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new, m.ObjVal
        else:
            return theta_new, pi_new, m.ObjVal
    except GurobiError:
        print('Error reported')

        return 0, 0, 0

def update_parameters_plc(v_samples, c_samples, thetas, pis, k, exponential_smoothing, silent=True):
    """
    Updates the parameter theta, pi given the sample path using least squares linear regression with constraints.
    Using gurobipy

    :param v_samples:
    :param c_samples:
    :return:
    """
    set_i = np.arange(len(v_samples))
    set_t = np.arange(len(thetas))
    set_h = np.arange(len(pis[0]))
    set_s = np.arange(intervals_capacities_num(capacities_thresholds) - 1)

    # i, t, h, s
    f = np.array([[[np.zeros(len(set_s))] * len(set_h)] * len(set_t)] * len(set_i))
    for i in set_i:
        for t in set_t:
            for h in set_h:
                # # c_h <= b_h^{s-1}
                # index_c_smaller_lower = c_samples[i][t][h] <= capacities_thresholds[h][:-1]
                # f[i][t][h][index_c_smaller_lower] = 0
                # # b_h^{s-1} < c_h <= b_h^s
                # index_c_smaller = c_samples[i][t][h] <= capacities_thresholds[h][1:]
                # index_c_bigger = c_samples[i][t][h] > capacities_thresholds[h][:-1]
                # f[i][t][h][index_c_smaller & index_c_bigger] = c_samples[i][t][h] - capacities_thresholds[h][
                #     [*(index_c_smaller & index_c_bigger), False]]
                # # b_h^s > c_h
                # index_c_bigger_upper = c_samples[i][t][h] > capacities_thresholds[h][1:]
                # f[i][t][h][index_c_bigger_upper] = capacities_thresholds[h][[False, *(index_c_bigger_upper)]] - \
                #                                    capacities_thresholds[h][[*(index_c_bigger_upper), False]]

                # just the current capacity if active
                # find the relevant interval
                # for which index the capacity is smaller or equal (starting with upper bound of first interval)
                # for which index the capacity is greater (ending with lower bound of last interval)
                index_c_smaller = c_samples[i][t][h] <= capacities_thresholds[h][1:]
                index_c_bigger = c_samples[i][t][h] > capacities_thresholds[h][:-1]
                # trick to fill up correctly with false if capacity of h is not maximal capacity
                index_match = [*(index_c_smaller & index_c_bigger),
                               *[False for _ in np.arange(len(set_s) - len(index_c_smaller))]]
                f[i][t][h][index_match] = c_samples[i][t][h]

    theta_multidict = {}
    for t in set_t:
        theta_multidict[t] = thetas[t]
    theta_indices, theta_tuples = multidict(theta_multidict)

    pi_multidict = {}
    for t in set_t:
        for h in set_h:
            for s in set_s:
                pi_multidict[t, h, s] = pis[t, h, s]
    pi_indices, pi_tuples = multidict(pi_multidict)

    try:
        m = Model()

        # Variables
        m_theta = m.addVars(theta_indices, name="theta", lb=0.0)  # Constraint 10
        m_pi = m.addVars(pi_indices, name="pi", ub=max(revenues), lb=0.0)  # Constraint 11

        for t in set_t:
            m_theta[t].start = theta_tuples[t]
            for h in set_h:
                for s in set_s:
                    m_pi[t, h, s].start = pi_tuples[t, h, s]

        # Goal Function (14)
        lse = quicksum(
            (v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h, s] * f[i][t][h][s] for s in set_s for h in set_h)) *
            (v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h, s] * f[i][t][h][s] for s in set_s for h in set_h))
            for t in set_t for i in set_i)
        m.setObjective(lse, GRB.MINIMIZE)

        # Constraints
        for t in set_t[:-1]:
            m.addConstr(m_theta[t], GRB.GREATER_EQUAL, m_theta[t + 1], name="C15")  # Constraint 15
            for h in set_h:
                for s in set_s[:-1]:
                    # m.addConstr(m_pi[t, h, s], GRB.GREATER_EQUAL, m_pi[t, h, s + 1], name="C12")  # Constraint 12
                    m.addConstr(m_pi[t, h, s], GRB.GREATER_EQUAL, m_pi[t + 1, h, s], name="C16")  # Constraint 16

        if silent:
            m.setParam("OutputFlag", 0)

        m.optimize()

        theta_new = deepcopy(thetas)
        pi_new = deepcopy(pis)

        for t in set_t:
            theta_new[t] = m.getVarByName("theta[" + str(t) + "]").X
            for h in set_h:
                for s in set_s:
                    pi_new[t, h, s] = m.getVarByName("pi[" + str(t) + "," + str(h) + "," + str(s) + "]").X

        # check exponential smoothing
        if exponential_smoothing:
            return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new, m.ObjVal
        else:
            return theta_new, pi_new, m.ObjVal
    except GurobiError:
        print('Error reported')

        return 0, 0, 0

#%%
def save_files(storagepath, theta_all, pi_all, *args):
    makedirs(storagepath, exist_ok=True)

    with open(storagepath + "\\thetaToUse.data", "wb") as filehandle:
        pickle.dump(theta_all[-1], filehandle)

    with open(storagepath + "\\piToUse.data", "wb") as filehandle:
        pickle.dump(pi_all[-1], filehandle)

    for o in [theta_all, pi_all, *args]:
        o_name = re.sub("[\[\]']", "", str(varnameis(o)))
        # print(o_name)
        with open(storagepath + "\\" + o_name + ".data", "wb") as filehandle:
            pickle.dump(o, filehandle)

def varnameis(v): d = globals(); return [k for k in d if d[k] is v]


# %%
# Actual Code

# generate the random sample paths (T+1 => have real life indexing starting at t=1)
np.random.seed(12)
seed(12)
customer_stream = np.random.random((I, T+1))
sales_stream = np.random.random((I, T+1))
eps_random = np.random.random((I, T+1))

# summary statistics
offer_sets_all = {str(tuple(obj)): count for count, obj in enumerate(get_offer_sets_all(products))}


# %%
# number of linear iterations
if K_lin > K:
    raise ValueError("iteration before applying plc must be smaller than total number of iterations")

for capacities in var_capacities:
    for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ",
              preferences_no_purchase, "of", str(var_no_purchase_preferences.tolist()), "starting.")

        capacities_thresholds = np.array(
            [[i * chunk_size for i in range((capacities[j] + chunk_size) // chunk_size)] for j in
             range(len(resources))])
        cap_thresholds_len = max(len(i) for i in capacities_thresholds)

        # summary statistics
        offersets_offered = np.zeros((K + 1, len(offer_sets_all)))
        products_sold = np.zeros((K + 1, len(products) + 1))
        sold_out = np.zeros((K + 1, I))
        obj_vals = np.zeros(K + 1)
        time_measured = np.zeros(K + 1)

        value_result = np.zeros((K + 1, I, T + 1))
        capacities_result = np.zeros((K + 1, I, T + 1, len(resources)))

        # Actual Code
        # K+1 policy iterations (starting with 0)
        # T time steps
        # --> theta
        # m resources
        # max(S_h) intervals
        # --> pi
        theta_all = np.zeros((K+1, T+1, 1))
        pi_all = np.array([[np.zeros(len(resources))] * (T+1)] * (K_lin + 1))
        pi_all_plc = np.array([[[np.zeros(intervals_capacities_num(capacities_thresholds) - 1)] * len(resources)] * (T+1)] *
                              (K - K_lin + 1))

        # theta and pi for each time step
        # line 1
        thetas = 0
        pis = np.zeros(len(resources))

        for k in np.arange(K) + 1:
            print(k, "of", K, "starting.")
            t_in_k_start = time()

            v_samples = np.zeros((I, T + 1))
            c_samples = np.zeros((I, T + 1, len(capacities)))

            # initialize pi plc after the number of linear iterations went through
            if k == K_lin + 1:
                tT = len(pi_all_plc[k - K_lin - 1])
                hH = len(pi_all_plc[k - K_lin - 1][0])
                for t in np.arange(tT):
                    for h in np.arange(hH):
                        pi_all_plc[k - K_lin - 1][t][h] = pi_all[k - 1][t][h]

            for i in np.arange(I):
                # line 3
                r_sample_i = np.zeros(T + 1)  # will become v_sample
                c_sample_i = np.zeros(shape=(T + 1, len(capacities)), dtype=int)

                # line 5
                c = deepcopy(capacities)  # (starting capacity at time 0)

                for t in times + 1:
                    # line 7  (starting capacity at time t)
                    c_sample_i[t - 1] = c

                    # line 8-11  (adjust bid price)
                    pis[c == 0] = np.inf
                    if k <= K_lin:
                        # linear case
                        pis[c > 0] = pi_all[k - 1, t, c > 0]
                    else:
                        # plc case (compare equation 13)
                        for h in [i for i, x in enumerate(c > 0) if x]:
                            # find the relevant interval
                            # for which index the capacity is smaller or equal (starting with upper bound of first interval)
                            # for which index the capacity is greater (ending with lower bound of last interval)
                            index_c_smaller = c[h] <= capacities_thresholds[h][1:]
                            index_c_bigger = c[h] > capacities_thresholds[h][:-1]
                            # trick to fill up correctly with false if capacity of h is not maximal capacity
                            index_match = [*(index_c_smaller & index_c_bigger), *[False for _ in np.arange(cap_thresholds_len - len(index_c_smaller) - 1)]]
                            pis[h] = pi_all_plc[k - K_lin][t][h][index_match]

                    # line 12  (epsilon greedy strategy)
                    offer_set = determine_offer_tuple(pis, epsilon[k], eps_random[i, t], revenues, A,
                                                      arrival_probabilities, preference_weights,
                                                      preferences_no_purchase)
                    offersets_offered[k, offer_sets_all[str(offer_set)]] += 1

                    # line 13  (simulate sales)
                    sold = simulate_sales(offer_set, customer_stream[i, t], sales_stream[i, t],
                                          arrival_probabilities, preference_weights, preferences_no_purchase)
                    products_sold[k, sold] += 1

                    # line 14
                    try:
                        r_sample_i[t] = revenues[sold]
                        c -= A[:, sold]
                    except IndexError:
                        # no product was sold
                        pass

                    if all(c == 0):
                        sold_out[k, i] = t
                        break

                # also adjust last sell
                c_sample_i[t] = c

                # line 16-18
                v_samples[i] = np.cumsum(r_sample_i[::-1])[::-1]
                c_samples[i] = c_sample_i


            # line 20
            if k <= K_lin:
                # linear case
                theta_all[k], pi_all[k], obj_vals[k] = update_parameters(v_samples, c_samples, theta_all[k - 1],
                                                                     pi_all[k - 1],
                                                                     k, exponential_smoothing)
            else:
                # plc case
                theta_all[k], pi_all_plc[k-K_lin], obj_vals[k] = update_parameters_plc(v_samples, c_samples, theta_all[k - 1],
                                                                         pi_all_plc[k - K_lin - 1],
                                                                         k, exponential_smoothing)

            value_result[k, :, :] = v_samples
            capacities_result[k, :, :] = c_samples

            time_measured[k] = time() - t_in_k_start

        storagepath = newpath + "\\cap" + str(capacities) + " - pref" + str(preferences_no_purchase)

        save_files(storagepath, theta_all, pi_all, pi_all_plc, value_result, capacities_result, offersets_offered,
                   products_sold, sold_out, obj_vals, time_measured, chunk_size)

# %%
wrapup(logfile, time_start, newpath)

print("I_API_pl_multi_leg.py completed.")
