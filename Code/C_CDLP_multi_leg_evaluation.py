"""
This script will calculate the CDLP (choice based deterministic linear programme).
"""

from B_helper import *

# from joblib import Parallel, delayed, dump, load
# import multiprocessing

#%%
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("CDLP-evaluation")

capacities = var_capacities[0]
no_purchase_preference = var_no_purchase_preferences[0]

#%%
# generate the random sample paths (T+1 => have real life indexing starting at t=1)
# online_K = 100
np.random.seed(123)
seed(123)
customer_stream = np.random.random((online_K, T+1))
sales_stream = np.random.random((online_K, T+1))

def save_files(storagepath, *args):
    # makedirs(storagepath)

    for o in [*args]:
        o_name = re.sub("[\[\]']", "", str(varnameis(o)))
        # print(o_name)
        with open(storagepath + "\\" + o_name + ".data", "wb") as filehandle:
            pickle.dump(o, filehandle)

def varnameis(v): d = globals(); return [k for k in d if d[k] is v]

#%%
def revenue(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities, revenues):
    """
    R(S)

    :param offer_set_tuple: S
    :return: R(S)
    """
    return sum(revenues * customer_choice_vector(offer_set_tuple, preference_weights, preference_no_purchase,
                                               arrival_probabilities)[:-1])


def quantity_i(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities, i, A):
    """
    Q_i(S)

    :param offer_set_tuple: S
    :param i: resource i
    :return: Q_i(S)
    """
    return sum(A[i, :] * customer_choice_vector(offer_set_tuple, preference_weights, preference_no_purchase,
                                              arrival_probabilities)[:-1])


# Actual Code
def CDLP(capacities, preference_no_purchase, offer_sets: np.ndarray, filename_result="", silent=True):
    """
    Implements (4) of Bront et al. Needs the offer-sets to look at (N) as input.

    :param offer_sets: N
    :return: dictionary of (offer set, time offered), optimal value, dual prices of resources, dual price of time
    """
    offer_sets = pd.DataFrame(offer_sets)
    offer_sets = offer_sets.append(pd.DataFrame([np.zeros_like(products)]))  # append the empty set to ensure all times some offerset is offered
    offer_sets = offer_sets.reset_index(drop=True)
    T = len(times)

    S = {}
    R = {}
    Q = {}
    for index, offer_array in offer_sets.iterrows():
        S[index] = tuple(offer_array)
        R[index] = revenue(tuple(offer_array), preference_weights, preference_no_purchase, arrival_probabilities,
                           revenues)
        temp = {}
        for i in resources:
            temp[i] = quantity_i(tuple(offer_array), preference_weights, preference_no_purchase,
                                 arrival_probabilities, i, A)
        Q[index] = temp

    try:
        m = Model()

        # Variables
        mt = m.addVars(offer_sets.index.values, name="t", lb=0.0)  # last constraint

        # Objective Function
        m.setObjective(quicksum(R[s] * mt[s] for s in offer_sets.index.values), GRB.MAXIMIZE)

        mc = {}
        # Constraints
        for i in resources:
            mc[i] = m.addConstr(quicksum(Q[s][i] * mt[s] for s in offer_sets.index.values), GRB.LESS_EQUAL,
                                capacities[i],
                                name="constraintOnResource")
        msigma = m.addConstr(quicksum(mt[s] for s in offer_sets.index.values), GRB.EQUAL, T)

        if silent:
            m.setParam("OutputFlag", 0)

        m.optimize()

        ret = {}
        # have to get to index to reproduce the original offerset
        pat = r".*?\[(.*)\].*"
        for v in m.getVars():
            if v.X > 0:
                match = re.search(pat, v.VarName)
                erg_index = match.group(1)

                ret[int(erg_index)] = (tuple(offer_sets.loc[int(erg_index)]), v.X)

        dualPi = np.zeros_like(resources, dtype=float)
        for i in resources:
            dualPi[i] = mc[i].pi
        dualSigma = msigma.pi

        valOpt = m.objVal

        return ret, valOpt, dualPi, dualSigma

    except GurobiError:
        print('Error reported')


def column_MIP(preference_no_purchase, pi, w=0, silent=True):  # pass w to test example for greedy heuristic
    """
    Implements MIP formulation on p. 775 lhs

    :param pi:
    :param w:
    :return: optimal tuple of products to offer
    """
    # TODO: warum +1 zum Schluss?
    M = 1/preference_no_purchase.min()+1

    if isinstance(w, int) and w == 0:  # 'and' is lazy version of &
        w = np.zeros_like(revenues, dtype=float)
        for j in products:
            w[j] = revenues[j] - sum(A[:, j]*pi)

    try:
        m = Model()

        mx = {}
        my = {}
        mz = {}

        # Variables
        for j in products:
            my[j] = m.addVar(0, 1, vtype=GRB.BINARY, name="y["+str(j)+"]")
        for l in customer_segments:
            mx[l] = m.addVar(0.0, name="x["+str(l)+"]")
            temp = {}
            for j in products:
                temp[j] = m.addVar(0.0, name="z["+str(l)+","+str(j)+"]")
            mz[l] = temp

        # Objective
        m.setObjective(quicksum(arrival_probabilities[l] * w[j] * preference_weights[l, j] * mz[l][j]
                                for l in customer_segments for j in products), GRB.MAXIMIZE)

        # Constraints
        mc1 = m.addConstrs((mx[l]*preference_no_purchase[l] +
                            quicksum(preference_weights[l, j]*mz[l][j] for j in products) == 1
                            for l in customer_segments), name="mc1")
        mc2 = m.addConstrs((mx[l] - mz[l][j] <= M - M*my[j] for l in customer_segments for j in products),
                           name="mc2")
        mc3 = m.addConstrs((mz[l][j] <= mx[l] for l in customer_segments for j in products), name="mc3")
        mc4 = m.addConstrs((mz[l][j] <= M*my[j] for l in customer_segments for j in products), name="mc4")

        if silent:
            m.setParam("OutputFlag", 0)

        m.optimize()

        y = np.zeros_like(revenues)
        for j in products:
            y[j] = my[j].x

        return tuple(y), m.objVal

    except GurobiError:
        print('Error reported')


def column_greedy(preference_no_purchase, pi, w=0):  # pass w to test example for greedy heuristic
    """
    Implements Greedy Heuristic on p. 775 rhs

    :param pi:
    :param w:
    :return: heuristically optimal tuple of products to offer
    """
    # Step 1
    y = np.zeros_like(revenues)

    if isinstance(w, int) and w == 0:  # and is lazy version of &
        w = np.zeros_like(revenues, dtype=float)  # calculate the opportunity costs
        for j in products:
            w[j] = revenues[j] - sum(A[:, j]*pi)

    # Step 2
    Sprime = set(np.where(w > 0)[0])

    # Step 3
    value_marginal = np.zeros_like(w, dtype=float)
    for j in Sprime:
        for l in customer_segments:
            value_marginal[j] += arrival_probabilities[l] * preference_weights[l, j]/(preference_weights[l, j] + preference_no_purchase[l])
        value_marginal[j] *= w[j]

    jstar = np.argmax(value_marginal)
    v_new = value_marginal[jstar]

    S = {jstar}
    Sprime = Sprime-S

    # Step 4
    while True:
        v_akt = deepcopy(v_new)  # deepcopy to be on the safe side
        v_temp = np.zeros_like(revenues, dtype=float)  # uses more space then necessary, but simplifies indices below
        for j in Sprime:
            for l in customer_segments:
                z = 0
                n = 0
                for p in S.union({j}):
                    z += w[p]*preference_weights[l, p]
                    n += preference_weights[l, p]
                n += preference_no_purchase[l]
                v_temp[j] += arrival_probabilities[l]*z/n
        jstar = np.argmax(value_marginal)  # argmax always returns index of first maxima (if there is > 1)
        v_new = value_marginal[jstar]
        if v_new > v_akt:
            S = S.union({jstar})
            Sprime = Sprime - {jstar}
        else:
            break

    # Step 5
    y[list(S)] = 1
    return tuple(y), v_new


# CDLP by column generation  # changed to silent version
def CDLP_by_column_generation(capacities, preference_no_purchase, filename_result):
    """
    Implements Bront et als approach for CDLP by column generation as pointed out on p. 775 just above "5. Decomp..."

    :return:
    """
    # print("Start CDLP by column generation.")
    dual_pi = np.zeros(len(A))

    col_offerset, col_val = column_greedy(preference_no_purchase, dual_pi)
    if all(col_offerset == np.zeros_like(col_offerset)):
        # print("MIP solution used to solve CDLP by column generation")
        col_offerset, col_val = column_MIP(preference_no_purchase, dual_pi)

    offer_sets = pd.DataFrame([np.array(col_offerset)])

    val_akt_CDLP = 0
    ret, val_new_CDLP, dual_pi, dual_sigma = CDLP(capacities, preference_no_purchase, np.array(offer_sets))
    data_result = pd.DataFrame([{"val": val_new_CDLP, "optimal sets": ret,
                                 "dual pi": dual_pi, "dual sigma": dual_sigma}])

    count = 1  # counts how often the while loop runs
    while val_new_CDLP > val_akt_CDLP:
        count += 1
        # print("Actual value of CDLP: \t", val_new_CDLP)
        val_akt_CDLP = deepcopy(val_new_CDLP)  # deepcopy and new name to be on the safe side

        col_offerset, col_val = column_greedy(preference_no_purchase, dual_pi)
        if not offer_sets[(offer_sets == np.array(col_offerset)).all(axis=1)].index.empty:
            col_offerset, col_val = column_MIP(preference_no_purchase, dual_pi)
            if not offer_sets[(offer_sets == np.array(col_offerset)).all(axis=1)].index.empty:
                break  # nothing changed

        offer_sets = offer_sets.append([np.array(col_offerset)], ignore_index=True)
        ret, val_new_CDLP, dual_pi, dual_sigma = CDLP(capacities, preference_no_purchase, np.array(offer_sets))

    return ret, val_new_CDLP, dual_pi, dual_sigma

#%%
def checkup(offer_set, c, A):
    num_corrections = 0
    os_array = np.array(offer_set)
    for p in products[os_array==1]:
        if any(c[A[:,p] == 1] == 0):
            # not enough capacity to offer product p
            os_array[p] = 0
            num_corrections += 1
    return tuple(os_array), num_corrections

# to store when how many corrections occured
num_corrections_all = {}
num_corrections_results = np.zeros((online_K, len(times)))

#%%
os_dict = {tuple(v):k for k,v in enumerate(get_offer_sets_all(products))}

#%%
# online_K+1 policy iterations (starting with 0)
# values, products sold, offersets offered, capacities remaining
v_results = np.array([np.zeros(len(times))]*online_K)
p_results = np.array([np.zeros(len(times))]*online_K)
why_no_purchase_results = np.array([np.zeros(len(times))]*online_K)
o_results = np.zeros((online_K, len(times)))  # same as above, higher skill level
c_results = np.array([np.zeros(shape=(len(times), len(capacities)))]*online_K)

value_final = pd.DataFrame(v_results[:, 0])  # setup result storage empty
capacities_final = {}
products_all = {}
offersets_all = {}
why_no_purchase_all = {}

#%%
for capacities in var_capacities:
    for no_purchase_preference in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities), " - and - ", no_purchase_preference, "of", str(var_no_purchase_preferences), "starting.")

        # reset the data
        v_results = np.zeros_like(v_results)
        p_results = np.zeros_like(p_results)
        why_no_purchase_results = np.zeros_like(why_no_purchase_results, dtype=int)
        c_results = np.zeros_like(c_results)
        o_results = np.zeros_like(o_results)

        num_corrections_results = np.zeros_like(num_corrections_results)

        for k in np.arange(online_K)+1:
            if k % 100 == 0:
                print("k: ", k)
            customer_random_stream = customer_stream[k-1]
            sales_random_stream = sales_stream[k-1]

            # line 3
            r_result = np.zeros(len(times))  # will become v_result
            c_result = np.zeros(shape=(len(times), len(capacities)), dtype=int)
            p_result = np.zeros(len(times))
            why_no_purchase_result = np.zeros(len(times), dtype=int)
            o_result = np.zeros(len(times))

            num_corrections_result = np.zeros(len(times))

            # line 5
            c = deepcopy(capacities)  # (starting capacity at time 0)

            # get results from CDLP and analyse ret to get offersets
            # determine which offerset is offered at time t and put it into offersets_tobeoffered[t]
            ret, val_new_CDLP, dual_pi, dual_sigma = CDLP_by_column_generation(c, no_purchase_preference, "")
            offersets_tobeoffered = {t:-1 for t in times}
            offerset_sets = np.arange(len(ret))
            offerset_times = np.arange(len(ret))
            i = 0
            for ke in ret.keys():
                offerset_times[i] = round(ret[ke][1])
                offerset_sets[i] = ke
                i += 1
            if sum(offerset_times) != T:
                raise ValueError("rounded times to offer sets don't add up to total length of time horizon")

            helper = np.repeat(offerset_sets, offerset_times)
            times_to_offer = np.zeros_like(times)
            for t in times:
                offersets_tobeoffered[t] = ret[helper[t]][0]

            for t in times:
                if any(c < 0):
                    raise ValueError
                # line 7  (starting capacity at time t)
                c_result[t] = c

                offer_set = offersets_tobeoffered[t]
                offer_set, num_corrections = checkup(offer_set, c, A)
                o_result[t] = os_dict[offer_set]
                num_corrections_result[t] = num_corrections

                # line 13  (simulate sales)
                # sold, customer = simulate_sales(offer_set, customer_random_stream[t], sales_random_stream[t],
                #                                 arrival_probabilities, preference_weights, no_purchase_preference)
                sold, why_no_purchase = simulate_sales_evaluation(offer_set, customer_random_stream[t], sales_random_stream[t], arrival_probabilities, preference_weights, no_purchase_preference)
                p_result[t] = sold
                why_no_purchase_result[t] = why_no_purchase

                # line 14
                try:
                    r_result[t] = revenues[sold]
                    c -= A[:, sold]
                except IndexError:
                    # no product was sold
                    pass

            # line 16-18
            v_results[k - 1] = np.cumsum(r_result[::-1])[::-1]
            c_results[k - 1] = c_result
            p_results[k - 1] = p_result
            why_no_purchase_results[k-1] = why_no_purchase_result
            o_results[k - 1] = o_result

            num_corrections_results[k - 1] = num_corrections_result

        value_final['' + str(capacities) + '-' + str(no_purchase_preference)] = pd.DataFrame(v_results[:, 0])
        capacities_final['' + str(capacities) + '-' + str(no_purchase_preference)] = pd.DataFrame(c_results[:, -1, :])
        products_all['' + str(capacities) + '-' + str(no_purchase_preference)] = p_results
        why_no_purchase_all['' + str(capacities) + '-' + str(no_purchase_preference)] = why_no_purchase_results
        offersets_all['' + str(capacities) + '-' + str(no_purchase_preference)] = o_results

        num_corrections_all['' + str(capacities) + '-' + str(no_purchase_preference)] = num_corrections_results



# %%
# write result of calculations
save_files(newpath, value_final, capacities_final, products_all, offersets_all, num_corrections_all, why_no_purchase_all)

# %%
wrapup(logfile, time_start, newpath)

