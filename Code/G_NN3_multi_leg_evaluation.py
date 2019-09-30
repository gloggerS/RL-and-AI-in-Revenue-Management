"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *
from warnings import warn, filterwarnings

from sklearn.neural_network import MLPRegressor
# to apply or on array and for memoization (determine offer-tuple)
import functools
import operator

#%%
result_folder = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-NN3multiLeg-fixed-190929-2022"
filterwarnings("ignore")

#%%
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
b = pd.read_csv(result_folder+"\\0_settings.csv", delimiter="\t", header=None)
if not(a.equals(b)):
    raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("NN3"+"-evaluation")


capacities = var_capacities[0]
no_purchase_preference = var_no_purchase_preferences[0]

#%%
storedpath = result_folder + "\\cap" + str(capacities) + " - pref" + str(no_purchase_preference)

# open result of calculations
with open(storedpath+"\\params.data", "rb") as filehandle:
    params = pickle.load(filehandle)

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
@memoize  # memoize sinnvoll bei Zufallszahlen als Input??? :D
def determine_offer_tuple_NN(nn, ti, ca, eps, eps_ran, revenues, A, arrival_probabilities, preference_weights, no_purchase_preference):
    """
    Determines the offerset given the bid prices for each resource.

    Implement the Greedy Heuristic from Bront et al: A Column Generation Algorithm ... 4.2.2
    and extend it for the epsilon greedy strategy
    :param pi: vector of dual prices for each resource (np.inf := no capacity)
    :param eps: epsilon value for epsilon greedy strategy (eps = 0 := no greedy strategy to apply)
    :param revenues: vector of revenue for each product
    :param A: matrix with resource consumption of each product (one row = one resource)
    :param arrival_probabilities: The arrival probabilities for each customer segment
    :param preference_weights: The preference weights for each customer segment (for each product)
    :param no_purchase_preference: The no purchase preferences for each customer segment.
    :return: the offer set to be offered
    """


    # opportunity costs
    opp_costs = 1.0*np.zeros_like(revenues)
    for pro in products:
        if functools.reduce(operator.or_, ca - A[:, pro] < 0):
            # set opp costs to infty if not enough capacity for product
            opp_costs[pro] = np.inf
        else:
            # calculate opportunity costs via Bellman: V(t+1, c) - V(t+1, c-A[i])
            t_df = pd.DataFrame([np.zeros(T + 1)] * 1)
            t_df.columns = ["t" + str(i) for i in t_df.columns]
            t_df.iloc[0, ti+1] = 1

            cs_unsold = {}
            for h in resources:
                c_df = pd.DataFrame([np.zeros(T +1)] * 1)
                c_df.columns = ["c-h" + str(h) + "-t" + str(i) for i in c_df.columns]
                c_df.iloc[0, ti+1] = ca[h]
                cs_unsold[h] = c_df

            cs_sold = {}
            for h in resources:
                c_df = pd.DataFrame([np.zeros(T + 1)] * 1)
                c_df.columns = ["c-h" + str(h) + "-t" + str(i) for i in c_df.columns]
                c_df.iloc[0, ti + 1] = ca[h] - A[h, pro]
                cs_sold[h] = c_df

            X_unsold = pd.concat([t_df, *[cs_unsold[h] for h in resources]], axis=1)
            X_sold = pd.concat([t_df, *[cs_sold[h] for h in resources]], axis=1)

            opp_costs[pro] = nn.predict(X_unsold) - nn.predict(X_sold)

    # epsilon greedy strategy - offer no products
    if eps_ran < eps / 2:
        return tuple(np.zeros_like(revenues))

    # epsilon greedy strategy - offer all products
    if eps_ran < eps:
        offer_tuple = np.ones_like(revenues)
        offer_tuple[opp_costs == np.inf] = 0  # one resource not available => don't offer product
        return tuple(offer_tuple)

    # setup
    offer_tuple = np.zeros_like(revenues)

    # line 1
    s_prime = revenues - opp_costs > 0
    if all(np.invert(s_prime)):
        return tuple(offer_tuple)

    # line 2-3
    # offer_sets_to_test has in each row an offer set, we want to test
    offer_sets_to_test = np.zeros((sum(s_prime), len(revenues)))
    offer_sets_to_test[np.arange(sum(s_prime)), np.where(s_prime)] = 1
    offer_sets_to_test += offer_tuple
    offer_sets_to_test = (offer_sets_to_test > 0)

    value_marginal = np.apply_along_axis(calc_value_marginal_nn, 1, offer_sets_to_test, opp_costs, revenues,
                                         arrival_probabilities, preference_weights, no_purchase_preference)

    offer_tuple = offer_sets_to_test[np.argmax(value_marginal)]*1
    s_prime = s_prime & offer_tuple == 0
    v_s = np.amax(value_marginal)

    # line 4
    while True:
        # 4a
        # offer_sets_to_test has in each row an offer set, we want to test
        offer_sets_to_test = np.zeros((sum(s_prime), len(revenues)))
        offer_sets_to_test[np.arange(sum(s_prime)), np.where(s_prime)] = 1
        offer_sets_to_test += offer_tuple
        offer_sets_to_test = (offer_sets_to_test > 0)

        # 4b
        value_marginal = np.apply_along_axis(calc_value_marginal_nn, 1, offer_sets_to_test, opp_costs, revenues,
                                             arrival_probabilities, preference_weights, no_purchase_preference)

        if np.amax(value_marginal) >= v_s:
            v_s = np.amax(value_marginal)
            offer_tuple = offer_sets_to_test[np.argmax(value_marginal)]*1  # to get 1 for product offered
            s_prime = (s_prime - offer_tuple) == 1  # only those products remain, that are neither in the offer_tuple
            if all(offer_tuple == 1):
                break
        else:
            break
    return tuple(offer_tuple)

@memoize
def calc_value_marginal_nn(indices_inner_sum, opp_costs, revenues, arrival_probabilities, preference_weights, no_purchase_preference):
    """
    Calculates the marginal value as indicated at Bront et al, 4.2.2 Greedy Heuristic -> step 4a

    :param indices_inner_sum: C_l intersected with (S union with {j})
    :param pi: vector of dual prices for each resource (np.inf := no capacity)
    :param revenues: vector of revenue for each product
    :param A: matrix with resource consumption of each product (one row = one resource)
    :param arrival_probabilities: The arrival probabilities for each customer segment
    :param preference_weights: The preference weights for each customer segment (for each product)
    :param no_purchase_preference: The no purchase preferences for each customer segment.
    :return: The value inside the argmax (expected marginal value given one set of products to offer)
    """
    v_temp = 0
    for l in np.arange(len(preference_weights)):  # sum over all customer segments
        tmp_nan_remove = indices_inner_sum * (revenues - opp_costs)
        where_are_nans = np.isnan(tmp_nan_remove)
        tmp_nan_remove[where_are_nans] = 0
        v_temp += arrival_probabilities[l] * \
                  sum(tmp_nan_remove * preference_weights[l, :]) / \
                  (sum(indices_inner_sum * preference_weights[l, :]) + no_purchase_preference[l])
    return v_temp

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

        storedpath = result_folder + "\\cap" + str(capacities) + " - pref" + str(no_purchase_preference)
        # open result of calculations
        with open(storedpath + "\\params.data", "rb") as filehandle:
            params = pickle.load(filehandle)

        with open(storedpath + "\\nn.data", "rb") as filehandle:
            nn = pickle.load(filehandle)


        # reset the data
        v_results = np.zeros_like(v_results)
        p_results = np.zeros_like(p_results)
        why_no_purchase_results = np.zeros_like(why_no_purchase_results, dtype=int)
        c_results = np.zeros_like(c_results)
        o_results = np.zeros_like(o_results)


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

            # line 5
            c = deepcopy(capacities)  # (starting capacity at time 0)


            for t in times:
                if any(c < 0):
                    raise ValueError

                c_result[t] = c

                offer_set = determine_offer_tuple_NN(nn, t, c, 0, 1, revenues, A,
                                                         arrival_probabilities, preference_weights,
                                                         no_purchase_preference)
                o_result[t] = os_dict[offer_set]

                # line 13  (simulate sales)
                # sold, customer = simulate_sales(offer_set, customer_random_stream[t], sales_random_stream[t],
                #                                 arrival_probabilities, preference_weights, no_purchase_preference)
                sold, why_no_purchase = simulate_sales_evaluation(offer_set, customer_random_stream[t],
                                                                  sales_random_stream[t], arrival_probabilities,
                                                                  preference_weights, no_purchase_preference)
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
            why_no_purchase_results[k - 1] = why_no_purchase_result
            o_results[k - 1] = o_result

        value_final['' + str(capacities) + '-' + str(no_purchase_preference)] = pd.DataFrame(v_results[:, 0])
        capacities_final['' + str(capacities) + '-' + str(no_purchase_preference)] = pd.DataFrame(c_results[:, -1, :])
        products_all['' + str(capacities) + '-' + str(no_purchase_preference)] = p_results
        why_no_purchase_all['' + str(capacities) + '-' + str(no_purchase_preference)] = why_no_purchase_results
        offersets_all['' + str(capacities) + '-' + str(no_purchase_preference)] = o_results



# %%
# write result of calculations
save_files(newpath, value_final, capacities_final, products_all, offersets_all, why_no_purchase_all)

# %%
print("\n\nData used:\n", result_folder, file=logfile)
wrapup(logfile, time_start, newpath)
print("G_NN3_evaluation.py completed.")

# #%%
# import shelve
#
# filename=newpath+'/shelve.out'
# my_shelf = shelve.open(filename,'n') # 'n' for new
#
# for key in dir():
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         #
#         # __builtins__, my_shelf, and imported modules can not be shelved.
#         #
#         print('ERROR shelving: {0}'.format(key))
#     except:
#         print("Error")
# my_shelf.close()