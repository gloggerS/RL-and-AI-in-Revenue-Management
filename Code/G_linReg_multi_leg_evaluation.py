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
result_folder = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-linRegSingleLeg-190927-1414"
# filterwarnings("ignore")
which_policy_iteration_to_use = -1
#%%
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
b = pd.read_csv(result_folder+"\\0_settings.csv", delimiter="\t", header=None)
if not(a.equals(b)):
    raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("linReg"+"-evaluation")


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
        # get data
        with open(storedpath + "\\pi_all.data", "rb") as filehandle:
            pi_all = pickle.load(filehandle)

        pis_for_setting = pi_all[which_policy_iteration_to_use, :, :]

        # reset the data
        v_results = np.zeros_like(v_results)
        p_results = np.zeros_like(p_results)
        why_no_purchase_results = np.zeros_like(why_no_purchase_results, dtype=int)
        c_results = np.zeros_like(c_results)
        o_results = np.zeros_like(o_results)

        for k in np.arange(online_K) + 1:
            if k % 100 == 0:
                print("k: ", k)
            customer_random_stream = customer_stream[k - 1]
            sales_random_stream = sales_stream[k - 1]

            # line 3
            r_result = np.zeros(len(times))  # will become v_result
            c_result = np.zeros(shape=(len(times), len(capacities)), dtype=int)
            p_result = np.zeros(len(times))
            why_no_purchase_result = np.zeros(len(times), dtype=int)
            o_result = np.zeros(len(times))

            # line 5
            c = deepcopy(capacities)  # (starting capacity at time 0)

            # theta and pi for each time step
            # line 1
            thetas = 0
            pis = np.zeros(len(resources))

            for t in times:
                if any(c < 0):
                    raise ValueError

                c_result[t] = c

                # line 8-11  (adjust bid price)
                pis[c == 0] = np.inf
                pis[c > 0] = pis_for_setting[t, c > 0]

                # line 12  (epsilon greedy strategy)
                offer_set = determine_offer_tuple(pis, 0, 1, revenues, A,
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

print("G_linReg_multi_leg_evaluation.py completed.")