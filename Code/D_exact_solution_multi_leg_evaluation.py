"""
This script evaluates the policy
"""

from B_helper import *


#%%
result_folder = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-ES-MultiLeg-190827-1516"

#%%
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
b = pd.read_csv(result_folder+"\\0_settings.csv", delimiter="\t", header=None)
if not(a.equals(b)):
    raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("ES-evaluation")

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

# %%
# Actual Code
def get_offer_set(no_purchase_preference, c, t):
    # via index of optimal offer set
    return tuple(offer_sets.iloc[dat[str(no_purchase_preference)][t].iloc[dict_offer_sets[tuple(c)], 1]])

def get_offer_set_index(no_purchase_preference, c, t):
    # via index of optimal offer set
    return dat[str(no_purchase_preference)][t].iloc[dict_offer_sets[tuple(c)], 1]

#%%
with open(result_folder+"\\totalresults.data", "rb") as filehandle:
    dat_lookup = pickle.load(filehandle)

dict_offer_sets = {k:v for v,k in enumerate(list(dat_lookup[str(no_purchase_preference)][0].index))}

dat = deepcopy(dat_lookup)
for no_purchase_preference in var_no_purchase_preferences:
    for t in np.arange(T+1):
        dat[t] = dat[str(no_purchase_preference)][t].reset_index(drop=True)

offer_sets = pd.DataFrame(get_offer_sets_all(products))

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
    for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities), " - and - ", preferences_no_purchase, "of", str(var_no_purchase_preferences), "starting.")

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


            offerset = np.zeros([len(times), offer_sets.shape[1]])
            for t in times:
                if any(c < 0):
                    raise ValueError
                # line 7  (starting capacity at time t)
                c_result[t] = c

                offer_set = get_offer_set(preferences_no_purchase, c, t)
                offerset[t, :] = np.array(offer_set)
                o_result[t] = get_offer_set_index(preferences_no_purchase, c, t)

                # line 13  (simulate sales)
                # sold, customer = simulate_sales(offer_set, customer_random_stream[t], sales_random_stream[t],
                #                                 arrival_probabilities, preference_weights, preferences_no_purchase)
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
wrapup(logfile, time_start, newpath)

