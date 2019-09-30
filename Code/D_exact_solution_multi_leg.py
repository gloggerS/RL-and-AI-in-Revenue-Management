"""
This script will calculate the exact solution to our optimization problem and store them in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *


#%%
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
    epsilon, exponential_smoothing \
    = setup("ES-MultiLeg")

# capacities max: run it just for one set of capacities, the rest follows)
capacity_max = np.max(var_capacities, axis=0)
capacity_max.dtype = "int32"


# %%
# Actual Code
# needed results: for each time and capacity: expected value of revenues to go, optimal set to offer,
# list of other optimal sets
def return_raw_data_per_time(capacity):
    capacity_product = np.array(list(product(*[range(i+1) for i in capacity])))
    index = pd.MultiIndex.from_arrays(capacity_product.T, names=["c"+str(i) for i in range(len(capacity))])
    df = pd.DataFrame(index=index, columns=['value', 'offer_set_optimal', 'num_offer_set_optimal'])
    df.value = pd.to_numeric(df.value)
    return df

def save_files(storagepath, *args):
    makedirs(storagepath)

    for o in [*args]:
        o_name = re.sub("[\[\]']", "", str(varnameis(o)))
        # print(o_name)
        with open(storagepath + "\\" + o_name + ".data", "wb") as filehandle:
            pickle.dump(o, filehandle)

def varnameis(v): d = globals(); return [k for k in d if d[k] is v]

# %%
# agglomerating the ultimate results
total_results = {}
capacity_empty = tuple(np.zeros_like(capacity_max))
capacity_product = np.array(list(product(*[range(i+1) for i in capacity_max])))

for no_purchase_preference in var_no_purchase_preferences:
    T = len(times)
    final_results = list(return_raw_data_per_time(capacity_max) for t in np.arange(T + 1))
    final_results[T].value = 0.0
    final_results[T].offer_set_optimal = 0
    final_results[T].num_offer_set_optimal = 0

    offer_sets = pd.DataFrame(get_offer_sets_all(products))
    prob = offer_sets.apply(tuple, axis=1)
    prob = prob.apply(customer_choice_vector, args=(preference_weights, no_purchase_preference, arrival_probabilities))
    prob = pd.DataFrame(prob.values.tolist(), columns=np.arange(len(products) + 1))  # probabilities for each product

    prob = np.array(prob)
    # add the probability of no arrival to no-purchase
    prob[:, -1] = 1 - np.apply_along_axis(sum, 1, prob[:, :-1])

    for t in times[::-1]:  # running through the other way round, starting from second last time
        print("Time point: ", t)

        # c = 0
        final_results[t].loc[capacity_empty, :] = (0.0, len(offer_sets)-1, 0)  # no resources => empty offerset

        # c > 0
        for c in capacity_product[1:]:
            # print(c)
            tmp_value = .0
            tmp_offer_set_optimal = 0
            tmp_num_offer_set_optimal = 0

            for o, offer_set in offer_sets[:-1].iterrows():
                # indices_c_minus_A = np.repeat(c, repeats=len(revenues), axis=0)
                # v = prob[o]*(np.append(revenues + final_results[t+1].loc[[-A], "values"]))
                v = 0.0
                for j in products[offer_set==1]:
                    i = c-A[:, j]
                    if all(i >= 0):
                        v += prob[o][j]*(revenues[j] + final_results[t+1].loc[tuple(i), "value"])
                v += prob[o][-1]*final_results[t+1].loc[tuple(c), "value"]

                if v > tmp_value:
                    tmp_value = v
                    tmp_offer_set_optimal = o
                    tmp_num_offer_set_optimal = 1
                if v == tmp_value:
                    tmp_num_offer_set_optimal += 1

            final_results[t].loc[tuple(c), :] = (tmp_value, tmp_offer_set_optimal, tmp_num_offer_set_optimal)

    storagepath = newpath + "\\cap" + str(capacity_max) + " - pref" + str(no_purchase_preference)

    save_files(storagepath, final_results)
    total_results[str(no_purchase_preference)] = deepcopy(final_results)

# %%
# write result of calculations
with open(newpath+"\\totalresults.data", "wb") as filehandle:
    pickle.dump(total_results, filehandle)

# %%
# write summary latex
erg_paper = pd.DataFrame(index=range(len(var_no_purchase_preferences)*len(var_capacities)),
                         columns=["capacity", "no-purchase preference", "DP-value", "DP-optimal offer set at start"])
i = 0
for c in var_capacities:
    for u in var_no_purchase_preferences:
        tmp = total_results[str(u)][0].loc[tuple(c), :]
        erg_paper.iloc[i, :] = (tuple(c), tuple(u),
                                round(float(tmp.value), 2),
                                str(tuple(offer_sets.iloc[int(tmp.offer_set_optimal), :])))
        i += 1
erg_latex = open(newpath+"\\erg_paper.txt", "w+")  # write and create (if not there)
print(erg_paper.to_latex(), file=erg_latex)
erg_latex.close()


# %%
wrapup(logfile, time_start, newpath)
