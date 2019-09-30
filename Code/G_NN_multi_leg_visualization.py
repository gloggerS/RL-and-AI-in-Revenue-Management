"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *
import seaborn as sns
import matplotlib.patches as mpatches

#%%
result_folder = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-NN3multiLeg-r2-190928-2013"

#%%
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
b = pd.read_csv(result_folder+"\\0_settings.csv", delimiter="\t", header=None)
if not(a.equals(b)):
    raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("NNMultiLeg-Visualization")

capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]
no_purchase_preference = preferences_no_purchase


#%%
# offersets_offered.data
# sold_out.data
# time_measured.data
result_folder = result_folder + "\\cap" + str(capacities) + " - pref" + str(preferences_no_purchase)

with open(result_folder+"\\nn.data", "rb") as filehandle:
    nn = pickle.load(filehandle)


# %%

# open result of calculations
with open(result_folder+"\\best_loss.data", "rb") as filehandle:
    best_loss = pickle.load(filehandle)

with open(result_folder+"\\n_iter.data", "rb") as filehandle:
    n_iter = pickle.load(filehandle)

with open(result_folder+"\\value_result.data", "rb") as filehandle:
    value_result = pickle.load(filehandle)

with open(result_folder+"\\capacities_result.data", "rb") as filehandle:
    capacities_result = pickle.load(filehandle)

with open(result_folder+"\\products_sold.data", "rb") as filehandle:
    p = pickle.load(filehandle)
    p = p[1:]

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


#%% products
fig, ax = plt.subplots()
for i in products:
    ax.plot(np.arange(K)+1, p[:, i], label="Product "+str(i+1), linewidth=2.0)
ax.plot(np.arange(K)+1, p[:, i+1], label="No Purchase", linewidth=2.0)
ax.legend(bbox_to_anchor=(0, -.4, 1, .102), loc="lower left",
          ncol=3, mode="expand")
plt.xticks(np.append([1], np.arange(5, K+1, 5)))
plt.xlabel("Policy Iteration k")
plt.ylabel("Total number of products sold")
fig.savefig(result_folder+"\\plotProducts1.pdf", bbox_inches="tight")
plt.show()


#%% products
bottom = np.zeros(K)
for i in products:
    plt.bar(np.arange(K)+1, p[:, i], label="Product "+str(i+1), bottom=bottom)
    bottom += p[:, i]
plt.legend(bbox_to_anchor=(0, -.4, 1, .102), loc="lower left",
          ncol=3, mode="expand")
plt.xticks(np.append([1], np.arange(5, K + 1, 5)))
plt.xlabel("Policy Iteration k")
plt.ylabel("Total number of products sold")
plt.savefig(result_folder+"\\plotProducts2.pdf", bbox_inches="tight")
plt.show()

#%% values at start (for all K, for all I)
v = pd.DataFrame(value_result[:, :, 0])
v = v.iloc[1:, :]
    
#%%
v.apply(sum, axis=1)

tmp = np.mean(v, axis=1)

plt.plot(np.arange(K)+1, np.mean(v, axis=1), label="Average value at start")
plt.legend(loc="lower right")
plt.xticks(np.append([1], np.arange(5, K+1, 5)))
plt.xlabel("Policy Iteration k")
plt.ylabel("Average value")
plt.savefig(result_folder+"\\plotValue.pdf", bbox_inches="tight")
plt.show()

#%% time sold out (not needed yet, as never sold out in exampleStefan)
with open(result_folder+"\\sold_out.data", "rb") as filehandle:
    sold_out = pickle.load(filehandle)


#%% capacities remaining
c = capacities_result

# TODO change to the above
capacity_last_t = c[:, :, -1, :]
# capacity_last_t = c[:, :, -2, :]

capacity_last_t_average = np.average(capacity_last_t, 1)
capacity_last_t_average = capacity_last_t_average[1:, :]

#%%
def plot_remaining_capacity(capacity_last_t_average, ks):
    bar_width = 1 / (capacity_last_t_average.shape[-1] + 1)
    r = 1.0 * np.arange(len(ks))
    plt.bar(r, capacity_last_t_average[ks, 0], label="Resource 1", width=bar_width, edgecolor="white")
    for h in resources[1:]:
        r += bar_width
        plt.bar(r, capacity_last_t_average[ks, h], label="Resource " + str(h + 1), width=bar_width, edgecolor="white")
    plt.xticks(r - 1.5 * bar_width, ks + 1)
    plt.ylim([0, 1.1 * capacity_last_t_average.max()])
    plt.legend(bbox_to_anchor=(0, -.26, 1, .102), loc="lower left",
               ncol=4, mode="expand")
    plt.xlabel("Policy Iteration k")
    plt.ylabel("Amount left over")
    plt.savefig(result_folder + "\\remainingCapacity" + str(ks[0]) + ".pdf", bbox_inches="tight")
    plt.show()

#%%
ks = np.arange(0, 10)
plot_remaining_capacity(capacity_last_t_average, ks)

#%%
ks = np.arange(50, 60)
plot_remaining_capacity(capacity_last_t_average, ks)

#%%
label_capacities = ["Resource "+str(i+1) for i in resources]
df_capacities = pd.DataFrame(np.round(capacity_last_t_average, 2), columns=label_capacities, index=(np.arange(K)+1))

erg_latex = open(result_folder+"\\remainingCapacity.txt", "w+")  # write and create (if not there)
print(df_capacities.to_latex(), file=erg_latex)
erg_latex.close()

df_capacities
#%% offersets
with open(result_folder+"\\offersets_offered.data", "rb") as filehandle:
    offer_sets = pickle.load(filehandle)

label_relevant = [str((products+1)[a==1]) for a in get_offer_sets_all(products)[offer_sets.sum(axis=0)>0]]

offer_sets_relevant = offer_sets[1:, offer_sets.sum(axis=0)>0]

df = pd.DataFrame(offer_sets_relevant, columns=[re.sub(" ", ",", str(l)) for l in label_relevant], index=(np.arange(K)+1), dtype=int)

erg_latex = open(result_folder+"\\offersetsOffered.txt", "w+")  # write and create (if not there)
print(df.to_latex(), file=erg_latex)
erg_latex.close()

df

# #%% offersets visualization
# pi1 = pi_result[-1]
# pi2 = pi_result[-2]
#
#
# def plot_pi(pi, name):
#     df = pd.DataFrame(pi, columns=resources + 1)
#     sns.heatmap(df, annot=True, cmap="Blues_r")
#     plt.xlabel("Resource")
#     plt.ylabel("Time")
#     plt.savefig(result_folder + "\\" + name + ".pdf", bbox_inches="tight")
#     plt.show()


# #%% visualize the values of the optimized variables
# plot_pi(pi1, "pi60")
# plot_pi(pi2, "pi59")
#
# plot_pi(np.abs(pi1-pi2), "piAbsDifference")
#%% visualize offersets (erstmal nur für exampleStefan)
dat = np.zeros((9*5*5*9, 20))
c1 = capacities[0]+1
c2 = capacities[1]+1
c3 = capacities[2]+1
c4 = capacities[3]+1

offer_dict = {tuple(b):a for a, b in enumerate(get_offer_sets_all(products))}

for h1 in np.arange(c1):
    for h2 in np.arange(c2):
        for h3 in np.arange(c3):
            for h4 in np.arange(c4):
                for t in times+1:
                    c = np.array([h1, h2, h3, h4])
                    offer_tuple = determine_offer_tuple_NN(nn, t - 1, c, 0, 1, revenues, A,
                                             arrival_probabilities, preference_weights, no_purchase_preference)
                    dat[h1*(c2*c3*c4) + h2*(c3*c4) + h3*(c4) + h4, t-1] = offer_dict[offer_tuple]

dat = dat.astype(int)

sns.heatmap(dat)
plt.show()


#%%
all_values = np.unique(dat)
n = len(all_values)


def plot_offersets_heat(all_values, n, dat, name):
    # create dictionary with value to integer mappings
    value_to_int = {value: i for i, value in enumerate(all_values)}

    d = copy(dat)
    for i in np.arange(len(d)):
        for j in np.arange(len(d[0])):
            d[i, j] = value_to_int[d[i, j]]

    cmap = sns.color_palette("colorblind", n)
    ax = sns.heatmap(d, cmap=cmap)
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    label_relevant = [str((products + 1)[a == 1]) for a in get_offer_sets_all(products)[all_values]]
    labels = [re.sub(" ", ",", str(l)) for l in label_relevant]
    colorbar.set_ticklabels(labels)
    plt.xlabel("Time")
    plt.ylabel("Combinations of remaining capacity")
    plt.savefig(result_folder + "\\" + name +".pdf", bbox_inches="tight")
    plt.show()

#%%
plot_offersets_heat(all_values, n, dat, "offersetsAll")

#%% num_iterations
with open(result_folder+"\\n_iter.data", "rb") as filehandle:
    d = pickle.load(filehandle)

plt.plot(np.arange(K)+1, d[:-1])
plt.xticks(np.append([1], np.arange(5, K+1, 5)))
plt.xlabel("Policy Iteration k")
plt.ylabel("Number of iterations")
plt.savefig(result_folder+"\\numIterations.pdf", bbox_inches="tight")
plt.show()


# %%
wrapup(logfile, time_start, newpath)

print("NN_multi_leg_visualization.py completed.")

#%%
import shelve

filename=newpath+'/shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
    except:
        print("Error")
my_shelf.close()