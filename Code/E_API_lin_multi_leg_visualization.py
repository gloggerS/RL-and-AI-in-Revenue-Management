"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *
import seaborn as sns
import matplotlib.patches as mpatches

#%%
result_folder = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-APILinearMultiLeg-190924-1450"

#%%
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
b = pd.read_csv(result_folder+"\\0_settings.csv", delimiter="\t", header=None)
if not(a.equals(b)):
    raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("APILinearMultiLeg-Visualization")

capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]


# %%
result_folder = result_folder + "\\cap" + str(capacities) + " - pref" + str(preferences_no_purchase)

# open result of calculations
with open(result_folder+"\\theta_all.data", "rb") as filehandle:
    theta_result = pickle.load(filehandle)

with open(result_folder+"\\pi_all.data", "rb") as filehandle:
    pi_result = pickle.load(filehandle)

with open(result_folder+"\\value_result.data", "rb") as filehandle:
    value_result = pickle.load(filehandle)

with open(result_folder+"\\capacities_result.data", "rb") as filehandle:
    capacities_result = pickle.load(filehandle)

with open(result_folder+"\\thetaToUse.data", "rb") as filehandle:
    theta_to_use = pickle.load(filehandle)

with open(result_folder+"\\piToUse.data", "rb") as filehandle:
    pi_to_use = pickle.load(filehandle)

with open(result_folder+"\\products_sold.data", "rb") as filehandle:
    p = pickle.load(filehandle)
    p = p[1:]
    
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
    plt.ylim([0, 1.1*capacity_last_t_average.max()])
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

#%%
np.random.seed(12)
seed(12)
customer_stream = np.random.random((I, T+1))
sales_stream = np.random.random((I, T+1))
eps_random = np.random.random((I, T+1))

#%% epsilon
eps = eps_random.flatten()
plt.hist(eps, bins=np.arange(0, 1, .01), density=True)
plt.savefig(result_folder + "\\epsilon1.pdf", bbox_inches="tight")
plt.show()

#%%
sns.distplot(eps, hist=True, kde=True, color="darkblue", hist_kws={"edgecolor":"black"})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel("p")
plt.xlabel("$\epsilon$")
plt.savefig(result_folder + "\\epsilon2.pdf", bbox_inches="tight")
plt.show()

#%% customers
def assign_customer(random_customer):
    customer = random_customer <= np.array([*np.cumsum(arrival_probabilities), 1.0])
    customer = min(np.array(range(0, len(customer)))[customer])
    return customer

customer_stream_pure_customer = np.array([[assign_customer(r) for r in row] for row in customer_stream ])

# customer "0" shall represent no customer arrived
customer_stream_pure_customer += 1
customer_stream_pure_customer[customer_stream_pure_customer==max(customer_segments)+2] = 0

#%%
customer_hist = np.zeros((I, len(customer_segments)+1))
for i in np.arange(len(customer_segments)+1):
    customer_hist[:, i] = np.sum(customer_stream_pure_customer==i, axis=1)/T

df = pd.DataFrame(customer_hist, columns=["No customer", *["Cust. " + str(c) for c in customer_segments+1]])
sns.boxplot(data=df)
plt.ylabel("p")
plt.savefig(result_folder + "\\customer.pdf", bbox_inches="tight")
plt.show()

#%%
customer_statistics = np.zeros((2, len(customer_segments)+1))
for i in np.arange(len(customer_segments)+1):
    count = np.sum(customer_stream_pure_customer==i, axis=1)
    customer_statistics[0, i] = np.mean(count/T)
    customer_statistics[1, i] = np.std(count/T)

df = pd.DataFrame(customer_statistics, index=["mean", "sd"], columns=["no customer", *["Cust. " + str(c) for c in customer_segments+1]])


#%% offersets visualization
pi1 = pi_result[-1]
pi2 = pi_result[-2]


def plot_pi(pi, name):
    df = pd.DataFrame(pi, columns=resources + 1)
    sns.heatmap(df, annot=True, cmap="Blues_r")
    plt.xlabel("Resource")
    plt.ylabel("Time")
    plt.savefig(result_folder + "\\" + name + ".pdf", bbox_inches="tight")
    plt.show()


#%% visualize the values of the optimized variables
plot_pi(pi1, "pi60")
plot_pi(pi2, "pi59")

plot_pi(np.abs(pi1-pi2), "piAbsDifference")
#%% visualize offersets (erstmal nur für exampleStefan)
dat = np.zeros((9*5*20, 9*5))
c1 = capacities[0]+1
c2 = capacities[1]+1
c3 = capacities[2]+1
c4 = capacities[3]+1

offer_dict = {tuple(b):a for a, b in enumerate(get_offer_sets_all(products))}

for h1 in np.arange(c1):
    for h2 in np.arange(c2):
        for h3 in np.arange(c3):
            for h4 in np.arange(c4):
                for t in times:
                    pis = pi[t+1]  # t+1 because time period and looking at E_API_lin_multi_leg.py line 170
                    offer_tuple = determine_offer_tuple(pis, 0, 1, revenues, A,
                                                    arrival_probabilities, preference_weights, preferences_no_purchase)
                    dat[h1*(c3*T) + h3*(T) + t, h2*(c4) + h4] = offer_dict[offer_tuple]

sns.heatmap(dat)
plt.show()

#%%
if not (pi1 - pi_to_use == 0).all():
    raise ValueError
dat1 = np.zeros((9*5*5*9, 20))
c1 = capacities[0]+1
c2 = capacities[1]+1
c3 = capacities[2]+1
c4 = capacities[3]+1

offer_dict = {tuple(b):a for a, b in enumerate(get_offer_sets_all(products))}

pis = 1.0*np.zeros_like(resources)
for h1 in np.arange(c1):
    for h2 in np.arange(c2):
        for h3 in np.arange(c3):
            for h4 in np.arange(c4):
                for t in times:
                    pis[[h1 == 0, h2 == 0, h3 == 0, h4 == 0]] = np.inf
                    pis[[h1>0, h2>0, h3>0, h4>0]] = pi1[t+1][[h1>0, h2>0, h3>0, h4>0]]  # t+1 because time period and looking at e_api_lin_multi_leg.py line 170
                    offer_tuple = determine_offer_tuple(pis, 0, 1, revenues, A,
                                                    arrival_probabilities, preference_weights, preferences_no_purchase)
                    dat1[h1*(c2*c3*c4) + h2*(c3*c4) + h3*(c4) + h4, t] = offer_dict[offer_tuple]
dat1 = dat1.astype(int)

#%%
dat2 = np.zeros((9*5*5*9, 20))
c1 = capacities[0]+1
c2 = capacities[1]+1
c3 = capacities[2]+1
c4 = capacities[3]+1

offer_dict = {tuple(b):a for a, b in enumerate(get_offer_sets_all(products))}

pis = 1.0*np.zeros_like(resources)
for h1 in np.arange(c1):
    for h2 in np.arange(c2):
        for h3 in np.arange(c3):
            for h4 in np.arange(c4):
                for t in times:
                    pis[[h1==0, h2==0, h3==0, h4==0]] = np.inf
                    pis[[h1>0, h2>0, h3>0, h4>0]] = pi2[t+1][[h1>0, h2>0, h3>0, h4>0]] # t+1 because time period and looking at e_api_lin_multi_leg.py line 170
                    offer_tuple = determine_offer_tuple(pis, 0, 1, revenues, A,
                                                    arrival_probabilities, preference_weights, preferences_no_purchase)
                    dat2[h1*(c2*c3*c4) + h2*(c3*c4) + h3*(c4) + h4, t] = offer_dict[offer_tuple]
dat2 = dat2.astype(int)
#%%
all_values = np.unique(np.append(np.unique(dat1), np.unique(dat2)))
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
plot_offersets_heat(all_values, n, dat1, "offersetsAll60")
plot_offersets_heat(all_values, n, dat2, "offersetsAll59")

