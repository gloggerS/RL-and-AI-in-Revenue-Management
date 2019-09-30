"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *
import seaborn as sns
import matplotlib.patches as mpatches

#%%
result_folder = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-NN1multiLeg-190928-0004"

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

