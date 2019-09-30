from B_helper import *

import seaborn as sns
import matplotlib.patches as mpatches


#%%
result_folder = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-NN3multiLeg-r2-190928-2013"

a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
b = pd.read_csv(result_folder+"\\0_settings.csv", delimiter="\t", header=None)
if not(a.equals(b)):
    raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("quick-analysis")

capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]
no_purchase_preference = preferences_no_purchase

result_folder = result_folder + "\\cap" + str(capacities) + " - pref" + str(preferences_no_purchase)
#%%
with open(result_folder+"\\n_iter.data", "rb") as filehandle:
    d = pickle.load(filehandle)

#%%
plt.plot(np.arange(K)+1, d[:-1])
plt.xticks(np.append([1], np.arange(5, K+1, 5)))
plt.xlabel("Policy Iteration k")
plt.ylabel("Number of iterations")
plt.savefig(result_folder+"\\numIterations.pdf", bbox_inches="tight")
plt.show()


#%%
with open(result_folder+"\\offersets_offered.data", "rb") as filehandle:
    d = pickle.load(filehandle)

with open(result_folder+"\\offersets_offered.data", "rb") as filehandle:
    offer_sets = pickle.load(filehandle)

label_relevant = [str((products+1)[a==1]) for a in get_offer_sets_all(products)[offer_sets.sum(axis=0)>0]]

offer_sets_relevant = offer_sets[1:, offer_sets.sum(axis=0)>0]

df = pd.DataFrame(offer_sets_relevant, columns=[re.sub(" ", ",", str(l)) for l in label_relevant], index=(np.arange(K)+1), dtype=int)


#%%
erg_latex = open(result_folder+"\\offersetsOffered.txt", "w+")  # write and create (if not there)
print(df.to_latex(), file=erg_latex)
erg_latex.close()
#%%
ind = [0, 12, 20, -2]
for i in np.arange(len(ind)-1):
    t = df.iloc[:, ind[i]:ind[i+1]]
    erg_latex = open(result_folder + "\\offersetsOffered"+str(i)+".txt", "w+")  # write and create (if not there)
    print(t.to_latex(), file=erg_latex)
    erg_latex.close()