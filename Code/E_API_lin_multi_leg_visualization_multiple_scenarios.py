"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *
from os import listdir
from os.path import isfile, join

#%%
result_folder = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\singleLegFlight-True-APILinearMultiLeg-190919-0859"

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
folders = [f for f in listdir(result_folder) if not(isfile(join(result_folder, f)))]

value_overview = pd.DataFrame()
for f in folders:
    with open(join(result_folder, f, "value_result.data"), "rb") as filehandle:
        tmp = pickle.load(filehandle)
    value_overview[f] = tmp[1:, :, 0].mean(axis=1)

erg_latex = open(result_folder+"\\value_overview.txt", "w+")  # write and create (if not there)
print(value_overview.to_latex(), file=erg_latex)
erg_latex.close()
