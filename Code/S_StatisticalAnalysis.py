"""
This script compares various policies
"""

from B_helper import *
import seaborn as sns
from scipy.stats import ttest_rel
from scipy.stats import binom

#%%
res_folder_ES = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-ES-evaluation-190925-1341"
res_folder_CDLP = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-CDLP-evaluation-190925-1329"
res_folder_API_last = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-APILinearMultiLeg-1-evaluation-190925-1342"
res_folder_API_plc = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-APIplcMultiLeg-1-evaluation-190926-1320"
res_folder_API_pl = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-APIplcMultiLeg-1-evaluation-190929-2035"
res_folder_linReg = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-linReg-evaluation-190929-2049"
res_folder_NN_hyper = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-NN3-evaluation-190928-1928"
res_folder_NN_given = r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\exampleStefan-True-NN3-evaluation-190929-2250"

res_folders = [res_folder_ES, res_folder_CDLP, res_folder_API_last, res_folder_API_plc, res_folder_API_pl, res_folder_linReg, res_folder_NN_hyper, res_folder_NN_given]
labels = ["ES", "CDLP", "API-lc", "API-plc", "API-pl", "linReg", "NN-hyper", "NN-given"]

# res_folders = [res_folder_ES, res_folder_CDLP, res_folder_API_last]
# labels = ["ES", "CDLP", "API"]

#%%
# ensure that all setting files are the same and we have the setting parameters at hand
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
for f in res_folders:
    b = pd.read_csv(f+"\\0_settings.csv", delimiter="\t", header=None)
    if not (a.equals(b)):
        raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("ALL-comparison")

capacities = var_capacities[0]
no_purchase_preference = var_no_purchase_preferences[0]

#%% general formulas
def get_all_values(dat_name):
    ret = {}
    for i in np.arange(len(labels)):
        with open(res_folders[i] + "\\" + dat_name + ".data", "rb") as filehandle:
            df = pickle.load(filehandle)
            df = df.drop(0, axis=1)
            df.columns = [labels[i]]
        ret[i] = df

    dat = ret[0]
    for i in np.arange(len(labels) - 1) + 1:
        dat[ret[i].columns[0]] = ret[i]

    return dat

def get_all_dict(dat_name, capacities, no_purchase_preference):
    ret = {}
    for i in np.arange(len(labels)):
        with open(res_folders[i] + "\\" + dat_name + ".data", "rb") as filehandle:
            df = pickle.load(filehandle)
            df = df['' + str(capacities) + '-' + str(no_purchase_preference)]
        ret[labels[i]] = df

    return ret




# %% Create graphs
#%% values
v = get_all_values("value_final")

#%%
sns.set(style="whitegrid")
v_plot = pd.melt(v, var_name="Policy", value_name="Value")
sns.boxplot(x=v_plot.Policy, y=v_plot.Value)
plt.savefig(newpath + "\\" + "values-Boxplot" +".pdf", bbox_inches="tight")
plt.show()

#%%
sns.set(style="whitegrid")
sns.violinplot(x=v_plot.Policy, y=v_plot.Value)
plt.savefig(newpath + "\\" + "values-Violinplot" +".pdf", bbox_inches="tight")
plt.show()


#%%
erg_latex = open(newpath+"\\values-Summary.txt", "w+")  # write and create (if not there)
print(np.round(v.describe(), 2).to_latex(), file=erg_latex)
erg_latex.close()
np.round(v.describe(), 2)

#%%
# rows = ["Mean", "Max", "Min", "sd"]
# rows_function = [np.mean, np.max, np.min, np.std]
# columns = [value_final.columns[1]]
# df_value_summary = pd.DataFrame(np.zeros((len(rows), len(columns))), index=rows, columns=columns)
#
# for col in columns:
#     for i in np.arange(len(rows)):
#         df_value_summary.loc[rows[i], col] = rows_function[i](value_final.loc[:, col])
#
#
# df_value_summary

#%% products
sns.set(style="whitegrid")
p = get_all_dict("products_all", capacities, no_purchase_preference)

label_product = ["No Purchase", *(products+1)]

p_all = []
for i in np.arange(len(labels)):
    a = p[labels[i]]
    dat = pd.DataFrame(labels[i], index=np.arange(len(a)), columns=["Policy"])
    for j in np.arange(len(products) + 1):
        dat[j] = np.sum(a == j, axis=1)
    dat = dat.melt(id_vars=["Policy"], var_name="Product", value_name="Purchases")
    p_all.append(dat)
p_plot = pd.concat(p_all)


order = [len(products), *products]

fig, ax = plt.subplots()
sns.barplot(x="Product", y="Purchases", hue="Policy", data=p_plot, order=order)
ax.set_xticklabels(label_product)
plt.savefig(newpath + "\\" + "products-Barplot" +".pdf", bbox_inches="tight")
plt.show()


#%% no-purchase
sns.set(style="whitegrid")
n = get_all_dict("why_no_purchase_all", capacities, no_purchase_preference)

label_bar = ["No customer", "Customer, no interest", "Customer, interest, no-purchase"]
n_to_plot = pd.DataFrame(np.zeros((3, len(labels))), index=label_bar, columns=labels)

for i in np.arange(len(labels)):
    a = n[labels[i]]
    for j in np.arange(len(label_bar))+1:
        n_to_plot.iloc[j-1, i] = np.mean(np.sum(a == j, axis=1))



# #%% products
# bottom = np.zeros(K)
# for i in products:
#     plt.bar(np.arange(K)+1, p[:, i], label="Product "+str(i+1), bottom=bottom)
#     bottom += p[:, i]
# plt.legend(bbox_to_anchor=(0, -.23, 1, .102), loc="lower left",
#           ncol=3, mode="expand")
# plt.xticks(np.append([1], np.arange(5, K + 1, 5)))
# plt.savefig(result_folder+"\\plotProducts2.pdf", bbox_inches="tight")
# plt.show()

#%% products
# # https://matplotlib.org/3.1.1/gallery/statistics/barchart_demo.html#sphx-glr-gallery-statistics-barchart-demo-py
# p = products_all['' + str(capacities) + '-' + str(preferences_no_purchase)]
#
# p_plot = np.zeros(len(products)+1)
# for i in np.arange(len(products)+1):
#     p_plot[i] = sum(sum(p==i))
#
# label = ["No Purchase", *["Prod. "+str(i+1) for i in products]]
# x = np.arange(len(products) + 1)
# fig, ax = plt.subplots()
# ax.bar(x, p_plot)
# ax.yaxis.grid(True, linestyle="--", color="grey", alpha=.25)
# ax.set_xticks(x)
# ax.set_xticklabels(label)
# plt.show()

#%% offersets
o = get_all_dict("offersets_all", capacities, no_purchase_preference)
threshold = 4000

o_all = []
for i in np.arange(len(labels)):
    a = o[labels[i]]
    dat = pd.DataFrame(labels[i], index=np.arange(len(a)), columns=["Policy"])
    for j in np.arange(len(get_offer_sets_all(products)) + 1):
        dat[j] = np.sum(a == j, axis=1)
    o_all.append(dat)
o_plot = pd.concat(o_all)

o_plot = o_plot.set_index("Policy")
offer_sets_relevant = o_plot.columns[np.sum(o_plot, axis=0) > threshold]
o_plot = o_plot.loc[:, offer_sets_relevant]
o_plot["Policy"] = o_plot.index

#%%
sns.set(style="whitegrid")


o_plot = o_plot.melt(id_vars="Policy", var_name="Offerset", value_name="Number of times offered")

# fig, ax = plt.subplots()
sns.barplot(x="Offerset", y="Number of times offered", hue="Policy", data=o_plot)
# ax.set_xticklabels(label_product)
plt.savefig(newpath + "\\" + "offersets-Barplot-"+str(threshold)+".pdf", bbox_inches="tight")
plt.show()

#%%
os_dict_reverse = {k:tuple(v) for k,v in enumerate(get_offer_sets_all(products))}

o_table = pd.DataFrame(offer_sets_relevant, columns=["Offerset Index"])
o_table["Offerset Tuple"] = ""
o_table["Offerset Items"] = ""
for i in o_table.index:
    tup = os_dict_reverse[o_table.loc[i, "Offerset Index"]]
    o_table.loc[i, "Offerset Tuple"] = tup
    o_table.loc[i, "Offerset Items"] = re.sub("\]", "}", re.sub("\[", "{", re.sub(" ", ",", str(products[np.array(tup)==1]+1))))


erg_latex = open(newpath+"\\offersets-Overview-"+str(threshold)+".txt", "w+")  # write and create (if not there)
print(o_table.to_latex(index=False), file=erg_latex)
erg_latex.close()
o_table

#%% remaining capacity left over
sns.set(style="whitegrid")
r = get_all_dict("capacities_final", capacities, no_purchase_preference)

r_all = []
for i in np.arange(len(labels)):
    a = r[labels[i]]
    dat = pd.DataFrame(labels[i], index=np.arange(len(a)), columns=["Policy"])
    for i in a.columns:
        dat[i] = a[i]
    r_all.append(dat)
r_plot = pd.concat(r_all)

r_plot = r_plot.melt(id_vars="Policy", var_name="Resource", value_name="Capacity left over")

#%%
label_resources = resources+1


fig, ax = plt.subplots()
sns.barplot(x="Resource", y="Capacity left over", hue="Policy", data=r_plot)
ax.set_xticklabels(label_resources)
plt.savefig(newpath + "\\" + "resources-Barplot" +".pdf", bbox_inches="tight")
plt.show()

#%%
sns.set(palette="Reds", style="whitegrid")
ax = n_to_plot.T.plot(kind="bar", stacked=True)

n_helper = np.cumsum(n_to_plot, axis=0)

x_coord = np.unique(np.array([a.get_x() for a in ax.patches]))
for i in np.arange(len(labels)):
    for j in np.arange(len(label_bar))+1:
        ax.text(x_coord[i]+.15, n_helper.iloc[j-1, i]-.6, str(np.round(n_to_plot.iloc[j-1, i], 2)), fontsize=15, color='dimgrey')

plt.savefig(newpath + "\\" + "no-purchase-stackedBarplot" +".pdf", bbox_inches="tight")
plt.show()

#%% statistical analysis of values

def test_get_df(v, func_test, samples_used, labels):
    df_stats = pd.DataFrame(np.zeros((v.shape[1], v.shape[1])), index=labels, columns=labels)
    for i in np.arange(v.shape[1]):
        for j in np.arange(v.shape[1]):
            if i == j:
                df_stats.iloc[i, i] = np.nan
                continue

            data_i = v.iloc[:samples_used, i]
            data_j = v.iloc[:samples_used, j]

            df_stats.iloc[i, j] = func_test(data_i, data_j)

    return df_stats

#%% paired t-test
samples_used = 100

df_stats = pd.DataFrame(np.zeros((v.shape[1], v.shape[1])), index=labels, columns=labels)

for i in np.arange(v.shape[1]):
    for j in np.arange(v.shape[1]):
        if i==j:
            df_stats.iloc[i, i] = ""
        df_stats.iloc[i, j] = ttest_rel(v.iloc[:samples_used,i], v.iloc[:samples_used,j]).pvalue

erg_latex = open(newpath+"\\statistics-tTest-"+str(samples_used)+".txt", "w+")  # write and create (if not there)
print(df_stats.to_latex(na_rep=""), file=erg_latex)
erg_latex.close()

df_stats

#%% paired t-test
samples_used = 5000

df_stats = pd.DataFrame(np.zeros((v.shape[1], v.shape[1])), index=labels, columns=labels)

for i in np.arange(v.shape[1]):
    for j in np.arange(v.shape[1]):
        if i==j:
            df_stats.iloc[i, i] = ""
        df_stats.iloc[i, j] = ttest_rel(v.iloc[:samples_used,i], v.iloc[:samples_used,j]).pvalue

erg_latex = open(newpath+"\\statistics-tTest-"+str(samples_used)+".txt", "w+")  # write and create (if not there)
print(df_stats.to_latex(na_rep=""), file=erg_latex)
erg_latex.close()

df_stats

#%% permutation test
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = abs(np.mean(data_1) - np.mean(data_2))

    return diff



#%% permutation test
samples_used = 100
perm_size = 10000

df_stats = pd.DataFrame(np.zeros((v.shape[1], v.shape[1])), index=labels, columns=labels)

for i in np.arange(v.shape[1]):
    for j in np.arange(v.shape[1]):
        if i==j:
            df_stats.iloc[i, i] = ""
            continue

        data_i = v.iloc[:samples_used, i]
        data_j = v.iloc[:samples_used, j]

        # Compute difference of mean
        empirical_diff_means = diff_of_means(data_i, data_j)

        # Draw 10,000 permutation replicates: perm_replicates
        perm_replicates = draw_perm_reps(data_i, data_j,
                                         diff_of_means, size=perm_size)

        # Compute p-value: p
        p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

        df_stats.iloc[i, j] = p



erg_latex = open(newpath+"\\statistics-permTest-"+str(samples_used)+".txt", "w+")  # write and create (if not there)
print(df_stats.to_latex(na_rep=""), file=erg_latex)
erg_latex.close()

df_stats

# %% permutation test
samples_used = 5000
perm_size = 10000

df_stats = pd.DataFrame(np.zeros((v.shape[1], v.shape[1])), index=labels, columns=labels)

for i in np.arange(v.shape[1]):
    for j in np.arange(v.shape[1]):
        if i == j:
            df_stats.iloc[i, i] = ""
            continue

        data_i = v.iloc[:samples_used, i]
        data_j = v.iloc[:samples_used, j]

        # Compute difference of mean
        empirical_diff_means = diff_of_means(data_i, data_j)

        # Draw 10,000 permutation replicates: perm_replicates
        perm_replicates = draw_perm_reps(data_i, data_j,
                                         diff_of_means, size=perm_size)

        # Compute p-value: p
        p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

        df_stats.iloc[i, j] = p

erg_latex = open(newpath + "\\statistics-permTest-" + str(samples_used) + ".txt",
                 "w+")  # write and create (if not there)
print(df_stats.to_latex(na_rep=""), file=erg_latex)
erg_latex.close()

df_stats


#%% rank test (much nicer implemented)
def rank_test(data_i, data_j):
    n = sum(data_i != data_j)
    # c = sum of all pairs data_i > data_j
    c = sum(data_i > data_j)
    p = 1 - binom.cdf(n=n, k=c, p=.5)

    return p


#%%
samples_used=100
df_stats = test_get_df(v, rank_test, samples_used=samples_used, labels=labels)
erg_latex = open(newpath + "\\statistics-rankTest-" + str(samples_used) + ".txt",
                 "w+")  # write and create (if not there)
print(np.round(df_stats, 6).to_latex(na_rep=""), file=erg_latex)
erg_latex.close()

np.round(df_stats, 6)

#%%
samples_used=5000
df_stats = test_get_df(v, rank_test, samples_used=samples_used, labels=labels)
erg_latex = open(newpath + "\\statistics-rankTest-" + str(samples_used) + ".txt",
                 "w+")  # write and create (if not there)
print(np.round(df_stats, 6).to_latex(na_rep=""), file=erg_latex)
erg_latex.close()

np.round(df_stats, 6)

#%%
print("\n\nData used:\n", file=logfile)
for i_ind, i_name in enumerate(res_folders):
    print(labels[i_ind], i_name, file=logfile)
wrapup(logfile, time_start, newpath)


#%%

df = pd.DataFrame(np.zeros((v.shape[1], v.shape[1])), index=labels, columns=labels, dtype=int)

for i in np.arange(v.shape[1]):
    for j in np.arange(v.shape[1]):
        df.iloc[i, j] = sum(v.iloc[:, i] != v.iloc[:, j])

erg_latex = open(newpath + "\\num-different-values.txt", "w+")  # write and create (if not there)
print(df.to_latex(na_rep=""), file=erg_latex)
erg_latex.close()

df

#%% CDLP num corrections
with open(res_folders[1] + "\\" + "num_corrections_all" + ".data", "rb") as filehandle:
    df = pickle.load(filehandle)

#%%
num_corrections = df['' + str(capacities) + '-' + str(no_purchase_preference)]

number_corrections_distinct = np.unique(num_corrections[:, -1])
label_columns = [int(i) for i in number_corrections_distinct]
df_corrections = pd.DataFrame(1.0*np.zeros((online_K, len(number_corrections_distinct))), columns=label_columns)
for i in number_corrections_distinct:
    print(i, ": ", np.sum(num_corrections[:, -1] == i))
    first_occurence = 1.0*np.array([min((np.arange(T)+1)[d == i], default=-1) for d in num_corrections])
    first_occurence[first_occurence==-1] = np.nan
    df_corrections[int(i)] = first_occurence

erg_latex = open(newpath + "\\CDLP-numberCorrections" + ".txt", "w+")  # write and create (if not there)
print(np.round(df_corrections.describe(), 2).to_latex(), file=erg_latex)
erg_latex.close()
np.round(df_corrections.describe(), 2)
