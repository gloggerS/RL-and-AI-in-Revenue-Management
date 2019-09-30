"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *
import seaborn as sns
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

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
    = setup_testing("value-func-Visualization")

capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]


#%%
# offersets_offered.data
# sold_out.data
# time_measured.data
result_folder = result_folder + "\\cap" + str(capacities) + " - pref" + str(preferences_no_purchase)

with open(result_folder+"\\value_result.data", "rb") as filehandle:
    value_result = pickle.load(filehandle)

with open(result_folder+"\\capacities_result.data", "rb") as filehandle:
    capacities_result = pickle.load(filehandle)

#%%
k = 1
v = value_result[k, :, :]
v = v[:, 1:]

c = capacities_result[k, :, :, :]
c = c[:, 1:, :]

#%%
def save_files(storagepath, theta_all, pi_all, *args):
    makedirs(storagepath, exist_ok=True)

    with open(storagepath + "\\thetaToUse.data", "wb") as filehandle:
        pickle.dump(theta_all[-1], filehandle)

    with open(storagepath + "\\piToUse.data", "wb") as filehandle:
        pickle.dump(pi_all[-1], filehandle)

    for o in [theta_all, pi_all, *args]:
        o_name = re.sub("[\[\]']", "", str(varnameis(o)))
        # print(o_name)
        with open(storagepath + "\\" + o_name + ".data", "wb") as filehandle:
            pickle.dump(o, filehandle)

def varnameis(v): d = globals(); return [k for k in d if d[k] is v]

# %%

#%%

#%%
def value_mean(X, y, c, t):
    indices = np.logical_and(X[:, 0] == c, X[:, 1] == t)

    if sum(indices) == 0:
        return -1
    else:
        return np.nanmean(y[indices])


def plot_bar(tab_means, X, y, h):
    c_max = np.max(X[:, 0])
    t_max = np.max(X[:, 1])

    # surface plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    y_axis = np.arange(tab_means.shape[0])
    x_axis = np.arange(tab_means.shape[1])
    X_axis, Y_axis = np.meshgrid(x_axis, y_axis)

    Z_axis = np.zeros_like(X_axis)
    for c in y_axis:
        for t in x_axis:
            Z_axis[c, t] = value_mean(X, y, c, t)

    indices = Z_axis > -1

    z_pos = [0] * sum(sum(indices))
    x_size = [1] * sum(sum(indices))
    y_size = [1] * sum(sum(indices))

    ax.bar3d(X_axis[indices], Y_axis[indices], z_pos, x_size, y_size, Z_axis[indices])

    ax.set_title('Mean of Value Function')
    ax.set_xlabel('Time')
    ax.set_ylabel('Capacity of resource '+str(h))
    ax.set_xlim(-1, t_max+2)
    ax.set_ylim(-1, c_max+2)
    plt.yticks(np.arange(0, c_max + 1, 1.0)+.5, np.arange(0, c_max + 1, 1.0, dtype=int))
    plt.xticks(np.arange(0, t_max + 1, 2)+.5, np.arange(0, t_max + 1, 2, dtype=int)+1)

    fig.savefig(newpath+"\\plotMeanValueBar-res"+str(h)+".pdf", bbox_inches="tight")

    plt.show()


def plot_surface(tab_means, X, y):
    # surface plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    y_axis = np.arange(tab_means.shape[0])
    x_axis = np.arange(tab_means.shape[1])
    X_axis, Y_axis = np.meshgrid(x_axis, y_axis)

    Z_axis = np.zeros_like(X_axis)
    for c in y_axis:
        for t in x_axis:
            Z_axis[c, t] = value_mean(X, y, c, t)

    ax.plot_surface(X_axis, Y_axis, Z_axis, cmap='viridis')

    ax.set_title('Mean of Value Function')
    ax.set_xlabel('Time')
    ax.set_ylabel('Capacity')
    plt.yticks(np.arange(0, c_max + 1, 1.0))

    plt.show()


def value_analysis(c_samples, X, y):
    c_max = int(np.max(c_samples))

    tab_means = np.zeros((c_max + 1, T))
    tab_numbers = np.zeros((c_max + 1, T))

    for c in np.arange(tab_means.shape[0]):
        for t in np.arange(tab_means.shape[1]):
            indices = np.logical_and(X[:, 0] == c, X[:, 1] == t)
            tab_means[c_max - c, t] = value_mean(X, y, c, t)
            tab_numbers[c_max - c, t] = sum(indices)

    df_means = pd.DataFrame(tab_means,
                            columns=["t" + str(t) for t in np.arange(tab_means.shape[1])],
                            index=["c" + str(c) for c in (c_max - np.arange(tab_means.shape[0]))])
    df_numbers = pd.DataFrame(tab_numbers,
                              columns=["t" + str(t) for t in np.arange(tab_numbers.shape[1])],
                              index=["c" + str(c) for c in (c_max - np.arange(tab_numbers.shape[0]))])

    return df_means, df_numbers, tab_means




#%%
for h in resources:
    v_tmp = pd.melt(pd.DataFrame(v))
    c_tmp = pd.melt(pd.DataFrame(c[:, :, h]))

    # check if indices equal
    sum([not a == b for a, b in zip(c_tmp.iloc[:, 0], v_tmp.iloc[:, 0])])

    X = np.zeros((len(v_tmp), 2))
    X[:, 0] = c_tmp.iloc[:, 1]
    X[:, 1] = c_tmp.iloc[:, 0]

    y = v_tmp.iloc[:, 1]

    df_means, df_numbers, tab_means = value_analysis(c[:, :, h], X, y)
    plot_bar(tab_means, X, y, h + 1)


# %%
wrapup(logfile, time_start, newpath)

#%%
print("J_visualize.py completed.")

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
# my_shelf.close()