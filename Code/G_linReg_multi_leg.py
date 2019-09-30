"""
This script will calculate the MLP regressor with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
https://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/

IDEE: Alles analog zu API single leg, aber jetzt theta und pi mit MLP Regressor lernen
"""


from B_helper import *


from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm

#%%
# Setup of parameters
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("linRegMultiLeg")
capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]


#%%
def update_parameters(v_samples, c_samples, thetas, pis, k, exponential_smoothing, silent=True):
    """
    Updates the parameter theta, pi given the sample path using least squares linear regression with constraints.
    Using statsmodels

    :param v_samples:
    :param c_samples:
    :return:
    """

    # Prepare the data into statistical setting of
    # X: training data (capacity, time) in row
    # y: true outcomes (value) in row

    v = v_samples
    c = c_samples

    v_df = pd.DataFrame(v)  # nothing happening in time period 0 (first action in t=1)
    v_df.columns = ["vt"+str(i) for i in v_df.columns]
    v_df = v_df.stack()

    t_df = pd.DataFrame([np.arange(T+1)] * len(v))
    t_df.columns = ["t"+str(i) for i in t_df.columns]
    t_df = tidy_up_t(t_df)

    # to allow for multiple resources
    cs = {}
    for h in resources:
        c_df = pd.DataFrame(c[0, :, h]).T
        for i in np.arange(len(c) - 1) + 1:
            c_df = c_df.append(pd.DataFrame(c[i, :, h]).T)
        c_df.columns = ["c-h"+str(h)+"-t"+str(i) for i in c_df.columns]
        c_df = tidy_up_c(c_df)
        c_df.index = t_df.index
        cs[h] = c_df

    X = pd.concat([t_df, *[cs[h] for h in resources]], axis=1)
    y = v_df
    y.index = X.index


    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    model.params


    predictions = model.predict(X)  # make the predictions by the model

    # Print out the statistics
    if not silent:
        model.summary()

    theta_new = np.array([[i] for i in model.params[0:(T+1)]])
    pi_new = np.zeros_like(pis)
    for h in resources:
        pi_new[:, h] = np.array([i for i in model.params[(h+1)*(T+1):(h+2)*(T+1)]])

    # check exponential smoothing
    if exponential_smoothing:
        return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new, model.ess
    else:
        return theta_new, pi_new, model.ess


def tidy_up_c(t_df):
    col = t_df.columns
    ind = t_df.index
    n_obs = t_df.shape[0]
    n_time = t_df.shape[1]
    t_numpy = np.zeros((n_obs * n_time, n_time))
    for i in np.arange(n_time):
        t_numpy[i * n_obs:(i + 1) * n_obs, i] = t_df[col[i]]
    t_df = pd.DataFrame(t_numpy)
    t_df.columns = col
    t_df.index = np.tile(ind, n_time)
    return t_df

def tidy_up_t(t_df):
    col = t_df.columns
    ind = t_df.index
    n_obs = t_df.shape[0]
    n_time = t_df.shape[1]
    t_numpy = np.zeros((n_obs * n_time, n_time))
    for i in np.arange(n_time):
        t_numpy[i * n_obs:(i + 1) * n_obs, i] = 1
    t_df = pd.DataFrame(t_numpy)
    t_df.columns = col
    t_df.index = np.tile(ind, n_time)
    return t_df

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
# Actual Code

# generate the random sample paths (T+1 => have real life indexing starting at t=1)
np.random.seed(12)
seed(12)
customer_stream = np.random.random((I, T+1))
sales_stream = np.random.random((I, T+1))
eps_random = np.random.random((I, T+1))

# summary statistics
offer_sets_all = {str(tuple(obj)): count for count, obj in enumerate(get_offer_sets_all(products))}

#%%

for capacities in var_capacities:
   for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ", preferences_no_purchase, "of",
              str(var_no_purchase_preferences.tolist()), "starting.")

        # summary statistics
        offersets_offered = np.zeros((K + 1, len(offer_sets_all)))
        products_sold = np.zeros((K + 1, len(products) + 1))
        sold_out = np.zeros((K + 1, I))
        obj_vals = np.zeros(K + 1)
        time_measured = np.zeros(K + 1)

        value_result = np.zeros((K+1, I, T+1))
        capacities_result = np.zeros((K+1, I, T+1, len(resources)))

        # store parameters over all policy iterations
        # K+1 policy iterations (starting with 0)
        # T time steps
        theta_all = np.zeros((K+1, T+1, 1))
        pi_all = np.zeros((K+1, T+1, len(resources)))

        # theta and pi for each time step
        # line 1
        thetas = 0
        pis = np.zeros(len(resources))

        for k in np.arange(K) + 1:
            print(k, "of", K, "starting.")
            t_in_k_start = time()

            v_samples = np.zeros((I, T + 1))
            c_samples = np.zeros((I, T + 1, len(capacities)))

            for i in np.arange(I):
                # line 3
                r_sample_i = np.zeros(T + 1)  # will become v_sample
                c_sample_i = np.zeros(shape=(T + 1, len(capacities)), dtype=int)

                # line 5
                c = deepcopy(capacities)  # (starting capacity at time 0)

                for t in times + 1:
                    # line 7  (starting capacity at time t)
                    c_sample_i[t - 1] = c

                    # line 8-11  (adjust bid price)
                    pis[c == 0] = np.inf
                    pis[c > 0] = pi_all[k - 1, t, c > 0]

                    # line 12  (epsilon greedy strategy)
                    offer_set = determine_offer_tuple(pis, epsilon[k], eps_random[i, t], revenues, A,
                                                      arrival_probabilities, preference_weights,
                                                      preferences_no_purchase)
                    offersets_offered[k, offer_sets_all[str(offer_set)]] += 1

                    # line 13  (simulate sales)
                    sold = simulate_sales(offer_set, customer_stream[i, t], sales_stream[i, t],
                                          arrival_probabilities, preference_weights, preferences_no_purchase)
                    products_sold[k, sold] += 1

                    # line 14
                    try:
                        r_sample_i[t] = revenues[sold]
                        c -= A[:, sold]
                    except IndexError:
                        # no product was sold
                        pass

                    if all(c == 0):
                        sold_out[k, i] = t
                        break

                # also adjust last sell
                c_sample_i[t] = c

                # line 16-18
                v_samples[i] = np.cumsum(r_sample_i[::-1])[::-1]
                c_samples[i] = c_sample_i

            # line 20
            theta_all[k], pi_all[k], obj_vals[k] = update_parameters(v_samples, c_samples, theta_all[k - 1],
                                                                     pi_all[k - 1],
                                                                     k, exponential_smoothing)

            value_result[k, :, :] = v_samples
            capacities_result[k, :, :] = c_samples

            time_measured[k] = time() - t_in_k_start

        storagepath = newpath + "\\cap" + str(capacities) + " - pref" + str(preferences_no_purchase)

        save_files(storagepath, theta_all, pi_all, value_result, capacities_result, offersets_offered,
                   products_sold, sold_out, obj_vals, time_measured)


#%%
look_at_value = np.zeros([I, K])
for k in np.arange(K)+1:
    tmp = np.array(value_result[k])
    look_at_value[:,k-1] = tmp[:,0]
np.mean(look_at_value, 0)

# %%
wrapup(logfile, time_start, newpath)

print("G_linReg_single_leg.py completed.")
