"""
This script will calculate the MLP regressor with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
https://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/

IDEE: Alles analog zu API single leg, aber jetzt theta und pi mit MLP Regressor lernen
"""

#%%
from B_helper import *

# Setup of parameters
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("NNmultiLeg-hyperparam")
capacities = var_capacities[0]
no_purchase_preference = var_no_purchase_preferences[0]

silent = True

#%% domain space
space = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [0.001, 0.01, 0.1],
    'hidden_layer_sizes': [(10,), (50,10,), (100, 10, 10,)]
}

import ast

def parse_tuple(string):
    try:
        s = ast.literal_eval(str(string))
        if type(s) == tuple:
            return s
        return
    except:
        return

#%%

from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm

from sklearn.metrics import r2_score
import sklearn.metrics

# to apply or on array and for memoization (determine offer-tuple)
import functools
import operator

# Modeling
import lightgbm as lgb
from hyperopt import STATUS_OK, hp
from hyperopt.pyll.stochastic import sample

N_FOLDS = 10

# Evaluation of the model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import random as ra

# Algorithm
from hyperopt import tpe
tpe_algorithm = tpe.suggest

# History
from hyperopt import Trials# Trials object to track progress
bayes_trials = Trials()

import csv

# File to save first results
out_file = 'gbm_trials.csv'
of_connection = open(newpath+"\\"+out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Optimization
from hyperopt import fmin
MAX_EVALS = 5000

#%%

#%%
# times = np.arange(100)
# T = len(times)
# K = 20
# I = 50


#%%
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

# %% HELPER-FUNCTIONS
def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer

def update_parameters_NN(v_samples, c_samples, silent=True):
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
    neural_net = MLPRegressor(alpha=0.1, hidden_layer_sizes=(10,), max_iter=50000,
                              activation='logistic', verbose=not silent, learning_rate='adaptive')
    m = neural_net.fit(X, y)

    y_true = y
    y_pred = m.predict(X)
    r2_score(y_true, y_pred)

    # Still the OLS for comparison
    model = sm.OLS(y, X).fit()
    model.params

    y_pred_ols = model.predict(X)  # make the predictions by the model
    r2_score(y_true, y_pred_ols)

    # # Print out the statistics
    # model.summary()
    #
    #
    # # Print out the statistics
    # if not silent:
    #     model.summary()

    return m




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

#%%
def save_files(storagepath, *args):
    makedirs(storagepath, exist_ok=True)

    for o in [*args]:
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
   for no_purchase_preference in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ", no_purchase_preference, "of",
              str(var_no_purchase_preferences.tolist()), "starting.")

        # summary statistics
        offersets_offered = np.zeros((K + 1, len(offer_sets_all)))
        products_sold = np.zeros((K + 1, len(products) + 1))
        sold_out = np.zeros((K + 1, I))
        best_loss = np.zeros(K + 1)
        n_iter = np.zeros(K + 1)
        time_measured = np.zeros(K + 1)

        value_result = np.zeros((K+1, I, T+1))
        capacities_result = np.zeros((K+1, I, T+1, len(resources)))


        k = 1
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

               # line 12  (epsilon greedy strategy)
                if k == 1:
                    offer_set = tuple(np.ones_like(products))
                else:
                    offer_set = determine_offer_tuple_NN(nn, t-1, c, epsilon[k], eps_random[i, t], revenues, A, arrival_probabilities, preference_weights, no_purchase_preference)
                offersets_offered[k, offer_sets_all[str(offer_set)]] += 1

                # line 13  (simulate sales)
                sold = simulate_sales(offer_set, customer_stream[i, t], sales_stream[i, t],
                                      arrival_probabilities, preference_weights, no_purchase_preference)
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

#%% hyperparameter tuning
        v = v_samples
        c = c_samples

        v_df = pd.DataFrame(v)  # nothing happening in time period 0 (first action in t=1)
        v_df.columns = ["vt" + str(i) for i in v_df.columns]
        v_df = v_df.stack()

        t_df = pd.DataFrame([np.arange(T + 1)] * len(v))
        t_df.columns = ["t" + str(i) for i in t_df.columns]
        t_df = tidy_up_t(t_df)

        # to allow for multiple resources
        cs = {}
        for h in resources:
            c_df = pd.DataFrame(c[0, :, h]).T
            for i in np.arange(len(c) - 1) + 1:
                c_df = c_df.append(pd.DataFrame(c[i, :, h]).T)
            c_df.columns = ["c-h" + str(h) + "-t" + str(i) for i in c_df.columns]
            c_df = tidy_up_c(c_df)
            c_df.index = t_df.index
            cs[h] = c_df

        X = pd.concat([t_df, *[cs[h] for h in resources]], axis=1)
        y = v_df
        y.index = X.index

        y_true = y

        y_pred_all = {}

        df = pd.DataFrame(
            np.zeros((len(space['activation']) * len(space['alpha']) * len(space['hidden_layer_sizes']), 5)),
            columns=['activation', 'alpha', 'hidden_layer_sizes', 'r2', 'mae'])

        for act_i, act_name in enumerate(space['activation']):
            for alp_i, alp_name in enumerate(space['alpha']):
                for hl_i, hl_name in enumerate(space['hidden_layer_sizes']):
                    # Note the difference in argument order
                    print(act_name, " - ", alp_name, " - ", hl_name)
                    neural_net = MLPRegressor(alpha=alp_name, hidden_layer_sizes=hl_name, max_iter=50000,
                                              activation=act_name, verbose=not silent, learning_rate='adaptive', random_state=149)
                    m = neural_net.fit(X, y)
                    y_pred = m.predict(X)
                    df.iloc[act_i * (len(space['alpha']) * len(space['hidden_layer_sizes'])) + alp_i * len(
                        space['hidden_layer_sizes']) + hl_i, :] = [act_name, alp_name, str(hl_name),
                                                                   r2_score(y_true, y_pred),
                                                                   mean_absolute_error(y_true, y_pred)]
                    y_pred_all[act_i * (len(space['alpha']) * len(space['hidden_layer_sizes'])) + alp_i * len(
                        space['hidden_layer_sizes']) + hl_i] = y_pred

        opt_r2_index = np.argmax(df.iloc[:, 3].values)
        params_r2 = {
            'activation': df.iloc[opt_r2_index, 0],
            'alpha': df.iloc[opt_r2_index, 1],
            'hidden_layer_sizes': parse_tuple(df.iloc[opt_r2_index, 2])
        }
        n_r2 = MLPRegressor(alpha=df.iloc[opt_r2_index, 1], hidden_layer_sizes=parse_tuple(df.iloc[opt_r2_index, 2]), max_iter=50000,
                                              activation=df.iloc[opt_r2_index, 0], verbose=not silent, learning_rate='adaptive')
        
        opt_mae_index = np.argmin(df.iloc[:, 4].values)
        params_mae = {
            'activation': df.iloc[opt_r2_index, 0],
            'alpha': df.iloc[opt_r2_index, 1],
            'hidden_layer_sizes': parse_tuple(df.iloc[opt_r2_index, 2])
        }
        n_mae = MLPRegressor(alpha=df.iloc[opt_mae_index, 1], hidden_layer_sizes=parse_tuple(df.iloc[opt_mae_index, 2]), max_iter=50000,
                                              activation=df.iloc[opt_mae_index, 0], verbose=not silent, learning_rate='adaptive')
        
        np.argmax([sklearn.metrics.r2_score(y_true, y_pred_all[y]) for y in y_pred_all])

        opt_explvar_index = np.argmax([sklearn.metrics.explained_variance_score(y_true, y_pred_all[y]) for y in y_pred_all])
        params_explvar = {
            'activation': df.iloc[opt_r2_index, 0],
            'alpha': df.iloc[opt_r2_index, 1],
            'hidden_layer_sizes': parse_tuple(df.iloc[opt_r2_index, 2])
        }
        n_explvar = MLPRegressor(alpha=df.iloc[opt_explvar_index, 1], hidden_layer_sizes=parse_tuple(df.iloc[opt_explvar_index, 2]),
                             max_iter=50000,
                             activation=df.iloc[opt_explvar_index, 0], verbose=not silent, learning_rate='adaptive')
        
        opt_meansquared_index = np.argmax([sklearn.metrics.mean_squared_error(y_true, y_pred_all[y]) for y in y_pred_all])
        params_meansquared = {
            'activation': df.iloc[opt_r2_index, 0],
            'alpha': df.iloc[opt_r2_index, 1],
            'hidden_layer_sizes': parse_tuple(df.iloc[opt_r2_index, 2])
        }
        n_meansquared = MLPRegressor(alpha=df.iloc[opt_meansquared_index, 1], hidden_layer_sizes=parse_tuple(df.iloc[opt_meansquared_index, 2]),
                             max_iter=50000,
                             activation=df.iloc[opt_meansquared_index, 0], verbose=not silent, learning_rate='adaptive')

        opt_meansquaredlog_index = np.argmax([sklearn.metrics.mean_squared_log_error(y_true, y_pred_all[y]) for y in y_pred_all])
        params_meansquaredlog = {
            'activation': df.iloc[opt_meansquaredlog_index, 0],
            'alpha': df.iloc[opt_meansquaredlog_index, 1],
            'hidden_layer_sizes': parse_tuple(df.iloc[opt_meansquaredlog_index, 2])
        }
        n_meansquaredlog = MLPRegressor(alpha=df.iloc[opt_meansquaredlog_index, 1], hidden_layer_sizes=parse_tuple(df.iloc[opt_meansquaredlog_index, 2]),
                             max_iter=50000,
                             activation=df.iloc[opt_meansquaredlog_index, 0], verbose=not silent, learning_rate='adaptive')

        storagepath = newpath + "\\cap" + str(capacities) + " - pref" + str(no_purchase_preference)

        save_files(storagepath, df, n_r2, n_mae, n_meansquared, n_meansquaredlog, params_r2, params_explvar, params_mae, params_meansquared, params_meansquaredlog)

        df_file = pd.DataFrame(
            np.zeros((len(space['activation']) * len(space['alpha']) * len(space['hidden_layer_sizes']), 8)),
            columns=['activation', 'alpha', 'hidden_layer_sizes', 'r2', 'mae', 'explvar', 'msq', 'msqlog'])

        df_file.iloc[:, :5] = df
        df_file.iloc[:, 5] = [sklearn.metrics.explained_variance_score(y_true, y_pred_all[y]) for y in y_pred_all]
        df_file.iloc[:, 6] = [sklearn.metrics.mean_squared_error(y_true, y_pred_all[y]) for y in y_pred_all]
        df_file.iloc[:, 7] = [sklearn.metrics.mean_squared_log_error(y_true, y_pred_all[y]) for y in y_pred_all]

        df_file.iloc[:, 3:] = np.round(df_file.iloc[:, 3:], 4)

        erg_latex = open(storagepath + "\\hyperparamopt.txt", "w+")  # write and create (if not there)
        print(df_file.to_latex(), file=erg_latex)
        erg_latex.close()

        df_file



# %%
wrapup(logfile, time_start, newpath)

print("G_NN_hyperparam_tuning.py completed.")

