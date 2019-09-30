"""
This script will calculate the MLP regressor with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
https://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/

IDEE: Alles analog zu API single leg, aber jetzt theta und pi mit MLP Regressor lernen
"""


from A_data_read import *
from B_helper import *

from keras.models import Sequential
from keras.layers import Input, Dense, Embedding
from keras.utils import plot_model
from keras import optimizers
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model

from keras.models import Model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings

from mpl_toolkits import mplot3d

#%%
# Setup of parameters
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("NN2")

#%%
capacities = np.array([3])
preferences_no_purchase = var_no_purchase_preferences[0]

times = np.arange(10)
T = len(times)

#%%
print(capacities[0]*T)

#%%
K = 3
I = 10


#%%
def update_parameters(v_samples, c_samples, thetas, pis, k):
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



    c_np = c_samples.reshape(-1, 1)
    t_np = np.array([np.arange(T)] * I).reshape(-1,1)

    v_np = v_samples.reshape(-1, 1)

    X = np.concatenate((c_np, t_np), axis=1)
    y = v_np


#%% tables to visualize means of value function
    c_max = np.max(c_np)
    df_means, df_numbers, tab_means = value_analysis(c_samples, X, y)
    plot_surface(tab_means, X, y)
    plot_bar(tab_means, X, y)

#%%
    model = load_model('20190803-1558-NN1.h5')
    model.predict(X)

#%% NN1 Direct Output
    model1 = Sequential()
    model1.add(Dense(1, activation='linear', input_shape=(2,)))
    model1.summary()

    # plot_model1(model1, to_file='model1-NN2.png')

    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    opt = optimizers.Adadelta(lr=100)

    model1.compile(optimizer=opt, loss='mean_squared_error')
    model1.fit(X, y, epochs=10000, callbacks=[es])

    model1.predict(X)

    K.eval(model1.optimizer.lr)

    mean_squared_error(y, model1.predict(X))

    model1.save('NN1.h5')
    # model1.get_weights()

#%% NN2 with embedding
    L = np.zeros((X.shape[0], (int(max(c_np))+1)*T))
    for i in np.arange(L.shape[0]):
        L[i, int(X[i, 0]*T + X[i, 1])] = 1

    inputs = Input(shape=(L.shape[1],))
    preds = Dense(1, activation='linear')(inputs)
    model2 = Model(inputs, preds)
    model2.summary()

    # plot_model1(model1, to_file='model1-NN2.png')

    # simple early stopping
    es = EarlyStopping(patience=3)

    opt = optimizers.Adadelta(lr=100)

    model2.compile(optimizer=opt, loss='mean_squared_error')
    model2.fit(L, y, epochs=10000, callbacks=[es])

    y_pred = model2.predict(L)
    np.around(y_pred)


    model2.save('NN2.h5')
    # model1.get_weights()

#%% NN Direct Output
    inputs = Input(shape=(2,))
    preds = Dense(1, activation='linear')(inputs)
    model2 = Model(inputs, preds)
    model2.summary()
    
    model2.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    model2.fit(X, y, batch_size=1, epochs=30)

    model2.predict(X)

    model2.get_weights()
#%%
    # linear regression
    reg = LinearRegression()
    reg.fit(X, y)

    reg.predict(X)

    mean_squared_error(y, reg.predict(X))

# %%
    neural_net = MLPRegressor(alpha=0.1, hidden_layer_sizes = (10,), max_iter = 50000,
                 activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')
    m = neural_net.fit(X, y)

    y_true = y
    y_pred = m.predict(X)
    r2_score(y_true, y_pred)


    # Still the OLS for comparison
    model = sm.OLS(y, X).fit()
    model.params


    y_pred_ols = model.predict(X)  # make the predictions by the model
    r2_score(y_true, y_pred_ols)

    # Print out the statistics
    model.summary()

    theta_new = np.array([[i] for i in model.params[0:len(times)]])
    pi_new = np.array([[i] for i in model.params[len(times):]])

    # check exponential smoothing
    if exponential_smoothing:
        return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new
    else:
        return theta_new, pi_new


def tidy_up(t_df):
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


# %%
# Actual Code

pi_result = {}
theta_result = {}
value_result = {}
capacities_result = {}

# generate the random sample paths
np.random.seed(12)
random.seed(12)
customer_stream = [np.random.random(T) for _ in range(I)]
sales_stream = [np.random.random(T) for _ in range(I)]

# for capacities in var_capacities:
    value_result[str(capacities)] = {}
    capacities_result[str(capacities)] = {}
    theta_result[str(capacities)] = {}
    pi_result[str(capacities)] = {}

    # for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ", preferences_no_purchase, "of",
              str(var_no_purchase_preferences.tolist()), "starting.")

        value_result[str(capacities)][str(preferences_no_purchase)] = {}
        capacities_result[str(capacities)][str(preferences_no_purchase)] = {}

        # store parameters over all policy iterations
        # K+1 policy iterations (starting with 0)
        # T time steps
        theta_all = np.array([[np.zeros(1)] * T] * (K + 1))
        pi_all = np.array([[np.zeros(len(resources))] * T] * (K + 1))

        # theta and pi for each time step
        # line 1
        thetas = 0
        pis = np.zeros(len(resources))

        for k in np.arange(K) + 1:
            print(k, "of", K, "starting.")
            random.seed(13)  # to have the epsilon's (exploration vs exploitation) also the same for each policy iteration k

            v_samples = np.array([np.zeros(len(times))] * I)
            c_samples = np.array([np.zeros(shape=(len(times), len(capacities)))] * I)

            for i in np.arange(I):
                # line 3
                r_sample_i = np.zeros(len(times))  # will become v_sample
                c_sample_i = np.zeros(shape=(len(times), len(capacities)), dtype=int)

                # line 5
                c = copy.deepcopy(capacities)  # (starting capacity at time 0)

                for t in times:
                    # line 7  (starting capacity at time t)
                    c_sample_i[t] = c

                    # line 8-11  (adjust bid price)
                    pis[c_sample_i[t] == 0] = np.inf
                    pis[c_sample_i[t] > 0] = pi_all[k - 1][t][c_sample_i[t] > 0]

                    # line 12  (epsilon greedy strategy)
                    offer_set = determine_offer_tuple(pis, epsilon[k], revenues, A,
                                                      arrival_probabilities, preference_weights,
                                                      preferences_no_purchase)

                    # line 13  (simulate sales)
                    sold = simulate_sales(offer_set, customer_stream[i][t], sales_stream[i][t],
                                          arrival_probabilities, preference_weights, preferences_no_purchase)

                    # line 14
                    try:
                        r_sample_i[t] = revenues[sold]
                        c -= A[:, sold]
                    except IndexError:
                        # no product was sold
                        pass

                # line 16-18
                v_samples[i] = np.cumsum(r_sample_i[::-1])[::-1]
                c_samples[i] = c_sample_i

            # line 20
            theta_all[k], pi_all[k] = update_parameters(v_samples, c_samples, theta_all[k - 1], pi_all[k - 1], k)

            value_result[str(capacities)][str(preferences_no_purchase)][k] = v_samples
            capacities_result[str(capacities)][str(preferences_no_purchase)][k] = c_samples

        theta_result[str(capacities)][str(preferences_no_purchase)] = theta_all
        pi_result[str(capacities)][str(preferences_no_purchase)] = pi_all

# %%
# write result of calculations
with open(newpath + "\\thetaAll.data", "wb") as filehandle:
    pickle.dump(theta_result, filehandle)

with open(newpath + "\\piAll.data", "wb") as filehandle:
    pickle.dump(pi_result, filehandle)

with open(newpath + "\\valueAll.data", "wb") as filehandle:
    pickle.dump(value_result, filehandle)

with open(newpath + "\\capacitiesAll.data", "wb") as filehandle:
    pickle.dump(capacities_result, filehandle)

with open(newpath + "\\thetaToUse.data", "wb") as filehandle:
    tmp = {}
    for capacities in var_capacities:
        tmp[str(capacities)] = {}
        for preferences_no_purchase in var_no_purchase_preferences:
            tmp[str(preferences_no_purchase)] = theta_result[str(capacities)][str(preferences_no_purchase)][K]
    pickle.dump(tmp, filehandle)

with open(newpath + "\\piResult.data", "wb") as filehandle:
    tmp = {}
    for capacities in var_capacities:
        tmp[str(capacities)] = {}
        for preferences_no_purchase in var_no_purchase_preferences:
            tmp[str(preferences_no_purchase)] = pi_result[str(capacities)][str(preferences_no_purchase)][K]
    pickle.dump(tmp, filehandle)

# %%
wrapup(logfile, time_start, newpath)
