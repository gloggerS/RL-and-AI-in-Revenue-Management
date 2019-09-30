"""
This file contains helper functions for all other methods. It is standalone and functions just use their parameters.
"""


#%%  PACKAGES
from A_data_read import *

# Data
import numpy as np
import pandas as pd

# Calculation and Counting
import math
from itertools import product
from copy import copy, deepcopy

# Memoization
import functools

# Gurobi
from gurobipy import *
# import re  # rausgenommen am 27.8. beim Überarbeiten von allem, da nicht auftaucht

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Some hacks
# import sys  # rausgenommen am 27.8. beim Überarbeiten von allem, da nicht auftaucht
from contextlib import redirect_stdout
# from ast import literal_eval  # rausgenommen am 27.8. beim Überarbeiten von allem, da nicht auftaucht

# System
from os import getcwd, makedirs
from shutil import copyfile, move

# Time
from datetime import datetime
from time import strftime, time

from random import random, seed




# %% small helpers

def get_offer_sets_all(products):
    """
    Generates all possible offer sets, starting with offering nothing.

    :param products: array of all products that can be offered.
    :return: two dimensional array with rows containing all possible offer sets, starting with offering no product.
    In one row (offer set) a product to be offered is indicated by "1", if the product is not offered "0".
    """
    n = len(products)
    offer_sets_all = np.array(list(map(list, itertools.product([0, 1], repeat=n))))
    offer_sets_all = np.vstack((offer_sets_all, offer_sets_all[0]))[1:]  # move empty offer set to the end
    return offer_sets_all


# @memoize  # easy calculations, save memory
def customer_choice_individual(offer_set_tuple, preference_weights, preference_no_purchase):
    """
    For one customer of one customer segment, determine its purchase probabilities given one offer set and given
    that one customer of this segment arrived.

    Tuple needed for memoization.

    :param offer_set_tuple: tuple with offered products indicated by 1=product offered, 0=product not offered
    :param preference_weights: preference weights of one customer
    :param preference_no_purchase: no purchase preference of one customer
    :return: array of purchase probabilities ending with no purchase
    """

    if offer_set_tuple is None:
        ret = np.zeros_like(preference_weights)
        return np.append(ret, 1 - np.sum(ret))

    offer_set = np.asarray(offer_set_tuple)
    ret = preference_weights * offer_set
    ret = np.array(ret / (np.sum(ret) + preference_no_purchase))
    ret = np.append(ret, 1 - np.sum(ret))
    return ret


# @memoize
def customer_choice_vector(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities):
    """
    From perspective of retailer: With which probability can he expect to sell each product (respectively non-purchase)

    :param offer_set_tuple: tuple with offered products indicated by 1=product offered
    :param preference_weights: preference weights of all customers
    :param preference_no_purchase: preference for no purchase for all customers
    :param arrival_probabilities: vector with arrival probabilities of all customer segments
    :return: array with probabilities of purchase for each product ending with no purchase

    NOTE: probabilities don't have to sum up to one? BEACHTE: Unterschied zu (1) in Bront et all
    """
    if sum(arrival_probabilities) > 1:
        raise ValueError("The sum of all arrival probabilities has to be <= 1.")

    probs = np.zeros(len(offer_set_tuple) + 1)
    for l in np.arange(len(preference_weights)):
        probs += arrival_probabilities[l] * customer_choice_individual(offer_set_tuple, preference_weights[l, :],
                                                                       preference_no_purchase[l])
    return probs


#%% general simulation
def simulate_sales(offer_set, random_customer, random_sales, arrival_probabilities, preference_weights, preferences_no_purchase):
    """
    Simulates a sales event given two random numbers (customer, sale) and a offerset. Eventually, no customer arrives.
    This would be the case if the random number random_customer > sum(arrival_probabilities).

    :param offer_set: Products offered
    :param random_customer: Determines the arriving customer (segment).
    :param random_sales: Determines the product purchased by this customer.
    :param arrival_probabilities: The arrival probabilities for each customer segment
    :param preference_weights: The preference weights for each customer segment (for each product)
    :param preferences_no_purchase: The no purchase preferences for each customer segment.
    :return: The product that has been purchased. No purchase = len(products) = n   (products indexed from 0 to n-1)
    """
    customer = random_customer <= np.array([*np.cumsum(arrival_probabilities), 1.0])
    customer = min(np.array(range(0, len(customer)))[customer])

    if customer == len(arrival_probabilities):
        return len(preference_weights[0])  # no customer arrives => no product sold (product out of range)
    else:
        product = random_sales <= np.cumsum(customer_choice_individual(offer_set,
                                                             preference_weights[customer],
                                                             preferences_no_purchase[customer]))
        product = min(np.arange(len(preference_weights[0]) + 1)[product])
        return product

def simulate_sales_evaluation(offer_set, random_customer, random_sales, arrival_probabilities, preference_weights, preferences_no_purchase):
    """
    Also returns why no-purchase occured:
    1 No customer arrived
    2 Customer arrived but is not interested in any given product
    3 Customer arrived, interested, but chose no-purchase
    0 Customer arrived and purchased a product
    Simulates a sales event given two random numbers (customer, sale) and a offerset. Eventually, no customer arrives.
    This would be the case if the random number random_customer > sum(arrival_probabilities).

    :param offer_set: Products offered
    :param random_customer: Determines the arriving customer (segment).
    :param random_sales: Determines the product purchased by this customer.
    :param arrival_probabilities: The arrival probabilities for each customer segment
    :param preference_weights: The preference weights for each customer segment (for each product)
    :param preferences_no_purchase: The no purchase preferences for each customer segment.
    :return: The product that has been purchased. No purchase = len(products) = n   (products indexed from 0 to n-1)
    """
    customer = random_customer <= np.array([*np.cumsum(arrival_probabilities), 1.0])
    customer = min(np.array(range(0, len(customer)))[customer])



    if customer == len(arrival_probabilities):
        return len(preference_weights[0]), 1  # no customer arrives => no product sold (product out of range)
    elif not any([product_offered and product_considered for product_offered, product_considered in zip(np.array(offer_set) > 0, preference_weights[customer] > 0)]):
        return len(preference_weights[0]), 2  # customer arrived but is not interested in any given product
    else:
        product = random_sales <= np.cumsum(customer_choice_individual(offer_set,
                                                             preference_weights[customer],
                                                             preferences_no_purchase[customer]))
        product = min(np.arange(len(preference_weights[0]) + 1)[product])
        if product == len(preference_weights[0]):
            return product, 3
        else:
            return product, 0

#%% API
def determine_offer_tuple(pi, eps, eps_ran, revenues, A, arrival_probabilities, preference_weights, no_purchase_preference):
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

    # no resources left => nothing to be sold
    if all(pi == np.inf):
        return tuple(np.zeros_like(revenues))

    # epsilon greedy strategy - offer no products
    if eps_ran < eps/2:
        return tuple(np.zeros_like(revenues))

    # epsilon greedy strategy - offer all products
    if eps_ran < eps:
        offer_tuple = np.ones_like(revenues)
        offer_tuple[np.sum(A[list(pi == np.inf), :], axis=0) > 0] = 0  # one resource not available => don't offer product
        return tuple(offer_tuple)

    # setup
    offer_tuple = np.zeros_like(revenues)

    # line 1
    s_prime = revenues - np.nansum(A.T * pi, 1) > 0
    if all(np.invert(s_prime)):
        return tuple(offer_tuple)

    # line 2-3
    # offer_sets_to_test has in each row an offer set, we want to test
    offer_sets_to_test = np.zeros((sum(s_prime), len(revenues)))
    offer_sets_to_test[np.arange(sum(s_prime)), np.where(s_prime)] = 1
    offer_sets_to_test += offer_tuple
    offer_sets_to_test = (offer_sets_to_test > 0)

    value_marginal = np.apply_along_axis(calc_value_marginal, 1, offer_sets_to_test, pi, revenues,
                                         A, arrival_probabilities, preference_weights, no_purchase_preference)

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
        value_marginal = np.apply_along_axis(calc_value_marginal, 1, offer_sets_to_test, pi, revenues,
                                             A, arrival_probabilities, preference_weights, no_purchase_preference)

        if np.amax(value_marginal) >= v_s:
            v_s = np.amax(value_marginal)
            offer_tuple = offer_sets_to_test[np.argmax(value_marginal)]*1  # to get 1 for product offered
            s_prime = (s_prime - offer_tuple) == 1  # only those products remain, that are neither in the offer_tuple
            if all(offer_tuple == 1):
                break
        else:
            break
    return tuple(offer_tuple)


def calc_value_marginal(indices_inner_sum, pi, revenues, A, arrival_probabilities, preference_weights, preferences_no_purchase):
    """
    Calculates the marginal value as indicated at Bront et al, 4.2.2 Greedy Heuristic -> step 4a

    :param indices_inner_sum: C_l intersected with (S union with {j})
    :param pi: vector of dual prices for each resource (np.inf := no capacity)
    :param revenues: vector of revenue for each product
    :param A: matrix with resource consumption of each product (one row = one resource)
    :param arrival_probabilities: The arrival probabilities for each customer segment
    :param preference_weights: The preference weights for each customer segment (for each product)
    :param preferences_no_purchase: The no purchase preferences for each customer segment.
    :return: The value inside the argmax (expected marginal value given one set of products to offer)
    """
    v_temp = 0
    for l in np.arange(len(preference_weights)):  # sum over all customer segments
        tmp_nan_remove = indices_inner_sum * (revenues - np.nansum(A.T * pi, 1))
        where_are_nans = np.isnan(tmp_nan_remove)
        tmp_nan_remove[where_are_nans] = 0
        v_temp += arrival_probabilities[l] * \
                  sum(tmp_nan_remove * preference_weights[l, :]) / \
                  (sum(indices_inner_sum * preference_weights[l, :]) + preferences_no_purchase[l])
    return v_temp

# %% System Helpers
def get_storage_path(storage_location):
    return getcwd()+"\\Results\\"+storage_location


def setup(scenario_name):
    # Get settings, prepare data, create storage for results
    print(scenario_name, "starting.\n\n")

    # settings
    settings = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
    example = settings.iloc[0, 1]
    use_variations = (settings.iloc[1, 1] == "True") | (settings.iloc[1, 1] == "true")  # if var. capacities should be used
    storage_folder = example + "-" + str(use_variations) + "-" + scenario_name + "-" + strftime("%y%m%d-%H%M")
    epsilon = eval(str(settings.loc[settings[0] == "epsilon", 1].item()))
    exponential_smoothing = settings.loc[settings[0] == "exponential_smoothing", 1].item()
    exponential_smoothing = (exponential_smoothing == "True") | (exponential_smoothing == "true")

    # data
    dat = get_all(example)
    print("\n Data used. \n")
    for key in dat.keys():
        print(key, ":\n", dat[key])
    print("\n\n")
    del dat

    # prepare storage location
    newpath = get_storage_path(storage_folder)
    makedirs(newpath)

    # copy settings to storage location
    copyfile("0_settings.csv", newpath+"\\0_settings.csv")
    logfile = open(newpath+"\\0_logging.txt", "w+")  # write and create (if not there)

    # time
    print("Time:", datetime.now())
    print("Time (starting):", datetime.now(), file=logfile)
    time_start = time()

    # settings
    for row in settings:
        print(settings.loc[row, 0], ":\t", settings.loc[row, 1])
        print(settings.loc[row, 0], ":\t", settings.loc[row, 1], file=logfile)

    # variations (capacity and no purchase preference)
    if use_variations:
        var_capacities, var_no_purchase_preferences = get_variations(example)
    else:
        capacities, no_purchase_preferences = get_capacities_and_preferences_no_purchase(example)
        var_capacities = np.array([capacities])
        var_no_purchase_preferences = np.array([no_purchase_preferences])

    # other data
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations(example)
    T = len(times)

    print("\nEverything set up.")

    return logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing


def setup_testing(scenario_name):
    settings = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
    K = int(settings.loc[settings[0] == "K", 1])
    online_K = int(settings.loc[settings[0] == "online_K", 1].item())
    I = int(settings.loc[settings[0] == "I", 1])

    logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start, \
        epsilon, exponential_smoothing = setup(scenario_name)

    return logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I


def wrapup(logfile, time_start, newpath):
    time_elapsed = time() - time_start
    print("\n\nTotal time needed:\n", time_elapsed, "seconds = \n", time_elapsed / 60, "minutes", file=logfile)
    logfile.close()
    print("\n\n\n\nDone. Time elapsed:", time() - time_start, "seconds.")
    print("Results stored in: " + newpath)



#%% move stuff up, if used indeed

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




# %% FUNCTIONS


@memoize
def value_expected(capacities, t, preference_no_purchase, example):
    """
    Recursive implementation of the value function, i.e. dynamic program (DP) as described on p. 241.

    :param capacities:
    :param t: time to go (last chance for revenue is t=0)
    :param preference_no_purchase:
    :return: value to be expected and optimal policy (products to offer)
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations(example)
    T = len(times)

    offer_sets_to_test = get_offer_sets_all(products)

    max_index = 0
    max_val = 0

    if all(capacities == 0):
        return 0, None
    if any(capacities < 0):
        return -math.inf, None
    if t == T:
        return 0, None

    for offer_set_index in range(len(offer_sets_to_test)):
        offer_set = offer_sets_to_test[offer_set_index, :]
        probs = customer_choice_vector(tuple(offer_set), preference_weights, preference_no_purchase,
                                       arrival_probabilities)

        val = float(value_expected(capacities, t + 1, preference_no_purchase, example)[0])
        # ohne "float" würde ein numpy array
        #  zurückgegeben, und das später (max_val = val) direkt auf val verknüpft (call by reference)
        for j in products:  # nett, da Nichtkaufalternative danach (products+1) kommt und so also nicht betrachtet wird
            p = float(probs[j])
            if p > 0.0:
                value_delta_j = delta_value_j(j, capacities, t + 1, A, preference_no_purchase, example)
                val += p * (revenues[j] - value_delta_j)

        if val > max_val:
            max_index = offer_set_index
            max_val = val
    return max_val, tuple(offer_sets_to_test[max_index, :])


def delta_value_j(j, capacities, t, A, preference_no_purchase, example):
    """
    For one product j, what is the difference in the value function if we sell one product.
    TODO: stört mich etwas, Inidices von t, eng verbandelt mit value_expected()

    :param j:
    :param capacities:
    :param t:
    :param preference_no_purchase:
    :return:
    """
    return value_expected(capacities, t, preference_no_purchase, example)[0] - \
        value_expected(capacities - A[:, j], t, preference_no_purchase, example)[0]




# %%
# DECOMPOSITION APPROXIMATION ALGORITHM

# leg level decomposition directly via (11)
@memoize
def value_leg_i_11(i, x_i, t, pi, preference_no_purchase):
    """
    Implements the table of value leg decomposition on p. 776

    :param i:
    :param x_i:
    :param t:
    :param pi:
    :return: optimal value, index of optimal offer set, tuple optimal offer set
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()
    T = len(times)
    lam = sum(arrival_probabilities)

    if t == T+1:
        return 0, None, None, None
    elif x_i <= 0:
        return 0, None, None, None

    offer_sets_all = get_offer_sets_all(products)
    offer_sets_all = pd.DataFrame(offer_sets_all)

    val_akt = 0.0
    index_max = 0

    for index, offer_array in offer_sets_all.iterrows():
        temp = np.zeros_like(products, dtype=float)
        for j in products:
            if offer_array[j] > 0:
                temp[j] = (revenues[j] -
                           (value_leg_i_11(i, x_i, t+1, pi, preference_no_purchase)[0] -
                            value_leg_i_11(i, x_i-1, t+1, pi, preference_no_purchase)[0] - pi[i]) * A[i, j] -
                           sum(pi[A[:, j] == 1]))
        val_new = sum(purchase_rate_vector(tuple(offer_array), preference_weights,
                                           preference_no_purchase, arrival_probabilities)[:-1] * temp)
        if val_new > val_akt:
            index_max = copy(index)
            val_akt = deepcopy(val_new)

    return lam * val_akt + value_leg_i_11(i, x_i, t+1, pi, preference_no_purchase)[0], \
        index_max, tuple(offer_sets_all.iloc[index_max])

def displacement_costs_vector(capacities_remaining, preference_no_purchase, t, pi, beta=1):
    """
    Implements the displacement vector on p. 777

    :param capacities_remaining:
    :param t:
    :param pi:
    :param beta:
    :return:
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    delta_v = 1.0*np.zeros_like(resources)
    for i in resources:
        delta_v[i] = beta * (value_leg_i_11(i, capacities_remaining[i], t, pi, preference_no_purchase)[0] -
                             value_leg_i_11(i, capacities_remaining[i] - 1, t, pi, preference_no_purchase)[0]) + \
                     (1-beta) * pi[i]
    return delta_v


def calculate_offer_set(capacities_remaining, preference_no_purchase, t, pi, beta=1, dataName=""):
    """
    Implements (14) on p. 777

    :param capacities_remaining:
    :param t:
    :param pi:
    :param beta:
    :return: index of optimal offer set, optimal offer set (the products)
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations(dataName)
    lam = sum(arrival_probabilities)

    val_akt = 0
    index_max = 0

    offer_sets_all = get_offer_sets_all(products)
    offer_sets_all = pd.DataFrame(offer_sets_all)

    displacement_costs = displacement_costs_vector(capacities_remaining, preference_no_purchase, t + 2, pi, beta)

    for index, offer_array in offer_sets_all.iterrows():
        val_new = 0
        purchase_rate = purchase_rate_vector(tuple(offer_array), preference_weights,
                                             preference_no_purchase, arrival_probabilities)
        for j in products:
            if offer_array[j] > 0 and all(capacities_remaining - A[:, j] >= 0):
                val_new += purchase_rate[j] * \
                           (revenues[j] - sum(displacement_costs*A[:, j]))
        val_new = lam*val_new

        if val_new > val_akt:
            index_max = copy(index)
            val_akt = deepcopy(val_new)

    return index_max, products[np.array(offer_sets_all.iloc[[index_max]] == 1)[0]] + 1, offer_sets_all

#%%
# Talluri and van Ryzin
def efficient_sets(data_sets):
    """
    Calculates the efficient sets by marginal revenue ratio as discussed in Talluri and van Ryzin p. 21 upper left

    :param data_sets: A dataset with the quantities and revenues of all sets. Index has the sets.
    :return:
    """
    ES = ['0']
    data_sets = data_sets.sort_values(['q'])
    sets_quantities = data_sets.loc[:, 'q']
    sets_revenues = data_sets.loc[:, 'r']

    while True:
        print(ES)
        q_max = max(sets_quantities[ES])
        r_max = max(sets_revenues[ES])
        tocheck = set(data_sets.index[(sets_quantities >= q_max) & (sets_revenues >= r_max)])
        tocheck = tocheck - set(ES)
        if len(tocheck) == 0:
            return ES
        marg_revenues = pd.DataFrame(data=np.zeros(len(tocheck)), index=tocheck)
        for i in tocheck:
            marg_revenues.loc[i] = (sets_revenues[i] - r_max) / (sets_quantities[i] - q_max)
        ES.append(max(marg_revenues.idxmax()))



#%%
def exclude_index(narray: np.array , index_to_exclude):
    return narray[np.arange(len(narray)) != index_to_exclude]

# DPD
def DPD(capacities, preference_no_purchase, dualPrice, t=0):
    """
    Implements Bront et als approach for DPD (12) on p. 776

    :return:
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    val = 0.0
    for i in resources:
        val += value_leg_i_11(i, capacities[i], t, dualPrice, preference_no_purchase)[0]
        temp = dualPrice*capacities
        val += sum(exclude_index(temp, i))

    val = val/len(resources)

    return val
