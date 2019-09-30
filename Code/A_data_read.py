""""
This script provides functionality to load data by name.
"""
import pickle

pickle_off = open("0-data_by_name", "rb")
data = pickle.load(pickle_off)
pickle_off.close()


# %% Export functionality
def get_data(name):
    return data[name]["resources"], data[name]["capacities"], \
           data[name]["products"], data[name]["revenues"], data[name]["A"], \
           data[name]["customer_segments"], data[name]["preference_weights"], data[name]["preference_no_purchase"], \
           data[name]["arrival_probabilities"], \
           data[name]["times"]


def get_data_without_variations(name):
    return data[name]["resources"], \
           data[name]["products"], data[name]["revenues"], data[name]["A"], \
           data[name]["customer_segments"], data[name]["preference_weights"], \
           data[name]["arrival_probabilities"], \
           data[name]["times"]


def get_capacities_and_preferences_no_purchase(name):
    return data[name]["capacities"], data[name]["preference_no_purchase"]


def get_variations(name):
    return data[name]["var_capacities"], data[name]["var_no_purchase_preferences"]


def get_preference_no_purchase(name):
    return data[name]["preference_no_purchase"]


def get_capacities_thresholds(name):
    return data[name]["capacities_thresholds"]


def get_all(name):
    return data[name]
