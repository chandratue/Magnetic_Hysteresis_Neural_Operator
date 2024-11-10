import numpy as np

def min_max_scaling(data, min_val=-1, max_val=1):
    data_min = np.min(data)
    data_max = np.max(data)
    scaled_data = min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)
    return scaled_data

def inverse_min_max_scaling(scaled_data, data, min_val=-1, max_val=1):
    data_min = np.min(data)
    data_max = np.max(data)
    inverted_data = ((scaled_data - min_val) / (max_val - min_val)) * (data_max - data_min) + data_min
    return inverted_data