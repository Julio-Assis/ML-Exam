import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)

    # get the shape of the data
    num_samples, _ = data.shape
    
    # extend the data to include the bias term
    extended_data = np.hstack((np.ones((num_samples, 1), dtype=np.float), data))
    
    # calculate the weights
    partial_inverse = np.linalg.inv(np.dot(extended_data.T, extended_data))
    partial_data_times_label = np.dot(extended_data.T, label)
    weight = np.dot(partial_inverse, partial_data_times_label)

    # extract the bias term
    bias = weight[0]
    weight = weight[1:]
    
    return weight, bias