import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    high = np.exp(z) - np.exp(-z)
    low = np.exp(z) + np.exp(-z)

    return high / low

def relu(z):
    return np.where(z>0, z, 0)

def leaky_relu(z, a=0.01):
    return np.where(z>0, z, a*z)

def inverse_square_relu(z, a=0.1):
    coef = np.sqrt(1 + a*(z**2))

    return np.where(z>0, z, z / coef)

def exponential_lu(z, a=0.1):
    return np.where(z>0, z, a*(np.exp(z)-1))
