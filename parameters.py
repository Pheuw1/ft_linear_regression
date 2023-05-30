import numpy as np
def estimatePrice(mileage, theta):
    return (theta[0] + (theta[1] * mileage))

global alpha, num_iters
alpha = 0.1
num_iters = 1000
