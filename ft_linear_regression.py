import numpy as np
from parameters import *
from pandas import read_csv
import sys

def linear_regression(x, y, theta, alpha, num_iters):
    m = len(y)
    for _ in range(num_iters):
        h = estimatePrice(x, theta)
        theta[0] = theta[0] - alpha * (1/m) * np.sum(h - y)
        theta[1] = theta[1] - alpha * (1/m) * np.sum((h - y) * x)
    return theta

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def denormalize(x, x_orig):
    return (x * (x_orig.max() - x_orig.min())) + x_orig.min()

def plot(x, y, theta):
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'ro', ms=10, mec='k')
    plt.ylabel('Price')
    plt.xlabel('Mileage')
    plt.title('Price vs. Mileage')
    plt.plot(x, theta[0] + (theta[1] * x), '-')
    plt.show()

def main():
    global alpha, num_iters
    theta = np.zeros(2)
    data = read_csv("data.csv")
    km, price = data["km"], data["price"]
    x = km / km.max()
    y = price / price.max()
    theta = linear_regression(x, y, theta, alpha, num_iters)
    theta[0] = theta[0] * price.max()
    theta[1] = theta[1] * price.max() / km.max()
    np.save("trained_theta", theta)
    print("Theta found by gradient descent: %f, %f" % (theta[0], theta[1]))
    if (len(sys.argv) == 2 and sys.argv[1] == "-p"):
        plot(km, price, theta)

if __name__ == '__main__':
    main()