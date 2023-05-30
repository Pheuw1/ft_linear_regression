import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_mse(y_true, y_pred):
    n = len(y_true)
    squared_errors = (y_true - y_pred) ** 2
    mse = np.sum(squared_errors) / n
    return mse

def calculate_r2_score(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ssr / sst)
    return r2


def print_metrics(title ,y_true, y_pred):
    mse = calculate_mse(y_true, y_pred)
    r2 = calculate_r2_score(y_true, y_pred)
    print(f"Metrics for {title}:")
    print("Mean Squared Error (MSE):", mse)
    print("Coefficient of Determination (R-squared):", r2)
    
def main():
    data = pd.read_csv("data.csv")

    x  = data["km"].values.reshape(-1, 1)
    y_true  = data["price"].values.reshape(-1, 1)

    theta = np.zeros(2)
    try:
        theta = np.load("trained_theta.npy")
    except:
        print("No trained_theta.npy found. Please run ft_linear_regression.py first.")
        exit()
    print("theta : ",theta)
        
    y_pred = theta[0] + (theta[1] * x)
    print_metrics("trained theta", y_true, y_pred)

    model = LinearRegression()
    model.fit(x, y_true)
    y_pred = model.predict(x)
    print_metrics("sklearn", y_true, y_pred)

if __name__ == '__main__':
    main()