
from parameters import estimatePrice
import numpy as np

def main():
    try:
        theta = np.load("trained_theta.npy")
    except:
        print("No trained_theta.npy found. Please run ft_linear_regression.py first.")
        exit()
    mileage = int(input("Enter the mileage: "))
    price = estimatePrice(mileage, theta)
    print("The price of the car is: ", price)
    
if __name__ == '__main__':
    main()