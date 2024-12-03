import sys as system
import numpy as np
from params import params 
from linear_regression import linear_regression as ln
import signal

def signal_handler(signum, frame):
    print(f"\nSignal {signum} received")
    # Perform necessary cleanup
    exit(0)

def waiting_for_input():
    """
    Summary:
    Blocks the program until the user enters text.
    """
    str = ""
    try:
        while str == "" or str.isdigit() is False:
            str = input("Enter a mileage (km): ")
    except EOFError:
        print("\nEOF error")    
        exit(1)
    return (str)


def main():
    """
    Summary:
    Estimates the price of a given mileage in km with a linear regression
    model.
    """
    try:
        signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

        input = waiting_for_input()
        input = int(input)

        # creating X matrix
        X = np.array(input)
        X = np.hstack((X, np.ones(1)))

        # creating theta matrix
        theta = np.hstack((np.array(params.theta1), np.array(params.theta0))).reshape(2, 1)

        # Result
        res = int(ln.model(X, theta)[0])
        if res < 0:
            print(f"The model's price prediction for a car with {input} km is 0$")
        else:
            print(f"The model's price prediction for a car with {input} km is {res}$")
    except Exception as e:
        print(type(e).__name__ + ":", e)

if __name__ == "__main__":
    main()