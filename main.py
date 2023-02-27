import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn import linear_model
from past.builtins import xrange
from scipy.special import expit

# TODO: Think this is only needed for Colab
# %matplotlib inline

def main():

    # Read in data
    data = pd.read_csv("data/MSFT.csv")

    # Average between high and low values; what the stock is worth on average for any given day
    data['Average'] = (data['High'] + data['Low']) / 2

    # Add the column of ones to the data
    data.insert(0, 'Ones', 1)
    # Create an ID column, use these values for tracking since start of data
    data.insert(1, 'ID', range(0, len(data))) # TODO: This works, but Pycharm throws a type error (the range)
    # pd.to_datetime(data['Date'])
    # pd.to_timedelta(data['Date'], unit='d')
    print(data)

    '''
    Separate the data into x and y:
        x = The ones column and ID (date as integer)
        y = The average of the High and Low values
    '''
    cols = data.shape[1]
    x = data.iloc[:, [0, 1]]
    y = data.iloc[:, cols-1:cols]  # TODO: Maybe just use column names for this
    x = np.asarray(x.values)  # asarray worked the best, got type error with matrix before
    y = np.asarray(y.values)
    #theta = np.asarray(np.array([0, 0])).T
    # Use theta and create matrix
    theta = np.matrix(np.array([0, 0])).T  # TODO: This needs to stay as matrix
    print(data)
    print("x = ", x)
    print("y = ", y)
    print("theta = ", theta)

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(x, y)
    model_coef = model.coef_
    print("Model Coefficients: ", model_coef)

    # Create prediction line
    f = model.predict(x).flatten()
    print("f = ", f)
    plt.plot(x[:, 1], f, color="red", label='Prediction')

    # Plot the data with the prediction line
    # plt.scatter(data['Date'], data['Average'], label='Training Data')
    plt.scatter(pd.to_datetime(data['Date']), data['Average'], label='Training Data')
    plt.title("Microsoft Average Daily Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Average Daily Market Value")
    plt.show()

    # Calculate the cost
    def calcVectorizedCost(x, y, theta):
        # TODO: Both give same values, but the first give more?
        # m = len(y)
        # predictions = x.dot(theta)
        # sqErrors = (predictions - y)
        # j = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
        # return j

        # TODO: Other one
        inner = np.dot(((x * theta) - y).T, (x * theta) - y)
        return inner / (2 * len(x))

    print("Cost: ", calcVectorizedCost(x, y, theta))

    # Create a gradient descent function that takes int the x, y, theta, eta, and iters as parameteres
    def gradientDescent(x, y, theta, eta, iters):
        # Make sure to get rid of this error:
        # ValueError: operands could not be broadcast together with shapes (9083,2) (9083,)
        # Get rid of: ValueError: setting an array element with a sequence.
        # cost = np.zeros(iters, dtype=object)
        cost = np.zeros(iters)
        print("Theta Shape: ", theta.shape)
        theta = theta.reshape(2, 1)
        # theta = np.expand_dims(theta, axis = -1)
        for i in range(iters):
            gradients = 2 * (np.dot(x.T, ((np.dot(x, theta))) - y) / (len(x)))
            # theta = theta.reshape(2, 1)
            theta = theta - eta * gradients
            # The previous line produces the error: ValueError: operands could not be broadcast together with shapes
            # (9083,2) (9083,)
            # Make both are theta is the same shape
            # theta = theta.reshape(9083, 2)

            cost[i] = calcVectorizedCost(x, y, theta)
        return theta, cost

        # temp = np.matrix(np.zeros(theta.shape))
        # parameters = int(theta.ravel().shape[0])
        # # parameters = int(theta.ravel().shape[1])
        # # Rewrite parameters variable to remove tuple index out of range error
        # # parameters = theta.shape[1]
        #
        # cost = np.zeros(iters)
        #
        # for i in range(iters):
        #     error = (x * theta) - y
        #
        #     for j in range(parameters):
        #         term = np.multiply(error, x[:, j])
        #         # term = np.dot(error, x[:, j])
        #         temp[0, j] = theta[0, j] - ((eta / len(x)) * np.sum(term))
        #
        #     theta = temp
        #     cost[i] = calcVectorizedCost(x, y, theta)
        #
        # # Reshape remove this error:
        # # ValueError: operands could not be broadcast together with shapes (9083,2) (9083,)
        # # theta = theta.reshape(9083, 2)
        #
        #
        # return theta, cost

    # Run the gradient descent function
    eta = 0.5
    iters = 1000
    g, cost = gradientDescent(x, y, theta, eta, iters)
    print("Gradient: ", g)
    print("Cost: ", cost)












if __name__ == "__main__":
    main()
