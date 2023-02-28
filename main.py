import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet
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
    # Create a new column converting the date type into an integer, use these values for tracking since start of data
    data['Date'] = pd.to_datetime(data['Date'])
    data.insert(1, 'DateInt', range(0, len(data)))
    print(data)

    '''
    Separate the data into x and y:
        x = The ones column and ID (date as integer)
        y = The average of the High and Low values
    '''
    cols = data.shape[1]
    X = data[['Ones', 'DateInt']]
    Y = data['Average']
    X = np.asarray(X.values)
    Y = np.asarray(Y.values)
    print("X = ", X)
    print("Y = ", Y)

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    model_coef = model.coef_
    print("Model Coefficients: ", model_coef)

    # Create prediction line
    f = model.predict(X).flatten()
    print("f = ", f)

    # Plot the data with the prediction line
    plt.scatter(data['Date'], data['Average'], label='Training Data')
    plt.plot(data['Date'], f, color="red", label='Prediction')
    plt.title("Microsoft Average Daily Stock Price (No adjustments)")
    plt.xlabel("Date")
    plt.ylabel("Average Daily Market Value")
    plt.legend()
    plt.show()

    '''
    Non-Linear model for predicting average daily stock value
    '''

    def generate_polynomial_features(X, degree):
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        return X_poly

    def nonlinear_regression(X, y, degree):
        X_poly = generate_polynomial_features(X, degree)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        return y_pred, model.coef_

    # Low degree -> Underfitted model (like 2)
    # High degree -> Overfitted model (like 10)
    degree = 4
    y_pred, coef = nonlinear_regression(X, Y, degree)

    # Plot the data with the predicted line
    plt.scatter(data['Date'], data['Average'], label='Training Data')
    plt.plot(data['Date'], y_pred, color="red", label='Prediction')
    plt.title(f"Microsoft Average Daily Stock Price (Degree {degree} polynomial)")
    plt.xlabel("Date")
    plt.ylabel("Average Daily Market Value")
    plt.legend()
    plt.show()

    print(f"Model Coefficients: {coef}")

    ''' 
    Non-Linear regression with Elastic Net
    '''
    # @param alpha: The weight of the regularization term
    # @param l1_ratio: The ratio of L1 regularization to L2 regularization
    def nonlinear_regression_elastic(X, y, degree, alpha=1, l1_ratio=0.5):
        X_poly = PolynomialFeatures(degree=degree).fit_transform(X)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        return y_pred, model.coef_

    # Fit the model using Elastic Net regularization
    # Mess with params, overall produces better fit after elastic net at same degree value
    y_pred, model_coef = nonlinear_regression_elastic(X, Y, degree=4, alpha=2, l1_ratio=0.5)

    # Plot the data and the predicted values
    plt.scatter(data['DateInt'], data['Average'], label='Training Data')
    plt.plot(data['DateInt'], y_pred, color='red', label='Predicted')
    plt.title(f"Microsoft Average Daily Stock Price (Elastic Net, Degree {degree})")
    plt.xlabel('Date')
    plt.ylabel('Average Daily Market Value')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
