import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from past.builtins import xrange
from scipy.special import expit

# %matplotlib inline

#%matplotlib inline


def main():
    data = pd.read_csv("data/Iris.csv")
    print(data)

    # data["ones"] = "Add your code here"

    data.insert(0, 'Ones', 1)
    print(data)

    # seperate data into X, Y and convert them to numpy arrays.
    # you might need to reshape Y to convert it to a matrix of (100, 1) diminsions.
    cols = data.shape[1]
    X = data.iloc[:, [0, 2, 3]]
    Y = data.iloc[:, [cols - 1]]
    X = np.array(X.values)
    Y = np.array(Y.values)

    # If you have Setosa labeled as 0 and Versicolor is labeled as 1, what is the logistic regression output mean (in words) for a given sample?
    # The output is the probability that the sample belongs to class 1 (versicolor).
    # If the output is 0.5, then the sample belongs to class 0 (setosa) with 50% probability.
    # If the output is 0.8, then the sample belongs to class 1 (versicolor) with 80% probability.
    # If the output is 0.1, then the sample belongs to class 0 (setosa) with 10% probability.
    # If the output is 0.9, then the sample belongs to class 1 (versicolor) with 90% probability.
    # If the output is 0.0, then the sample belongs to class 0 (setosa) with 100% probability.
    # If the output is 1.0, then the sample belongs to class 1 (versicolor) with 100% probability.
    # If the output is 0.2, then the sample belongs to class 0 (setosa) with 20% probability.


    print("X shape =", X.shape, ", Y shape =", Y.shape, "X type =", type(X), "Y type =", type(Y))
    print(X)
    print(Y)
    # results should be "X shape = (100, 3) , Y shape = (100, 1)
    # X type = <class 'numpy.ndarray'> Y type = <class 'numpy.ndarray'>"

    # access class-based data
    setosa = data.loc[data['Species'] == 0]
    versicolor = data.loc[data['Species'] == 1]

    # # data plotting and specifications
    # plt.scatter(setosa['PetalLengthCm'], setosa['PetalWidthCm'], marker="+")
    # plt.scatter(versicolor['PetalLengthCm'], versicolor['PetalWidthCm'], marker="o")
    #
    # # labeling specification
    # plt.xlabel('PetalLengthCm')
    # plt.ylabel('PetalWidthCm')
    #
    # # legend and show calls
    # plt.legend(["setosa", "versicolor"])
    # plt.show()

    def hypothesis(X, theta):
        h = 1 / (1 + np.exp(-X.dot(theta)))
        return h

    # this is a test function for the hypothesis function implemented above.
    theta = np.zeros((X.shape[1], 1))
    h = hypothesis(X, theta)
    print(h.shape)  # this should be (100, 1)
    print(h[0])  # this should be 0.5 for all values in h

    def calcLogRegressionCost(X, Y, theta):
        """
        Calculate Logistic Regression Cost

        X: Features matrix
        Y: Output matrix
        theta: matrix of variable weights
        output: return the cost value.
        """
        # cost = add code here
        m = len(Y)
        h = hypothesis(X, theta)
        cost = 1 / m * (-Y.T.dot(np.log(h)) - (1 - Y).T.dot(np.log(1 - h)))

        return cost

    # This is a test function that will call "calcLogRegressionCost" using the theta initial parameters (zeros).
    # You should get about 0.693.
    theta = np.zeros((X.shape[1], 1))
    print(calcLogRegressionCost(X, Y, theta))

    # Desired output: 0.693...

    def logRegressionGradientDescent(X, Y, theta, eta, iters):
        """
        Performs gradient descent optimization on a set of data

        X: Features matrix
        Y: Output matrix
        theta: matrix of variable weights
        eta: learning rate
        iters: number of times to iterate the algorithm (epochs)
        output: return optimized theta and the cost array for each iteration (epoch).
        """
        # add your code here ...
        m = len(Y)
        cost = np.zeros(iters)  # TODO: This causes array / sequence error?

        for i in range(iters):
            # cost.append(0)
            h = hypothesis(X, theta)
            gradients = (2 / m) * X.T.dot(h - Y)
            theta = theta - eta * gradients
            cost[i] = calcLogRegressionCost(X, Y, theta)  # TODO: This!

        return theta, cost

    # this is a test function for logRegressionGradientDescent function. You can change eta and iters
    # but you will not get the values given below.
    eta = 0.1
    iters = 10000
    theta = np.zeros((X.shape[1], 1))
    theta, cost = logRegressionGradientDescent(X, Y, theta, eta, iters)
    print(calcLogRegressionCost(X, Y, theta))
    print(theta)
    # you should get these values
    # For cost [[0.00358296]]
    # For theta [[-11.73342243]
    #  [  3.24410413]
    #  [  4.8722057 ]]

    # Plot the cost by the number of iters.
    # plt.plot(cost)
    # plt.xlabel("Iterations")
    # plt.ylabel("Cost")
    # plt.show()




    x_new = np.array([[1, 4.5, 1.5]])
    # write your code here.
    print(hypothesis(x_new, theta))

    # Plot
    def plotData(feature1, feature2, label1, label2, feature1AxisLabel, feature2AxisLabel):
        plt.figure(figsize=(10, 6))
        plt.plot(feature1[:, 1], feature1[:, 2], 'ko', label=label1)
        plt.plot(feature2[:, 1], feature2[:, 2], 'r+', label=label2)
        plt.xlabel(feature1AxisLabel)
        plt.ylabel(feature2AxisLabel)
        plt.legend()
        plt.grid()
        plt.show()

    def plotDecisionBoundary(X, theta):
        # this will find min,max x values and solve for y = 0 at those positions
        boundary_xs = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
        boundary_ys = -1 * (-.5 + theta[0] + theta[1] * boundary_xs) / theta[2]

        # plot points
        plt.plot(boundary_xs, boundary_ys, 'b-', label='Decision Boundary')
        plt.show()

    pos = np.array([X[i] for i in xrange(X.shape[0]) if Y[i] == 1])
    neg = np.array([X[i] for i in xrange(X.shape[0]) if Y[i] == 0])

    # plotData(pos, neg, "versicolor", "setosa", 'PetalLengthCm', 'PetalWidthCm')
    # plotDecisionBoundary(X, theta)









    '''
        Part 2: Non-linear and Regularized Logistic Regression
    '''

    exData = pd.read_csv("data/ex2data2.txt")
    exData.insert(0, 'Ones', 1)
    cols = exData.shape[1]
    X = exData.iloc[:, [0, 1, 2]]
    Y = exData.iloc[:, [cols - 1]]
    X = np.array(X.values)
    Y = np.array(Y.values)

    # this code will visualize your data, test 1 will be in the X-axis,
    # test 2 will be on the Y-axis.
    # we assigned '+' for pass and 'o' for fail
    # your asnwer should look the figure below
    # print(X)
    pos = np.array([X[i] for i in xrange(X.shape[0]) if Y[i] == 1])
    neg = np.array([X[i] for i in xrange(X.shape[0]) if Y[i] == 0])

    def plotData(pos, neg):
        plt.plot(pos[:, 1], pos[:, 2], 'k+', label='y=1')
        plt.plot(neg[:, 1], neg[:, 2], 'yo', label='y=0')

        plt.legend()
        plt.grid()
        plt.show()

    plotData(pos, neg)

    # poly = add your code here
    # X_poly = add your code here
    poly = PolynomialFeatures(6)
    X_poly = poly.fit_transform(X[:, 1:3])
    # X_poly = poly.fit_transform(X)

    # This part is given for testing purpose, you can change eta and iters to different values
    # but you are not going to have same cost[0] reported below.
    eta = 0.5
    iters = 10000

    theta_poly_init = np.zeros((X_poly.shape[1], 1))
    theta_poly, cost = logRegressionGradientDescent(X_poly, Y, theta_poly_init, eta, iters)
    print(cost[0])  # cost [0] should be 0.6812373150879889

    def plotBoundary(theta, X, Y, poly, eta, iters):

        # find optimal thetas
        theta, cost = logRegressionGradientDescent(X, Y, theta, eta, iters)

        # create search space and placeholder
        xvals = np.linspace(-1, 1.5, 100)
        yvals = np.linspace(-1, 1.5, 100)
        zvals = np.zeros((len(xvals), len(yvals)))

        # compute zval for all combinations of xvals/yvals
        for i in range(len(xvals)):
            for j in range(len(yvals)):
                featuresij = poly.fit_transform(np.array([[xvals[i], yvals[j]]]))
                zvals[j][i] = np.dot(theta.T, featuresij.T)

        contour = plt.contour(xvals, yvals, zvals, [0])
        plt.title("Decision Boundary ")
        plt.show

    plt.figure(figsize=(6, 6))
    plotData(pos, neg)
    plotBoundary(theta_poly, X_poly, Y, poly, eta, iters)


    # Comment on the decision boundary. Is the decision boundary behaving as expected? Try higher and lower polynomial degrees and comment on the differences
    # Answer: The decision boundary is behaving as expected. The higher the polynomial degree, the more complex the decision boundary becomes. The lower the polynomial degree, the less complex the decision boundary becomes.



    # # You will need to create a PolynomialFeatures object (with the degree hyperparam set to 6) which we will denote as the variable "poly".
    # # Then you need to transform the data using this object that you have created.
    # # Use the docs to assist you with understanding and implementing your code https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    # # Note: please note that you are going to apply polynomials only on the original features without the added column of ones, as the sklearn PolynomialFeatures automatically adds a "ones" column to the output transformed X (No need to add "ones" after you transform using sklearn PolynomialFeatures).
    # poly = PolynomialFeatures(6)
    # X_poly = poly.fit_transform(X[:, 1:3])
    #
    #
    # # This part is given for testing purpose, you can change eta and iters to different values
    # # but you are not going to have same cost[0] reported below.
    # eta = 0.5
    # iters = 10000
    #
    # theta_poly_init = np.zeros((X_poly.shape[1], 1))
    # theta_poly, cost = logRegressionGradientDescent(X_poly, Y, theta_poly_init, eta, iters)
    # print(cost[0])



if __name__ == '__main__':
    main()