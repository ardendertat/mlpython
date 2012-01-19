'''
Author: Arden Dertat
Contact: ardendertat@gmail.com
License: MIT License
'''

#!/usr/bin/env python

from numpy import loadtxt, zeros, ones, array, linspace, logspace, dot, transpose
from pylab import scatter, show, title, xlabel, ylabel, plot, contour, subplot


def costFunction(X, y, theta):
    ''' computes squared error cost function'''
    m = y.size
    h = dot(X, theta).flatten()
    return (0.5/m) * sum(pow(h-y, 2))


def gradientDescent(X, y, theta, alpha, iterations):
    m = y.size
    thetaHistory = zeros(shape=(iterations, 2))
    costHistory = zeros(shape=(iterations, 1))
    for i in range(iterations):
        h = dot(X, theta).flatten()
        temp0 = theta[0, 0] - alpha * (1.0/m) * sum(h-y)
        temp1 = theta[1, 0] - alpha * (1.0/m) * sum( (h-y)*X[:,1] )
        theta[0, 0] = temp0
        theta[1, 0] = temp1
        thetaHistory[i, :] = theta.flatten()
        costHistory[i, 0] = costFunction(X, y, theta)
    return theta, thetaHistory, costHistory


def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history
    

if __name__=="__main__":
    data = loadtxt('ex1data1.txt', delimiter=',')
    
    subplot(2, 2, 1)
    scatter(data[:, 0], data[:, 1])
    title('Dataset')
    xlabel('Population of City in 10,000s')
    ylabel('Profit in $10,000s')
    
    m = data.shape[0]
    X = ones(shape=(m, 2))
    X[:, 1] = data[:, 0]
    y = data[:, 1]
    theta = zeros(shape=(2,1))

    theta, thetaHistory, costHistory = gradientDescent(X, y, theta, 0.01, 1000)
    predictions = dot(X, theta).flatten()
    
    subplot(2, 2, 2)
    scatter(data[:, 0], data[:, 1])
    title('Linear Regression')
    xlabel('Population of City in 10,000s')
    ylabel('Profit in $10,000s')
    plot(data[:, 0], predictions)
    
    
    theta0 = linspace(-10, 10, 100)
    theta1 = linspace(-1, 4, 100)
    J = zeros(shape=(theta0.size, theta1.size))
    for i0, t0 in enumerate(theta0):
        for i1, t1 in enumerate(theta1):
            t = array([ [t0], [t1] ])
            J[i0, i1] = costFunction(X, y, t)
    J = transpose(J)
    
    subplot(2, 2, 3)
    contour(theta0, theta1, J, logspace(-2, 3, 20))
    title('Gradient Descent')
    xlabel('theta0')
    ylabel('theta1')
    scatter(theta[0][0], theta[1][0])
    scatter(0, 0, color='red', marker='x')
    for i in range(0, len(thetaHistory), 100):
        scatter(thetaHistory[i, 0], thetaHistory[i, 1], color='red', marker='x')
    
    subplot(2, 2, 4)
    plot(costHistory)
    title('Cost History')
    xlabel('Number of Iterations')
    ylabel('Cost J')
    show()    
