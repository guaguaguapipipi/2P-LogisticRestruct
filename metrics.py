import sklearn.metrics as mt
import numpy as np
import math


def getMSE(y, y_hat):
    MSE = np.sum((y - y_hat) ** 2) / len(y)
    return MSE


def getRMSE(y, y_hat):
    RMSE = mt.mean_squared_error(y, y_hat) ** 0.5
    return RMSE


def getMAE(y, y_hat):
    MAE = np.sum(np.absolute(y - y_hat)) / len(y)
    return MAE


def getMAPE(y, y_hat):
    MAPE = mt.mean_absolute_percentage_error(y, y_hat)
    return MAPE


def getR2(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    return (SSR / SST) ** 2