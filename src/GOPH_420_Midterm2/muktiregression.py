import numpy as np


def multiregression(y, Z):
    """
    preform multiple linear regression
    ----------
    parameters:
    y : arraylike, shape (n, ) or (n,1)
        the vector of dependent variables
    Z : arraylike, shape (n, m)
        the matrix of independent variables

    returns:
    ----------
    numpy.ndarray, shape (m, ) or (m,1)
        the vector of regression coefficients
    numpy.ndarray, shape (n, ) or (n,1)
        the vector of residuals
    float
        the coefficient of determination (R^2)
    """
    y = np.array(y)
    Z = np.array(Z)

    y_mean = np.mean(y)

    ZtZ = np.dot(Z.T, Z)
    ZtY = np.dot(Z.T, y)
    aCoeff = np.linalg.solve(ZtZ, ZtY)

    ey = y - y_mean
    em = y.flatten() - np.dot(Z, aCoeff)
    Sy = np.dot(ey.T, ey)
    Sr = np.dot(em.T, em)


    R2 = (Sy - Sr) / Sy

    return aCoeff, em, R2