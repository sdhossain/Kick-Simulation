import numpy as np

def model_eval(function_type, input, ridge_coeff):
    """
    Evaluates output based on regression coefficients (inference)

    :param function_type: type of function, currently only supports Sigmoid
    :param input: input for regression, ex:- contractile-element velocity
    :param ridge_coeff: coefficients obtain from ridge-regression fitting
    :return output: returns output of regression
    """

    if function_type == 'Sigmoid':
        fun = lambda x, mu, sigma: 1 / (1 + np.exp(-(x-mu) / sigma))
        X = np.array([fun(input, i, 0.15) for i in np.arange(-1, -0.09, 0.2)])

    elif function_type == 'Gaussian':
        # Add code for Gaussian function here if needed
        pass

    X = np.reshape(X, ridge_coeff[1:].shape)

    output = ridge_coeff[0] + np.dot(X, ridge_coeff[1:])

    return output