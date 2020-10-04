import numpy as np

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(err)
    return grad, err

def get_mse_loss(y, tx, w):
    """Calculates the mse loss."""
    predicted = tx.dot(w)
    err = y - pred
    return 1/2 * np.mean(err ** 2)

def sigmoid(x):
    """Computes the sigmoid of x."""
    return 1.0 / (1 + np.exp(-x))

def compute_logistic_gradient(y, tx, w) :
    """Gradient of the loss function in logistic regression. """
    """Activation function used here is the sigmoid """
    return tx.T.dot(sigmoid(tx.dot(w)) - y)

def compute_logistic_loss(y, tx, w) : 
    """Loss is given by the negative log likelihood. """
    return np.sum(np.log(1. + np.exp(tx.dot(w))) - y * tx.dot(w))


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    batch_size = 1  # default value as indicated in project description
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Arguments:
        y: target labels
        tx: data features
    Return:
        w: the optimized weights vector for this model
        loss: the final MSE loss of the model
    """
    matrix = tx.T.dot(tx)
    vector = tx.T.dot(y)
    w = np.linalg.solve(matrix, vector)
    loss = get_mse_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    raise NotImplementedError

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using stochastic gradient descent """
    w = initial_w
    for n_iter in range(max_iters) : 
        grad = logistic_gradient(y, tx, w)
        loss = logistic_loss(y, tx, w)
        w = w - gamma * grad       
    return w, loss 

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
        Regularized logistic regression using gradient descent
        Arguments:
            y: target labels
            tx: data features
            lambda_: regularization parameter
            initial_w: w_0, initial weight vector
            max_iters: maximum iterations to run
            gamma: the learning rate or step size
        Returns:
            w: the optimized weights vector for this model
            loss: the final optimized logistic loss
        """
    raise NotImplementedError

