import numpy as np
from proj1_helpers import batch_iter


def compute_gradient(y, tx, w):
    """Compute the gradient.
	Parameters:
		y: target lables; an array of shape (N,1). N: Number of datapoints.
		tx: datapoint features; an array of shape (N,D+1). N: number of datapoints, D: number of features.  
		w: current weights.
	Returns:
		the gradient of the MSE loss function using w as weights.
	""" 
    N = len(y)
    e = y-tx.dot(w) #np.matmul(tx, w)
    return (tx.T.dot(e))/-N

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(err)
    return grad, err

def get_mse_loss(y, tx, w):
    """Calculates the mse loss."""
    pred = tx.dot(w)
    err = y - pred
    return 1/2 * np.mean(err ** 2)

def sigmoid(x):
    """Computes the sigmoid of x."""
    return 1.0 / (1 + np.exp(-x))


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
	"""Gradient descent algorithm.
	Parameters:
		y: target lables; an array of shape (N,1). N: Number of datapoints.
		tx: datapoint features; an array of shape (N,D+1). N: number of datapoints, D: number of features.  
		initial_w: initial weights.
		max_ietrs: number of iterations.
		gamma: learning rate.
	Returns: 
		w: optimized weights.
		loss: final loss.
	
	"""
	w = initial_w
	loss = 0
	w = initial_w
	for n_iter in range(max_iters):
		loss = get_mse_loss(y, tx, w)
		grad = compute_gradient(y, tx, w) 
		w = w - gamma * grad
	return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    batch_size = 1  # default value as indicated in project description
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = get_mse_loss(y, tx, w)
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
	"""
	Ridge regression using normal equations.
	Parameters:
		y: target lables; an array of shape (N,1). N: Number of datapoints. 
		tx: datapoint features; an array of shape (N,D+1). N: number of datapoints, D: number of features.
		lambda_: the hyperparametrs used to balance the tradeoff between model complexity and cost.
	Returns:
		w: optimized weights.
	"""
	lambda_ = lambda_ * 2 * len(y)
	M = tx.T.dot(tx)
	w = np.linalg.solve(M + lambda_*np.identity(M.shape[0]), tx.T.dot(y))
	return w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError


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

