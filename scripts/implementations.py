{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def logistic_regression(y, tx, initial_w, max_iters, gamma) : \n",
    "    \n",
    "    raise NotImplementedError \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Least squares sgd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_stoch_gradient(y, tx, w):\n",
    "    \"\"\"Compute a stochastic gradient from just few examples n and their corresponding y_n labels.\"\"\"\n",
    "    err = y - tx.dot(w)\n",
    "    grad = - tx.T.dot(err) / len(err)\n",
    "    return grad, err"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def least_squares_SGD(y, tx, initial_w, max_iters, gamma):\n",
    "    \"\"\"Stochastic gradient descent algorithm.\"\"\"\n",
    "    w = initial_w\n",
    "    batch_size=1                      #default value as indicated in project description\n",
    "    for n_iter in range(max_iters):\n",
    "        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):\n",
    "\n",
    "            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)\n",
    "            w = w - gamma*grad\n",
    "            loss = compute_loss(y, tx, w) \n",
    "\n",
    "    return w, loss "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
