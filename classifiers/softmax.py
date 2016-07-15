import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dim = X.shape[1]
  for i in xrange(num_train):
    z = X[i].dot(W)
    p = np.exp(z) / np.sum(np.exp(z))
    loss -= np.log(p)[y[i]]

    for j in xrange(dim):
      for k in xrange(num_classes):
        if k == y[i]:
          dW[j,k] += (p[k] - 1) * X[i,j]
        else:
          dW[j,k] += p[k] * X[i,j]

  dW = dW / num_train
  loss /= num_train
  loss += 0.5 * np.sum(W*W) * reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dim = X.shape[1]
  z = X.dot(W) # N x C
  p_tmp = np.exp(z)
  p = np.divide(p_tmp, np.sum(p_tmp, axis=1, keepdims=True)) # N x C
  # dW_pre = np.zeros((num_train, dim, num_classes)) # N x D x C
  dW_pre = np.multiply(X[:,:,np.newaxis], p[:,np.newaxis,:])

  for i in range(num_train):
    dW_pre[i,:,y[i]] -= X[i]

  dW = np.mean(dW_pre, axis=0)
  dW += reg * W
  loss = -np.mean(np.log(p[np.arange(num_train), y])) + 0.5 * np.sum(W*W) * reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

