import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW_pre = np.zeros((num_train, W.shape[0], W.shape[1]))
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if j == y[i]:
        pass
        # if margin > 0:
        #   dW_pre[i,:,j] = np.subtract(dW_pre[i,:,j], X[i])

      else:
        if margin > 0:
          loss += margin
          dW_pre[i,:,j] = X[i]
          dW_pre[i,:,y[i]] = np.subtract(dW_pre[i,:,y[i]], X[i])


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW = np.mean(dW_pre, axis=0)
  # print np.sum(dW, axis=1)
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # dW_pre = np.zeros((num_train, W.shape[0], W.shape[1]))
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train),y]
  margins = np.subtract(scores, np.subtract(correct_class_scores,1)[:,np.newaxis])
  margins[np.arange(num_train),y] = 0
  loss = np.sum(margins[margins > 0]) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dW_pre = np.matmul(X[:,:,np.newaxis],
                     (margins > 0)[:,np.newaxis,:].astype(int))
  # print dW_pre.shape
  tmp = np.sum(dW_pre, axis=2)

  for i in xrange(num_train):
    dW_pre[i,:,y[i]] = np.subtract(dW_pre[i,:,y[i]], tmp[i])

  dW = np.mean(dW_pre, axis=0)
  # print np.sum(dW, axis=1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
