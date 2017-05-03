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
  #pass
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)    
    scores -= np.max(scores) 
    sum_cores = np.sum(np.exp(scores))
    correct_class_score = scores[y[i]]

    p = np.exp(correct_class_score) / sum_cores # safe to do, gives the correct answer
    loss -= np.log(p)

    for j in range(num_classes):
	tmp = ((j==y[i]) - np.exp(scores[j]) / np.sum(np.exp(scores)))
	dW[:, j] -= (X[i, :]*tmp).T

  loss /= num_train
  dW   /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
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

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  scores = X.dot(W)
  c = np.amax(scores, axis=1)
  scores -= c.reshape(scores.shape[0], 1)
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis = 1)
  correct_exp_scores = exp_scores[range(num_train), y]  
  loss = np.mean(-np.log(correct_exp_scores/sum_exp_scores))


  tmp = np.zeros_like(scores)
  tmp = -exp_scores / sum_exp_scores.reshape(tmp.shape[0], 1)
  tmp[range(num_train), y] += 1
  dW  -= X.T.dot(tmp)
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW   += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

