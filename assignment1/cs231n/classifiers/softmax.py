from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label c, where 0 <= c < C.
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dW = (1/m) * dZ.dot(A_prev.T)

    num_classes = W.shape[1]
    num_examples = X.shape[0]

    # print(dW.shape, X.shape)

    for (i, x) in enumerate(X):
        f = x.dot(W)
        f -= np.max(f)
        softmax = np.exp(f) / np.sum(np.exp(f))
        # print(f"softmax shape {softmax.shape}")
        # print(f"example (x) shape {x.shape}")
        loss += -np.log(softmax[y[i]])
        for j in range(num_classes):
            if j != y[i]:
                dW[:, j] += x * softmax[j]
            else:
                dW[:, j] += x * (softmax[j] - 1)
            # dW[:, y[i]] -= x

    loss /= num_examples
    loss += reg * np.sum(W*W)
    dW /= num_examples
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    # print(f"X shape {X.shape}")
    f = np.dot(X, W)
    # print(f.shape)
    f -= np.max(f, axis=1, keepdims=True)
    softmax = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)
    # print(softmax.shape)
    loss = np.sum(-np.log(softmax[np.arange(len(X)), y]))
    loss /= num_train
    loss += reg * np.sum(W*W)

    dS = softmax.copy()
    dS[range(num_train), list(y)] += -1
    dW = (X.T).dot(dS)
    dW = dW/num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
