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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        dscores = p.copy()
        dscores[y[i]] -=1 #这边的dscores实际上是dLoss/dscores

        dW += np.outer(X[i],dscores)#这边在算dloss/dw  实际上
        #第j行m列是，图片的第j个特征对第m个图片的分数的影响权重，乘以这个dLoss/dscores
        #也就是分数到损失，这边的具体推到看笔记cs231n的p1p2








    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    dW /=num_train

    dW += 2* reg *W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
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

    scores = X.dot(W)
    num_train = X.shape[0]

    shift_scores = scores - np.max(scores,axis=1,keepdims = True)
    #对每一行取一个最大的数字，最后是一个（n，1）的向量，然后广播减去
    #keepdims = True 保证结果是(n,1)

    exp_scores = np.exp(shift_scores)
    probs = exp_scores/np.sum(exp_scores,axis = 1,keepdims=True)
    
    # 我们需要：第0行取y[0]列，第1行取y[1]列...
    # NumPy 高级索引技巧：probs[行索引列表, 列索引列表]
    correct_logprobs = -np.log(probs[np.arange(num_train), y])

    loss = np.sum(correct_logprobs) / num_train + reg * np.sum(W * W)

    #计算dw
    # 1. 复用算好的概率矩阵
    dscores = probs.copy()
    
    # 2. 对每一行的正确类别减 1
    # 利用刚才一样的高级索引技巧
    dscores[np.arange(num_train), y] -= 1
    
    # 3. 矩阵乘法算出 dW
    # 这一个 dot 操作，这就替代了那整个 for 循环和 += outer
    dW = X.T.dot(dscores)
    
    # 4. 平均化 + 正则化 (这步别忘了！)
    dW /= num_train
    dW += 2 * reg * W








    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
