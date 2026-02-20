import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""
"""
本文件实现了多种常用于训练神经网络的一阶更新规则。
每种更新规则接收当前权重、以及损失函数关于这些权重的梯度，并生成更新后的权重集合。
所有更新规则均遵循统一的接口规范：

def update(w, dw, config=None):

输入参数：
  - w：一个numpy数组，表示当前的权重。
  - dw：与w形状相同的numpy数组，表示损失函数关于w的梯度。
  - config：一个字典，包含学习率、动量等超参数值。若该更新规则需要在多次迭代中缓存数值，
    则config也会存储这些缓存值。

返回值：
  - next_w：更新后的权重值。
  - config：需要传递给下一次更新规则迭代的配置字典。

注意：对于大多数更新规则而言，默认的学习率很可能无法达到理想效果；
但其他超参数的默认值通常能适配多种不同的问题场景。

为提升效率，更新规则可能会执行**原地更新**（in-place updates）：
直接修改w的数值，并将next_w赋值为修改后的w。
"""

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    v = config["momentum"] * v - dw * config["learning_rate"]
    config.setdefault("velocity",v)
    next_w = w + v

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    
    # 获取参数
    cache = config['cache']
    dr = config['decay_rate']
    lr = config['learning_rate']
    eps = config['epsilon']

    # 更新缓存
    cache = dr * cache + (1 - dr) * (dw**2)
    # 更新权重
    next_w = w - lr * dw / (np.sqrt(cache) + eps)

    # 写回 config
    config['cache'] = cache



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # 获取参数
    m, v, t = config['m'], config['v'], config['t']
    beta1, beta2 = config['beta1'], config['beta2']
    lr, eps = config['learning_rate'], config['epsilon']

    # 1. 更新时间步
    t += 1

    # 2. 更新一阶和二阶矩 (Momentum & RMSProp 结合)
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw**2)

    # 3. 偏置校正 (关键：需要用 t 次方)
    m_unbias = m / (1 - beta1**t)
    v_unbias = v / (1 - beta2**t)

    # 4. 更新权重
    next_w = w - lr * m_unbias / (np.sqrt(v_unbias) + eps)

    # 5. 将更新后的值存回 config
    config['m'], config['v'], config['t'] = m, v, t


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
