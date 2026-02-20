from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    N = x.shape[0]
    x_reshaped = x.reshape(N,-1)
    out = x_reshaped.dot(w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               ########
    # ##################################################################
    N = x.shape[0]
    x_reshaped = x.reshape(N,-1)

    db = np.sum(dout,axis = 0)
    
    dw = x_reshaped.T.dot(dout)

    dx_reshaped = dout.dot(w.T)

    dx = dx_reshaped.reshape(x.shape)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout.copy()

    # 2. 利用布尔索引：将所有 x <= 0 的位置对应的梯度设为 0
    # 原理：ReLU 在 x > 0 时导数为 1 (dx = 1 * dout)，在 x <= 0 时导数为 0 (dx = 0)
    dx[x <= 0] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    计算批量归一化（Batch Normalization）的前向传播。

    训练阶段：从小批次数据中计算样本均值和（未校正的）样本方差，
    并用于归一化输入数据；同时维护每个特征的均值和方差的指数衰减滑动平均，
    这些平均值在测试阶段用于归一化数据。

    每一步都会用动量参数更新均值和方差的滑动平均：
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    注意：批量归一化的论文中建议测试阶段用大量训练图像计算每个特征的
    样本均值和方差，而非滑动平均；但本实现选择滑动平均（torch7的BN实现
    也用滑动平均），无需额外的估计步骤。

    输入：
    - x: 数据，形状为 (N, D)
    - gamma: 缩放参数，形状为 (D,)
    - beta: 偏移参数，形状为 (D,)
    - bn_param: 字典，包含以下键：
      - mode: 'train' 或 'test'；必填
      - eps: 数值稳定性常数（防止除0）
      - momentum: 滑动平均的动量常数
      - running_mean: 形状为 (D,) 的数组，保存特征的滑动均值
      - running_var: 形状为 (D,) 的数组，保存特征的滑动方差

    返回值（元组）：
    - out: 输出，形状为 (N, D)
    - cache: 反向传播需要的中间值元组
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        ###########################################################################
        # TODO: 实现训练阶段的BN前向传播                                          #
        # 1. 首先，计算小批次数据中每个特征维度的均值（sample_mean）和方差（sample_var）；
        # 2. 用这些统计量归一化输入数据；                                         #
        # 3. 然后用gamma（缩放）和beta（偏移）调整归一化后的数据；                 #
        # 4. 你需要将最终输出存储在变量out中；                                    #
        # 5. 同时，你需要结合动量参数（momentum），用计算出的样本均值和方差        #
        #    更新运行均值（running_mean）和运行方差（running_var），并将结果      #
        #    存储到running_mean和running_var变量中；                              #
        # 注意：尽管你需要维护运行方差（running_var），但在归一化数据时，          #
        #       必须基于标准差（方差的平方根）来计算，而非直接使用方差！          #
        # 6. 最后，将反向传播需要的所有中间值存储到cache中（比如x、归一化后的x、   #
        #    中心化后的x、标准差、gamma、beta、样本均值、样本方差、eps等）。       #
        ###########################################################################
        # 请在这里写你的代码
        sample_mean = np.mean(x,axis=0)
        sample_var = np.var(x,axis=0)
        # 2. 中心化输入（反向传播需要这个中间值）
        x_centered = x - sample_mean

        # 3. 计算标准差（加eps防止除0）
        std = np.sqrt(sample_var + eps)

        # 4. 归一化
        x_norm = x_centered / std

        out = gamma * x_norm + beta#话说这个格式对吗，不需要转置吗
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean#这个动量参数就是怎么保留之前的吗？，没动用这个的意义是什么，我记得sgd的改良版有动量
        running_var = momentum * running_var + (1 - momentum) * sample_var#这样计算为什么是对的？方差可以直接这样加减吗？我觉得应该是方差的平根加减然后根才对啊
        #必须基于标准差（方差的平方根）来计算，而非直接使用方差！ 是什么意思
        cache = (x, x_centered, x_norm, std, gamma, beta, sample_mean, sample_var, eps)

        pass
        ###########################################################################
        #                          代码结束标记                                   #
        ###########################################################################
    elif mode == "test":
        ###########################################################################
        # TODO: 实现测试阶段的BN前向传播                                          #
        # 1. 用预先维护的运行均值（running_mean）和运行方差（running_var）来归一化输入；
        # 2. 同样，归一化时要使用标准差（方差的平方根 + eps）；                   #
        # 3. 再用gamma和beta缩放、偏移归一化后的数据；                             #
        # 4. 你需要将最终输出存储在变量out中。                                    #
        ###########################################################################
        x_centered = x - running_mean
        std = np.sqrt(running_var + eps)
        x_norm = x_centered / std
        out = gamma * x_norm + beta
        # 请在这里写你的代码
        pass
        ###########################################################################
        #                          代码结束标记                                   #
        ###########################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # 将更新后的滑动均值/方差存回bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    计算批量归一化的反向传播。

    实现建议：先在纸上画出BN的计算图，再通过中间节点反向传播梯度。
    计算图：x → x_centered (x - sample_mean) → x_norm (x_centered/std) → out (gamma*x_norm + beta)

    输入：
    - dout: 上游梯度，形状为 (N, D)
    - cache: batchnorm_forward输出的中间值元组（train模式下）

    返回值（元组）：
    - dx: 关于输入x的梯度，形状为 (N, D)
    - dgamma: 关于缩放参数gamma的梯度，形状为 (D,)
    - dbeta: 关于偏移参数beta的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 实现BN的反向传播                                                  #
    # 步骤1：求dbeta（最简单）：dout在样本维度（axis=0）求和；                  #
    # 步骤2：求dgamma：dout * x_norm，再在样本维度求和；                       #
    # 步骤3：求dx_norm：dout * gamma（缩放梯度反向）；                         #
    # 步骤4：求dstd：dx_norm * x_centered * (-1) / (std^2)，再求和；           #
    # 步骤5：求dvar：dstd * 0.5 / std；                                       #
    # 步骤6：求dx_centered_part1：dx_norm / std（x_norm对x_centered的梯度）；   #
    # 步骤7：求dx_centered_part2：2 * x_centered * dvar / N（var对x_centered的梯度）；
    # 步骤8：dx_centered = dx_centered_part1 + dx_centered_part2；             #
    # 步骤9：求dmean：-1 * np.sum(dx_centered, axis=0)；                      #
    # 步骤10：dx = dx_centered + dmean / N（mean对x的梯度）；                  #
    ###########################################################################
    # 从cache中取出所有中间值
    x, x_centered, x_norm, std, gamma, beta, sample_mean, sample_var, eps = cache
    N, D = x.shape
    dbeta = np.sum(dout,axis = 0)#我知道按照计算图是dout = dbeta 也知道按照这个格式的确是要axis = 0求和，但是我不能理解究竟为什么是求和，不懂，求和的意义是什么？我觉得应该是直接等于dout才对啊，为什么还要求和呢？难道是因为每一列的dbeta都是一样的吗？我觉得不太对啊
    dgamma = np.sum(dout * x_norm,axis = 0 ) #为什么这边是不线形代数的那种矩阵乘法呢？我知道这两个一定是乘法，但是为什么是逐个相乘呢 还有numpy的dot是什么乘法，是点乘，还是就是线形代数的那种乘法,同时为什么不能dbeta来乘呢
    dx_norm = dout * gamma
    dstd = np.sum(dx_norm * x_centered * (-1) / (std ** 2),axis = 0)
    dvar = dstd * 0.5 / std
    dx_centered_part1 = dx_norm / std
    dx_centered_part2 = 2 * x_centered * dvar / N
    dx_centered = dx_centered_part1 + dx_centered_part2#var也是根据x_centered平方算出来的
    dmean = -1 * np.sum(dx_centered, axis=0)
    dx = dx_centered + dmean / N


    # 请在这里补全反向传播代码
    pass
    ###########################################################################
    #                          代码结束标记                                   #
    ###########################################################################

    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    计算层归一化（Layer Normalization）的前向传播。

    训练和测试阶段的行为完全相同：对每个数据样本单独归一化，
    再用与BN相同的gamma和beta参数缩放、偏移。

    注意：与BN不同，层归一化无需维护任何滑动平均。

    输入：
    - x: 数据，形状为 (N, D)
    - gamma: 缩放参数，形状为 (D,)
    - beta: 偏移参数，形状为 (D,)
    - ln_param: 字典，包含以下键：
        - eps: 数值稳定性常数

    返回值（元组）：
    - out: 输出，形状为 (N, D)
    - cache: 反向传播需要的中间值元组
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: 实现层归一化的前向传播                                            #
    # 提示：只需对批量归一化的代码稍作修改即可！核心区别：                     #
    # 1. BN按axis=0（样本维度）计算均值/方差，LN按axis=1（特征维度）计算；    #
    # 2. LN不需要维护running_mean/running_var；                                #
    # 3. 其他逻辑（中心化、归一化、缩放+偏移）完全和BN一样；                   #
    ###########################################################################
    N, D = x.shape

    # 步骤1：按特征维度（axis=1）计算每个样本的均值和方差
    sample_mean = np.mean(x,axis = 1,keepdims = True)  # 替换为你的代码（提示：np.mean(x, axis=1, keepdims=True)）
    sample_var = np.var(x,axis = 1,keepdims= True)  # 替换为你的代码（提示：np.var(x, axis=1, keepdims=True)）

    # 步骤2：中心化输入
    x_centered = x - sample_mean  # 替换为你的代码

    # 步骤3：计算标准差
    std = np.sqrt(sample_var + eps)  # 替换为你的代码

    # 步骤4：归一化
    x_norm = x_centered / std  # 替换为你的代码

    # 步骤5：缩放+偏移
    out = gamma * x_norm + beta

    # 步骤6：保存反向传播需要的中间值
    cache = (x, x_centered, x_norm, std, gamma, beta, sample_mean, sample_var, eps)
    ###########################################################################
    #                          代码结束标记                                   #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    计算层归一化的反向传播。

    实现建议：可以大量复用批量归一化的代码，核心区别：
    1. BN按axis=0求和，LN按axis=1求和；
    2. 其他梯度推导逻辑完全和BN一样。

    输入：
    - dout: 上游梯度，形状为 (N, D)
    - cache: layernorm_forward输出的中间值元组

    返回值（元组）：
    - dx: 关于输入x的梯度，形状为 (N, D)
    - dgamma: 关于缩放参数gamma的梯度，形状为 (D,)
    - dbeta: 关于偏移参数beta的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 实现层归一化的反向传播                                            #
    # 提示：只需对批量归一化的代码稍作修改即可！核心区别：                     #
    # 1. BN按axis=0求和，LN按axis=1求和；                                      #
    # 2. 其他梯度推导逻辑完全和BN一样；                                         #
    ###########################################################################
    # 从cache中取出所有中间值
    x, x_centered, x_norm, std, gamma, beta, sample_mean, sample_var, eps = cache
    N, D = x.shape

    dbeta = np.sum(dout,axis=0)
    dgamma = np.sum(dout * x_norm,axis=0)
    dx_norm = dout * gamma
    dstd = np.sum(dx_norm * x_centered * (-1) / (std ** 2), axis=1)
    dvar = dstd * 0.5 / std
    dx_centered_part1 = dx_norm / std
    dx_centered_part2 = 2 * x_centered * dvar
    dx_centered = dx_centered_part1 + dx_centered_part2
    dmean = -1 * np.sum(dx_centered, axis=1)
    dx = dx_centered + dmean / D

    ###########################################################################
    #                          代码结束标记                                   #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        # TODO: 实现训练阶段的反向Dropout前向传播
        # 步骤1：生成和x形状相同的随机掩码mask：每个元素以概率p为1（保留），1-p为0（丢弃）
        # 步骤2：mask生成后，除以p（反向Dropout的核心缩放）
        # 步骤3：x乘以mask，得到输出out
        # 步骤4：保存mask到cache
        # 提示：用np.random.rand()生成0~1的随机数，和p比较得到布尔掩码，再转成和x相同的数值类型
        #######################################################################
        mask = (np.random.rand(*x.shape) < p)/p
        out = x * mask
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    N = x.shape[0]
    
    shift_scores = x - np.max(x,axis=1,keepdims = True)
    #对每一行取一个最大的数字，最后是一个（n，1）的向量，然后广播减去
    #keepdims = True 保证结果是(n,1)

    exp_scores = np.exp(shift_scores)
    probs = exp_scores/np.sum(exp_scores,axis = 1,keepdims=True)
    
    # 我们需要：第0行取y[0]列，第1行取y[1]列...
    # NumPy 高级索引技巧：probs[行索引列表, 列索引列表]
    correct_logprobs = -np.log(probs[np.arange(N), y])

    loss = np.sum(correct_logprobs) / N 

    
    # 1. 复用算好的概率矩阵
    dx = probs.copy()
    
    # 2. 对每一行的正确类别减 1
    # 利用刚才一样的高级索引技巧
    dx[np.arange(N), y] -= 1
    
    dx /= N
    
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
