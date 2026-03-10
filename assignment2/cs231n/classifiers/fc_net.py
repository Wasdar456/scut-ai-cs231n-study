from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """
    多层全连接神经网络类

    网络包含任意数量的隐藏层、ReLU非线性激活函数，以及softmax损失函数。同时可选实现dropout机制和批归一化/层归一化。对于L层的网络，整体架构为：

    {仿射变换 - [批/层归一化] - ReLU激活 - [dropout]} × (L - 1) - 仿射变换 - softmax

    其中批/层归一化和dropout为可选项，{...} 代码块会重复 L-1 次。

    可学习的参数存储在 self.params 字典中，将通过 Solver 类完成训练学习。
    """

    def __init__(
            self,
            hidden_dims,
            input_dim=3 * 32 * 32,
            num_classes=10,
            dropout_keep_ratio=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """
        初始化一个新的全连接网络

        输入参数:
        - hidden_dims: 整数列表，指定每一个隐藏层的维度大小
        - input_dim: 整数，指定输入数据的维度
        - num_classes: 整数，指定分类任务的类别数量
        - dropout_keep_ratio: 0到1之间的标量，控制dropout的强度。
            若 dropout_keep_ratio=1，则网络完全不使用dropout
        - normalization: 网络使用的归一化类型，有效值为 "batchnorm"（批归一化）、
            "layernorm"（层归一化），默认None为不使用归一化
        - reg: 标量，L2正则化的强度
        - weight_scale: 标量，权重随机初始化时的标准差
        - dtype: numpy的数据类型对象，所有计算都将使用该数据类型执行。
            float32 速度更快但精度更低，数值梯度检查时建议使用 float64
        - seed: 若不为None，会将该随机种子传递给dropout层。
            这会让dropout层的行为固定可复现，方便我们对模型进行梯度检查
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 初始化网络的所有参数，将所有值存储在 self.params 字典中
        # 第一层的权重和偏置存储在 W1 和 b1 中；第二层使用 W2 和 b2，以此类推
        # 权重应从均值为0、标准差为 weight_scale 的正态分布中初始化
        # 偏置应初始化为0
        #
        # 当使用批归一化时，第一层的缩放(scale)和偏移(shift)参数存储在 gamma1 和 beta1 中
        # 第二层使用 gamma2 和 beta2，以此类推
        # 缩放参数应初始化为1，偏移参数应初始化为0
        ############################################################################
        # 统一处理前 (num_layers - 1) 层，也就是所有的隐藏层
        for i in range(1, self.num_layers):
            # 这里填初始化 W_i, b_i 的代码
            if i == 1:
                self.params[f'W{i}'] = np.random.randn(input_dim , hidden_dims[i-1]) * weight_scale
            else:
                self.params[f'W{i}'] = np.random.randn(hidden_dims[i-2] , hidden_dims[i-1]) * weight_scale
            self.params[f'b{i}'] = np.zeros(hidden_dims[i-1])
            if self.normalization is not None:
                self.params[f'gamma{i}'] = np.ones(hidden_dims[i-1])
                self.params[f'beta{i}'] = np.zeros(hidden_dims[i-1])

            pass
        self.params[f'W{self.num_layers}'] = np.random.randn(hidden_dims[-1], num_classes)*weight_scale
        self.params[f'b{self.num_layers}'] = np.zeros(num_classes)

        # 单独处理最后一层 (第 num_layers 层)
        # 这里填初始化 W_last, b_last 的代码

        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        # 使用dropout时，我们需要给每个dropout层传递一个 dropout_param 字典
        # 让层知道dropout的概率和运行模式（训练/测试）
        # 你可以给所有dropout层传递同一个 dropout_param
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # 使用批归一化时，我们需要跟踪滑动均值和滑动方差
        # 因此需要给每个批归一化层传递一个专用的 bn_param 对象
        # 你需要把 self.bn_params[0] 传给第一个批归一化层的前向传播
        # self.bn_params[1] 传给第二个批归一化层的前向传播，以此类推
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # 将所有参数转换为指定的数据类型
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        计算全连接网络的损失和梯度

        输入参数:
        - X: 输入数据数组，形状为 (N, d_1, ..., d_k)
        - y: 标签数组，形状为 (N,)，y[i] 对应 X[i] 的类别标签

        返回值:
        如果 y 为 None，运行模型的测试阶段前向传播，返回:
        - scores: 形状为 (N, C) 的分类分数数组，其中 scores[i, c] 是
            输入X[i]对应类别c的分类分数

        如果 y 不为 None，运行模型的训练阶段前向+反向传播，返回元组:
        - loss: 标量，总损失值
        - grads: 字典，键和 self.params 完全一致，键值为损失对应该参数的梯度
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # 给批归一化参数和dropout参数设置训练/测试模式
        # 因为它们在训练和测试阶段的行为是不同的
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: 实现全连接网络的前向传播，计算输入X的分类分数，并存入scores变量
        #
        # 使用dropout时，你需要给每个dropout的前向传播传递 self.dropout_param
        #
        # 使用批归一化时，你需要把 self.bn_params[0] 传给第一个批归一化层的前向传播
        # self.bn_params[1] 传给第二个批归一化层的前向传播，以此类推
        ############################################################################
        x = X.reshape(X.shape[0], -1)  # 将输入数据展平为 (N, D)
        caches = []
        for i in range(1, self.num_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']

            #affine  - [batchnorm] - relu- [dropout]
            #仿射变换 - [批/层归一化] - ReLU激活 - [dropout]} × (L - 1) - 仿射变换 - softmax
            #affine
            x, af_cache = affine_forward(x, W, b)
            caches.append(af_cache)

            #norm
            if self.normalization == "batchnorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                x, bn_cache = batchnorm_forward(x, gamma, beta, self.bn_params[i-1])
                caches.append(bn_cache)
            elif self.normalization == "layernorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                x, ln_cache = layernorm_forward(x, gamma, beta, self.bn_params[i-1])
                caches.append(ln_cache)

            #relu
            x, relu_cache = relu_forward(x)
            caches.append(relu_cache)

            #dropout
            if self.use_dropout:
                x, dropout_cache = dropout_forward(x, self.dropout_param)
                caches.append(dropout_cache)

        W_last = self.params[f'W{self.num_layers}']
        b_last = self.params[f'b{self.num_layers}']
        scores, cache_last = affine_forward(x, W_last, b_last)
        caches.append(cache_last)


        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        # 测试模式下提前返回
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: 实现全连接网络的反向传播，将损失值存入loss变量，梯度存入grads字典
        # 使用softmax计算数据损失，确保 grads[k] 存储的是损失对 self.params[k] 的梯度
        # 不要忘记添加L2正则化！
        #
        # 使用批/层归一化时，不需要对缩放和偏移参数做正则化
        #
        # 注意：为了保证你的实现和我们的一致，并且能通过自动化测试
        # 请确保你的L2正则化项包含0.5的系数，以简化梯度的表达式
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        regloss = 0
        for i in range(1, self.num_layers + 1):
            W = self.params[f'W{i}']
            regloss += 0.5 * self.reg * np.sum(W * W)
        loss = data_loss + regloss

        cache = caches.pop()  # 最后一层的缓存
        dx, dW, db = affine_backward(dscores, cache)
        grads[f'W{self.num_layers}'] = dW + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db

        for i in range(self.num_layers-1):
            idx = self.num_layers - i -1# 从最后一层开始反向遍历
            #总共有num_layers层，去掉最后一层，idx从num_layers - 1到1，依次处理每一层的反向传播

            # 处理隐藏层的反向传播
            if self.use_dropout:
                dropout_cache = caches.pop()
                dx = dropout_backward(dx, dropout_cache)

            relu_cache = caches.pop()
            dx = relu_backward(dx, relu_cache)

            if self.normalization == "batchnorm":
                bn_cache = caches.pop()
                dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache)
                grads[f'gamma{idx}'] = dgamma
                grads[f'beta{idx}'] = dbeta
            elif self.normalization == "layernorm":
                ln_cache = caches.pop()
                dx, dgamma, dbeta = layernorm_backward(dx, ln_cache)
                grads[f'gamma{idx}'] = dgamma
                grads[f'beta{idx}'] = dbeta

            af_cache = caches.pop()
            dx, dW, db = affine_backward(dx, af_cache)
            grads[f'W{idx}'] = dW + self.reg * self.params[f'W{idx}']
            grads[f'b{idx}'] = db


        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        return loss, grads

    def save(self, fname):
        """保存模型参数"""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """加载模型参数"""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True

