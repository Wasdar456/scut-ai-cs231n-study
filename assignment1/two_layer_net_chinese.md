这是一个 Jupyter Notebook 文件 (`two_layer_net.ipynb`) 的完整中文翻译。我保留了代码结构，并翻译了所有的 Markdown 文本、代码注释以及打印语句中的说明性文字。

---

### 单元格 1 [Code]

```python
# 挂载 Google Drive 到 Colab 虚拟机。
from google.colab import drive
drive.mount('/content/drive')

# TODO: 输入你在 Drive 中保存解压后的作业文件夹的路径
# 例如：'cs231n/assignments/assignment1/'
FOLDERNAME = 'cs231n/assignments/assignment1/'
assert FOLDERNAME is not None, "[!] 请输入文件夹名称。"

# 挂载 Drive 后，确保 Colab 虚拟机的 Python 解释器
# 可以从该路径加载 python 文件。
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# 如果 Drive 中不存在 CIFAR-10 数据集，则下载它。
%cd /content/drive/My\ Drive/$FOLDERNAME/cs231n/datasets/
!bash get_datasets.sh
%cd /content/drive/My\ Drive/$FOLDERNAME

```

### 单元格 2 [Markdown]

# 全连接神经网络

在这个练习中，我们将使用模块化的方法实现全连接网络。对于每一层，我们将实现一个 `forward`（前向传播）和一个 `backward`（反向传播）函数。`forward` 函数将接收输入、权重和其他参数，并返回输出以及一个存储反向传播所需数据的 `cache`（缓存）对象，如下所示：

```python
def layer_forward(x, w):
  """ 接收输入 x 和权重 w """
  # 做一些计算 ...
  z = # ... 一些中间值
  # 做更多计算 ...
  out = # 输出结果
   
  cache = (x, w, z, out) # 我们计算梯度所需的数值
   
  return out, cache

```

反向传播过程将接收上游导数（upstream derivatives）和 `cache` 对象，并返回关于输入和权重的梯度，如下所示：

```python
def layer_backward(dout, cache):
  """
  接收 dout (损失函数关于输出的导数) 和 cache，
  并计算关于输入的导数。
  """
  # 解包 cache 中的值
  x, w, z, out = cache
  
  # 使用 cache 中的值来计算导数
  dx = # 损失关于 x 的导数
  dw = # 损失关于 w 的导数
  
  return dx, dw

```

用这种方式实现一系列层之后，我们将能够轻松地组合它们来构建具有不同架构的分类器。

### 单元格 3 [Code]

```python
# 像往常一样，做一些设置
from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置绘图的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 用于自动重新加载外部模块
# 参见 http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

def rel_error(x, y):
  """ 返回相对误差 """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

```

### 单元格 4 [Code]

```python
# 加载（预处理过的）CIFAR10 数据。

data = get_CIFAR10_data()
for k, v in list(data.items()):
  print(('%s: ' % k, v.shape))

```

### 单元格 5 [Markdown]

# 仿射层 (Affine layer)：前向传播

打开文件 `cs231n/layers.py` 并实现 `affine_forward` 函数。

完成后，你可以通过运行以下代码来测试你的实现：

### 单元格 6 [Code]

```python
# 测试 affine_forward 函数

num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3

input_size = num_inputs * np.prod(input_shape)
weight_size = output_dim * np.prod(input_shape)

x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

out, _ = affine_forward(x, w, b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])

# 将你的输出与我们的输出进行比较。误差应该在 e-9 左右或更小。
print('Testing affine_forward function:')
print('difference: ', rel_error(out, correct_out))

```

### 单元格 7 [Markdown]

# 仿射层 (Affine layer)：反向传播

现在实现 `affine_backward` 函数，并使用数值梯度检查来测试你的实现。

### 单元格 8 [Code]

```python
# 测试 affine_backward 函数
np.random.seed(231)
x = np.random.randn(10, 2, 3)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)

dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

_, cache = affine_forward(x, w, b)
dx, dw, db = affine_backward(dout, cache)

# 误差应该在 e-10 左右或更小
print('Testing affine_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

```

### 单元格 9 [Markdown]

# ReLU 激活函数：前向传播

在 `relu_forward` 函数中实现 ReLU 激活函数的前向传播，并使用以下代码测试你的实现：

### 单元格 10 [Code]

```python
# 测试 relu_forward 函数

x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

out, _ = relu_forward(x)
correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                        [ 0.,          0.,          0.04545455,  0.13636364,],
                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

# 将你的输出与我们的输出进行比较。误差应该在 e-8 的量级
print('Testing relu_forward function:')
print('difference: ', rel_error(out, correct_out))

```

### 单元格 11 [Markdown]

# ReLU 激活函数：反向传播

现在在 `relu_backward` 函数中实现 ReLU 激活函数的反向传播，并使用数值梯度检查来测试你的实现：

### 单元格 12 [Code]

```python
np.random.seed(231)
x = np.random.randn(10, 10)
dout = np.random.randn(*x.shape)

dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

_, cache = relu_forward(x)
dx = relu_backward(dout, cache)

# 误差应该在 e-12 的量级
print('Testing relu_backward function:')
print('dx error: ', rel_error(dx_num, dx))

```

### 单元格 13 [Markdown]

## 内联问题 1:

我们只要求你实现 ReLU，但神经网络中可以使用许多不同的激活函数，每种都有其优缺点。特别是，激活函数常见的一个问题是在反向传播过程中出现零（或接近零）梯度流（即梯度消失）。以下哪些激活函数存在这个问题？如果考虑一维情况，什么样的输入会导致这种行为？

1. Sigmoid
2. ReLU
3. Leaky ReLU

 *在这里填写*

### 单元格 14 [Markdown]

# "三明治"层 (Sandwich layers)

在神经网络中，经常使用一些常见的层模式。例如，仿射层（affine layers）经常紧跟着一个 ReLU 非线性层。为了让这些常见模式更容易使用，我们在文件 `cs231n/layer_utils.py` 中定义了几个便捷层。

现在请查看 `affine_relu_forward` 和 `affine_relu_backward` 函数，并运行以下代码对反向传播进行数值梯度检查：

### 单元格 15 [Code]

```python
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
np.random.seed(231)
x = np.random.randn(2, 3, 4)
w = np.random.randn(12, 10)
b = np.random.randn(10)
dout = np.random.randn(2, 10)

out, cache = affine_relu_forward(x, w, b)
dx, dw, db = affine_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

# 相对误差应该在 e-10 左右或更小
print('Testing affine_relu_forward and affine_relu_backward:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

```

### 单元格 16 [Markdown]

# 损失层：Softmax

现在在 `cs231n/layers.py` 中的 `softmax_loss` 函数里实现 softmax 的损失和梯度。这应该与你在 `cs231n/classifiers/softmax.py` 中实现的类似。其他的损失函数（例如 `svm_loss`）也可以用模块化的方式实现，但在本次作业中不作要求。

你可以通过运行以下代码来确保实现是正确的：

### 单元格 17 [Code]

```python
np.random.seed(231)
num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)


dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
loss, dx = softmax_loss(x, y)

# 测试 softmax_loss 函数。Loss 应该接近 2.3，dx 误差应该在 e-8 左右
print('\nTesting softmax_loss:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))

```

### 单元格 18 [Markdown]

# 双层网络 (Two-layer network)

打开文件 `cs231n/classifiers/fc_net.py` 并完成 `TwoLayerNet` 类的实现。通读代码以确保你理解了 API。你可以运行下面的单元格来测试你的实现。

### 单元格 19 [Code]

```python
np.random.seed(231)
N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

std = 1e-3
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

print('Testing initialization ... ')
W1_std = abs(model.params['W1'].std() - std)
b1 = model.params['b1']
W2_std = abs(model.params['W2'].std() - std)
b2 = model.params['b2']
assert W1_std < std / 10, '第一层权重看起来不对'
assert np.all(b1 == 0), '第一层偏置看起来不对'
assert W2_std < std / 10, '第二层权重看起来不对'
assert np.all(b2 == 0), '第二层偏置看起来不对'

print('Testing test-time forward pass ... ')
model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.loss(X)
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
assert scores_diff < 1e-6, '测试阶段的前向传播有问题'

print('Testing training loss (no regularization)')
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
correct_loss = 3.4702243556
assert abs(loss - correct_loss) < 1e-10, '训练阶段的损失有问题'

model.reg = 1.0
loss, grads = model.loss(X, y)
correct_loss = 26.5948426952
assert abs(loss - correct_loss) < 1e-10, '正则化损失有问题'

# 误差应该在 e-7 左右或更小
for reg in [0.0, 0.7]:
  print('Running numeric gradient check with reg = ', reg)
  model.reg = reg
  loss, grads = model.loss(X, y)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

```

### 单元格 20 [Markdown]

# 求解器 (Solver)

打开文件 `cs231n/solver.py` 并通读它以熟悉 API。之后，使用 `Solver` 实例来训练一个 `TwoLayerNet`，使其在验证集上达到约 `36%` 的准确率。

### 单元格 21 [Code]

```python
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
model = TwoLayerNet(input_size, hidden_size, num_classes)
solver = None

##############################################################################
# TODO: 使用 Solver 实例来训练一个 TwoLayerNet，                             #
# 使其在验证集上达到约 36% 的准确率。                                        #
##############################################################################

##############################################################################
#                             你的代码结束                                   #
##############################################################################

```

### 单元格 22 [Markdown]

# 调试训练过程

使用上面提供的默认参数，你应该在验证集上获得约 0.36 的验证准确率。这并不是很好。

深入了解问题所在的一个策略是在优化过程中绘制损失函数以及训练集和验证集的准确率。

另一个策略是可视化网络第一层学习到的权重。在大多数在视觉数据上训练的神经网络中，第一层权重在可视化时通常会显示出一些可见的结构。

### 单元格 23 [Code]

```python
# 运行此单元格以可视化训练损失和训练/验证准确率

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

```

### 单元格 24 [Code]

```python
from cs231n.vis_utils import visualize_grid

# 可视化网络权重

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(3, 32, 32, -1).transpose(3, 1, 2, 0)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(model)

```

### 单元格 25 [Markdown]

# 调整你的超参数

**出了什么问题？** 观察上面的可视化结果，我们看到损失随时间大致呈线性下降，这似乎表明学习率可能太低了。此外，训练准确率和验证准确率之间没有差距，这表明我们使用的模型容量较低，我们应该增加其规模。另一方面，如果使用非常大的模型，我们预期会看到更多的过拟合，这表现为训练准确率和验证准确率之间有非常大的差距。

**调优**。调整超参数并建立关于它们如何影响最终性能的直觉是使用神经网络的一大部分内容，所以我们希望你获得大量的练习。在下面，你应该尝试各种超参数的不同值，包括隐藏层大小、学习率、训练轮数（epochs）和正则化强度。你也可以考虑调整学习率衰减，但使用默认值应该也能获得良好的性能。

**近似结果**。你的目标应该是使验证集上的分类准确率超过 48%。我们最好的网络在验证集上达到了超过 52% 的准确率。

**实验**：本次练习中你的目标是使用全连接神经网络在 CIFAR-10 上获得尽可能好的结果（52% 可以作为一个参考）。请随意实现你自己的技术（例如 PCA 降维，或添加 dropout，或向 solver 添加功能等）。

### 单元格 26 [Code]

```python
best_model = None

#################################################################################
# TODO: 使用验证集调整超参数。将训练得最好的模型存储在 best_model 中。          #
#                                                                               #
# 为了帮助调试你的网络，使用类似于上面的可视化可能会有所帮助；                  #
# 这些可视化结果与我们上面看到的调整不当的网络会有显著的质的区别。              #
#                                                                               #
# 手动调整超参数可能很有趣，但你可能会发现编写代码来自动扫描可能的超参数组合    #
# 会很有用，就像我们在之前的练习中所做的那样。                                  #
#################################################################################

################################################################################
#                              你的代码结束                                    #
################################################################################

```

### 单元格 27 [Markdown]

# 测试你的模型！

在验证集和测试集上运行你最好的模型。你应该在验证集和测试集上都获得超过 48% 的准确率。

### 单元格 28 [Code]

```python
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())

```

### 单元格 29 [Code]

```python
y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

```

### 单元格 30 [Code]

```python
# 保存最佳模型
best_model.save("best_two_layer_net.npy")

```

### 单元格 31 [Markdown]

## 内联问题 2:

现在你已经训练了一个神经网络分类器，你可能会发现你的测试准确率远低于训练准确率。有哪些方法可以减小这个差距？选择所有适用的选项。

1. 在更大的数据集上训练。
2. 增加更多的隐藏单元。
3. 增加正则化强度。
4. 以上都不是。