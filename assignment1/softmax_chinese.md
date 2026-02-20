这是一份关于 **Softmax 分类器练习**（来自斯坦福大学 CS231n 课程）的 Jupyter Notebook 文件的详细中文翻译。

---

## Softmax 分类器练习

*完成并提交本工作表（包括其输出和工作表之外的任何辅助代码）。更多详情请访问课程网站上的 [作业页面](http://vision.stanford.edu/teaching/cs231n/assignments.html)。*

在本练习中，你将：

* 为 Softmax 分类器实现一个完全**向量化（fully-vectorized）的损失函数**。
* 为其**解析梯度（analytic gradient）**实现完全向量化的表达式。
* 使用数值梯度**检查你的实现**。
* 使用验证集来**调节学习率和正则化**强度。
* 使用 **SGD**（随机梯度下降）**优化**损失函数。
* **可视化**最终学习到的权重。

---

### 代码单元格 1：设置环境

```python
# 将你的 Google Drive 挂载到 Colab 虚拟机。
from google.colab import drive
drive.mount('/content/drive')

# TODO: 输入你保存解压后作业文件夹的 Drive 文件夹名称，
# 例如 'cs231n/assignments/assignment1/'
FOLDERNAME = 'cs231n/assignments/assignment1/'
assert FOLDERNAME is not None, "[!] 请输入文件夹名称。"

# 挂载 Drive 后，确保 Colab 虚拟机的 Python 解释器
# 可以从中加载 python 文件。
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# 如果 CIFAR-10 数据集尚不存在，则将其下载到你的 Drive。
%cd /content/drive/My\\ Drive/$FOLDERNAME/cs231n/datasets/
!bash get_datasets.sh
%cd /content/drive/My\\ Drive/$FOLDERNAME

```

### 代码单元格 2：配置库

```python
# 为本 notebook 运行一些设置代码。
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# 这是一个魔法命令，使 matplotlib 图形在 notebook 内联显示，
# 而不是在新窗口中显示。
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置默认绘图尺寸
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 更多魔法命令，以便 notebook 自动重新加载外部 python 模块；
# 参见 http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

```

### 代码单元格 3：数据加载

```python
# 加载原始 CIFAR-10 数据。
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# 清理变量以防止多次加载数据（这可能导致内存问题）
try:
   del X_train, y_train
   del X_test, y_test
   print('清除之前加载的数据。')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 作为完整性检查，我们打印训练和测试数据的尺寸。
print('训练数据形状: ', X_train.shape)
print('训练标签形状: ', y_train.shape)
print('测试数据形状: ', X_test.shape)
print('测试标签形状: ', y_test.shape)

```

### 代码单元格 4：数据可视化

```python
# 可视化数据集中的一些示例。
# 我们展示了每个类别中一些训练图像的示例。
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

```

### 代码单元格 5：数据划分

```python
# 将数据分为训练集、验证集和测试集。此外，我们将
# 创建一个小型开发集（development set）作为训练数据的子集；
# 我们可以将其用于开发，这样我们的代码运行得更快。
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# 我们的验证集将是原始训练集中的 num_validation 个点。
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# 我们的训练集将是原始训练集中的前 num_train 个点。
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# 我们还将创建一个开发集，它是训练集的一个随机子集。
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# 我们使用原始测试集的前 num_test 个点作为我们的测试集。
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print('训练数据形状: ', X_train.shape)
print('训练标签形状: ', y_train.shape)
print('验证数据形状: ', X_val.shape)
print('验证标签形状: ', y_val.shape)
print('测试数据形状: ', X_test.shape)
print('测试标签形状: ', y_test.shape)

```

### 代码单元格 6：数据预处理

```python
# 预处理：将图像数据重塑为行向量
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# 作为完整性检查，打印出数据的形状
print('训练数据形状: ', X_train.shape)
print('验证数据形状: ', X_val.shape)
print('测试数据形状: ', X_test.shape)
print('开发数据形状: ', X_dev.shape)

# 预处理：减去平均图像
# 第一步：根据训练数据计算图像均值
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # 打印前 10 个元素
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # 可视化平均图像
plt.show()

# 第二步：从训练和测试数据中减去平均图像
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# 第三步：追加全 1 的偏置维度（即偏置技巧），
# 这样我们的分类器只需处理一个权重矩阵 W 的优化。
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

```

---

## Softmax 分类器

你本节的所有代码都将写在 `cs231n/classifiers/softmax.py` 中。

如你所见，我们已经预填了函数 `softmax_loss_naive`，该函数使用 for 循环来评估 Softmax 损失函数。

### 代码单元格 8：初步评估

```python
# 评估我们为你提供的损失函数的原始（naive）实现：
from cs231n.classifiers.softmax import softmax_loss_naive
import time

# 生成一个由随机小数组成的随机 Softmax 分类器权重矩阵
W = np.random.randn(3073, 10) * 0.0001

loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
print('loss: %f' % (loss, ))

# 作为初步的完整性检查，我们的损失应该接近 -log(0.1)。
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))

```

**内联问题 1**

为什么我们预期损失会接近 -log(0.1)？请简要解释。

 *在此处填写*

---

### 代码单元格 10：梯度检查

```python
# 一旦你实现了梯度，请使用下面的代码重新计算它
# 并使用我们提供的函数进行梯度检查

# 计算 W 处的损失及其梯度。
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# 在几个随机选择的维度上数值计算梯度，
# 并将它们与你解析计算的梯度进行比较。在所有维度上，这些数值应该几乎完全匹配。
from cs231n.gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# 在开启正则化的情况下再次进行梯度检查
# 你没忘记正则化梯度吧？
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)

```

**内联问题 2**

虽然 Softmax 损失的梯度检查（gradcheck）是可靠的，但对于 SVM 损失，偶尔可能会出现梯度检查中某个维度不完全匹配的情况。这种差异可能由什么原因引起？这是值得担心的问题吗？在的一维空间中，SVM 损失梯度检查可能失败的简单例子是什么？改变边界（margin）会如何影响这种情况发生的频率？

*提示：严格来说，SVM 损失函数不是可微的。*

 *在此处填写*

---

### 代码单元格 12：向量化实现对比

```python
# 接下来实现函数 softmax_loss_vectorized；目前只计算损失；
# 我们稍后会实现梯度。
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

from cs231n.classifiers.softmax import softmax_loss_vectorized
tic = time.time()
loss_vectorized, _ = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# 损失应该匹配，但你的向量化实现应该快得多。
print('difference: %f' % (loss_naive - loss_vectorized))

```

### 代码单元格 13：完成向量化梯度

```python
# 完成 softmax_loss_vectorized 的实现，并以向量化的方式计算损失函数的梯度。

# 原始实现和向量化实现应该匹配，但向量化版本应该快得多。
tic = time.time()
_, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss and gradient: computed in %fs' % (toc - tic))

tic = time.time()
_, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss and gradient: computed in %fs' % (toc - tic))

# 损失是一个数值，因此很容易比较两个实现计算出的值。
# 另一方面，梯度是一个矩阵，因此我们使用 Frobenius 范数来比较它们。
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('difference: %f' % difference)

```

---

### 随机梯度下降 (SGD)

### 代码单元格 15：训练模型

```python
# 在 linear_classifier.py 文件中，实现 LinearClassifier.train() 函数中的 SGD，
# 然后使用下面的代码运行它。
from cs231n.classifiers import Softmax
softmax = Softmax()
tic = time.time()
loss_hist = softmax.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print('That took %fs' % (toc - tic))

```

### 代码单元格 16：绘制损失曲线

```python
# 一个有用的调试策略是将损失作为迭代次数的函数绘制出来：
plt.plot(loss_hist)
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.show()

```

### 代码单元格 17：预测与准确率

```python
# 编写 LinearClassifier.predict 函数并评估其在训练集和验证集上的表现。
# 你的验证准确率应该达到约 0.34 (> 0.33)。
y_train_pred = softmax.predict(X_train)
print('训练准确率: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = softmax.predict(X_val)
print('验证准确率: %f' % (np.mean(y_val == y_val_pred), ))

```

---

### 代码单元格 19：超参数调节

```python
# 使用验证集来调节超参数（正则化强度和学习率）。
# 你应该尝试不同范围的学习率和正则化强度；
# 如果你足够细心，你应该能在验证集上获得约 0.365 (> 0.36) 的分类准确率。

# 注意：在超参数搜索期间你可能会看到运行时/溢出警告。
# 这可能是由极端值引起的，并不是 bug。

# results 是一个字典，将 (learning_rate, regularization_strength) 形式的元组
# 映射到 (training_accuracy, validation_accuracy) 形式的元组。
# 准确率只是正确分类的数据点比例。
results = {}
best_val = -1   # 到目前为止我们见过的最高验证准确率。
best_softmax = None # 达到最高验证率的 Softmax 对象。

################################################################################
# TODO:                                                                        #
# 编写代码，通过在验证集上调节来选择最佳超参数。对于超参数的每种组合，             #
# 在训练集上训练一个 Softmax，计算其在训练集和验证集上的准确率，                 #
# 并将这些数字存储在 results 字典中。此外，将最佳验证准确率存储在 best_val 中，    #
# 并将达到此准确率的 Softmax 对象存储在 best_softmax 中。                         #
#                                                                              #
# 提示：在开发验证代码时，你应该使用较小的 num_iters 值，以免分类器训练花费太多时间；#
# 一旦你确信验证代码工作正常，就应该使用较大的 num_iters 值重新运行代码。           #
################################################################################

# 作为参考提供。你可能想也可能不想更改这些超参数。
learning_rates = [1e-7, 1e-6]
regularization_strengths = [2.5e4, 1e4]

# 打印结果。
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e 训练准确率: %f 验证准确率: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('交叉验证期间达到的最佳验证准确率: %f' % best_val)

```

---

### 代码单元格 22：权重可视化

```python
# 可视化每个类别的学习权重。
# 取决于你选择的学习率和正则化强度，这些图像看起来可能会或可能不会很美观。
w = best_softmax.W[:-1,:] # 剥离出偏置项
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    
    # 将权重缩放到 0 到 255 之间
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

```

**内联问题 3**

描述你可视化的 Softmax 分类器权重看起来像什么，并简要解释为什么它们长成那样。

 *在此处填写*

**内联问题 4 - 正误判断**

假设总训练损失定义为所有训练示例上每数据点损失的总和。向训练集中添加一个新的数据点是有可能改变 Softmax 损失，但保持 SVM 损失不变的。





---

**我能为您做的下一步：**
您是否需要我为您推导 Softmax 的解析梯度公式，或者为您提供 `softmax_loss_vectorized` 函数的参考实现方案？