没问题！我们将之前的**理论总结**与**实际代码**完美融合在一起。

这一份是**“代码 + 笔记”的终极合订本**。我把每一块核心功能的代码都拆解出来，配上详细的思路解析，最后给你一份**可以直接复制使用的完整 Class 代码**。

你可以把这一段直接放进你的笔记软件里，既有代码又有原理。

---

# 📚 k-NN 算法实战：代码与思路全解

## 1. 核心类结构 (Class Structure)

k-NN 的类结构非常简单。因为它是一种**“懒惰学习” (Lazy Learning)** 算法，核心在于“存数据”和“算距离”。

```python
class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练阶段：不做任何计算，只是把数据存起来。
        时间复杂度：O(1)
        """
        self.X_train = X
        self.y_train = y

```

---

## 2. 距离计算的三重境界 (Distance Computation)

这是作业中最核心的部分，目标是计算测试集  (形状 ) 和训练集  (形状 ) 之间的距离矩阵。

### 🐢 版本一：Two Loops (双重循环)

**思路**：最符合直觉的写法。像在一个二维表格里填空，一个格子一个格子地算。

* **缺点**：Python 的 `for` 循环非常慢，无法利用 CPU 的并行计算能力。

```python
def compute_distances_two_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    for i in range(num_test):
        for j in range(num_train):
            # 1. 核心公式：L2 距离 (Euclidean)
            # 务必转为 float 防止 uint8 溢出
            diff = X[i].astype(float) - self.X_train[j].astype(float)
            # 2. 累加平方差然后开根号
            dists[i, j] = np.sqrt(np.sum(diff**2))
            
    return dists

```

### 🚗 版本二：One Loop (单重循环)

**思路**：利用 NumPy 的**广播机制 (Broadcasting)**。

* **操作**：拿**一张**测试图  减去**所有**训练图 。
* **缺点**：虽然少了一层循环，但会产生巨大的临时矩阵 (5000 x 3072)，导致**内存吞吐瓶颈**，甚至比双重循环还慢。

```python
def compute_distances_one_loop(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    for i in range(num_test):
        # Broadcasting: (D,) - (N, D) -> (N, D)
        # 这一步会瞬间申请大量内存
        distances = X[i].astype(float) - self.X_train.astype(float)
        distances_square = np.square(distances)
        # axis=1 表示把每一行的维度加起来
        dists[i, :] = np.sqrt(np.sum(distances_square, axis=1))
        
    return dists

```

### 🚀 版本三：No Loops (完全向量化)

**思路**：利用数学公式展开，把减法变成矩阵乘法。这是**速度最快**的方法。

* **公式**：
* **核心技巧**：`np.dot` (矩阵乘法) 调用了底层的 BLAS 库，极度优化。

```python
def compute_distances_no_loops(self, X):
    # 1. 预处理：转为 float 类型，防止任何溢出风险
    X_float = X.astype(float)
    X_train_float = self.X_train.astype(float)

    # 2. 中间项：-2xy (矩阵乘法)
    # X (M, D) dot X_train.T (D, N) -> (M, N)
    dists = -2 * np.dot(X_float, X_train_float.T)
    
    # 3. 平方项：x^2 + y^2
    # axis=1 对行求和，得到每个样本的长度平方
    sq_train = np.sum(X_train_float**2, axis=1)
    sq_test = np.sum(X_float**2, axis=1).reshape(-1, 1) # reshape 变成列向量以便广播
    
    # 4. 合并并开根号
    dists = np.sqrt(dists + sq_train + sq_test)
    
    return dists

```

---

## 3. 预测与投票 (Prediction)

**思路**：有了距离矩阵后，我们需要找到最近的  个邻居并让它们投票。

* `np.argsort`: 获取排序后的索引（而不是排序后的值）。
* `np.bincount`: 统计票数。
* `np.argmax`: 找到票数最多的那一类。

```python
def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    
    for i in range(num_test):
        distance_i = dists[i] # 第 i 个测试点的所有距离
        
        # 1. 排序：找到距离最小的 k 个点的索引
        sorted_indices = np.argsort(distance_i)
        k_nearest_indices = sorted_indices[0:k]
        
        # 2. 查表：找到这些索引对应的真实标签
        closest_y = self.y_train[k_nearest_indices]
        
        # 3. 投票：统计哪个标签出现次数最多
        # np.bincount 统计每个整数出现的次数
        label_counts = np.bincount(closest_y)
        
        # 4. 决断：取次数最大的那个
        best_label = np.argmax(label_counts)
        y_pred[i] = best_label

    return y_pred

```

---

## 4. 📝 完整合并后的代码 (Copy & Paste)

这是你作业文件 `k_nearest_neighbor.py` 的最终形态，包含了所有修复和优化。

```python
from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange
import math

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Naive implementation using nested loops.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                # Using .astype(float) to prevent uint8 overflow
                diff = X[i].astype(float) - self.X_train[j].astype(float)
                dists[i, j] = np.sqrt(np.sum(diff**2))
        return dists

    def compute_distances_one_loop(self, X):
        """
        Implementation using partial vectorization (broadcasting).
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # Broadcasting X[i] across all X_train
            distances = X[i].astype(float) - self.X_train.astype(float)
            distances_square = np.square(distances)
            dists[i, :] = np.sqrt(np.sum(distances_square, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        """
        Fully vectorized implementation using matrix multiplication formula:
        (x-y)^2 = x^2 + y^2 - 2xy
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        # 1. Convert to float first for safety and precision
        X_float = X.astype(float)
        X_train_float = self.X_train.astype(float)

        # 2. Calculate the -2xy term using matrix multiplication
        # Shape: (num_test, num_train)
        dists = -2 * np.dot(X_float, X_train_float.T)
        
        # 3. Add x^2 and y^2 terms
        # sum(X_train**2) shape: (num_train,) -> broadcast to each row
        sq_train = np.sum(X_train_float**2, axis=1)
        # sum(X**2) shape: (num_test, 1) -> broadcast to each column
        sq_test = np.sum(X_float**2, axis=1).reshape(-1, 1)
        
        dists = np.sqrt(dists + sq_train + sq_test)
        
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances, predict labels using majority voting.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 1. Find indices of k nearest neighbors
            closest_y = []
            distance_i = dists[i]
            sorted_indices = np.argsort(distance_i)
            k_nearest_indices = sorted_indices[0:k]
            closest_y = self.y_train[k_nearest_indices]

            # 2. Majority Vote
            # bincount counts occurrences of each non-negative integer
            label_counts = np.bincount(closest_y)
            # argmax finds the index of the maximum count (the most frequent label)
            best_label = np.argmax(label_counts)
            
            y_pred[i] = best_label

        return y_pred

```