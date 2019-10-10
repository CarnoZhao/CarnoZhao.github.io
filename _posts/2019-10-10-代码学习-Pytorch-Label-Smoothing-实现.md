---
layout: post
title: 代码学习 Pytorch Label Smoothing 实现
subtitle: 看看别人怎么写代码的
date:       2019-10-10
author:     Carno
# header-img: 
catalog: true
tags:
- Python
- Pytorch
---

## 数学推导

Label Smoothing的定义如下：
$$
y_k^{LS} = y_k(1 - \alpha) + \frac{\alpha}{K} \\
y_k = \left\{
\begin{aligned}
0,\ &k \text{ is correct label}\\
1,\ &k\text{ is wrong label}
\end{aligned}
\right.
$$
在原来OneHot函数转化标签后，交叉熵的定义为：
$$
\begin{aligned}
H(y,\ p)&=\sum^{K}_{k = 1}-y_klog(p_k)\\
&=-log(p_t),\ \text{where }p_t\text{ is prob. of true}
\end{aligned}
$$
在Label Smoothing后：
$$
\begin{aligned}
H(y,\ p)&=\sum^{K}_{k = 1}-y_k^{LS}log(p_k)\\
&=-(1 - \alpha + \frac{\alpha}{K})log(p_k) - \sum^K_{i \neq k}\frac{\alpha}{K}log(p_i)
\end{aligned}
$$
所以被判断错误的概率也被计算在内，并且正确类的$logit$和错误类的$logit$之间相差一个常数，具体看[这里](https://zhuanlan.zhihu.com/p/73054583)

## Pytorch实现

源代码来自[这里](https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631)，相比于源代码，我添加了`weight`参数。

```python
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        if weight is None:
            self.weight = torch.ones(classes) / classes
        else:
            self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred * self.weight, dim=self.dim))
```

`classes`是类别的数量，`smoothing`是超参数$\alpha$，`dim`是矩阵求和的维数，一般不变，`weight`是对各个类别损失函数值的权重。

`pred`是经过网络计算后的直接输出值，并且没有经过softmax计算，在二维情况下是$n\times K$的矩阵，$n$是数据大小，一般也就是Batch Size，$K$是类别数。`target`是对应的$n\times 1$大小的标签矩阵，`target[i]`$\in[0, K-1]$。

假设有`pred`、`target`如下：

```python
pred = tensor([[ 0.1796, -0.0440,  0.4484],
               [-0.2746,  0.1361,  0.3606],
               [-0.1053,  0.0179,  0.4667],
               [-0.0686, -0.2447,  0.4115],
               [-0.0467,  0.3776,  0.0937]]) # n = 5, K = 3
target = tensor([0, 1, 2, 1, 0]) # K = 3
```

经过log_softmax：

```python
pred = pred.log_softmax(dim = -1)
## output ##
pred = tensor([[-1.1340, -1.3576, -0.8652],
               [-1.4805, -1.0699, -0.8453],
               [-1.3617, -1.2385, -0.7897],
               [-1.2397, -1.4158, -0.7597],
               [-1.3027, -0.8784, -1.1623]])
```

损失函数可以如下分解：
$$
H(y,\ p)=-1\times(\frac{\alpha}{K}+\beta)\times log(p)\\
\beta=\left\{\begin{aligned}1-\alpha,&\text{ true label}\\0,&\text{ wrong label}\end{aligned}\right.
$$
对于括号前一部分，创建一个相同大小的矩阵并赋值为$\alpha/K$（或者$\alpha/(K - 1)$？）：

```python
true_dist = torch.zeros_like(pred)
true_dist.fill_(smoothing / (classes - 1)) # classes = K = 3, smoothing = 0.01
## output ##
true_dist = tensor([[0.0050, 0.0050, 0.0050],
                    [0.0050, 0.0050, 0.0050],
                    [0.0050, 0.0050, 0.0050],
                    [0.0050, 0.0050, 0.0050],
                    [0.0050, 0.0050, 0.0050]])
```

再用[`torch.scatter_`函数](https://zhuanlan.zhihu.com/p/59346637)在正确标签概率的位置加上$1-\alpha$：

```python
true_dist.scatter_(1, target.data.unsqueeze(1), 1 - smoothing)
## output ##
target.data.unsqueeze(1) = 
	tensor([[0],
            [1],
            [2],
            [1],
            [0]])
true_dist = tensor([[0.9900, 0.5000, 0.5000], # true label is 0
                    [0.5000, 0.9900, 0.5000], # true label is 1
                    [0.5000, 0.5000, 0.9900], # true label is 2
                    [0.5000, 0.9900, 0.5000], # true label is 1
                    [0.9900, 0.5000, 0.5000]]) # true label is 0
```

最后，每一列乘以各自的`weight[k]`，返回平均损失。