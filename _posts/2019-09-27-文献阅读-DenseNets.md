---
layout: post
title: 文献阅读 DenseNets
subtitle: 神经网络模型学习（一）
date: 2019-09-27
author: Carno
header-img: img/post-paper-densenets.png
header-mask: 0.5
catalog: true
tags:
    - Neural Network
    - DenseNets
---

# 介绍

神经网络已经发展很多年了，硬件的支持允许训练越来越深的网络。但是随着层数的加深，输入和回溯的梯度会逐渐减弱，所以出现了ResNets和Highway Networks把一层的信号直接不变的传到下一层。

和ResNets不同的是，DenseNet并不是简单的对传入下一层的信息进行求和，而是将它们拼接在一起。所以在一个$L$层的Dense块中，第$l$行会和后$L-l$行链接，最终有$\frac{(L + 1)L}{2}$个链接。

DenseNet的优点之一是它减少了参数的数目，（还没看懂）。

另一个优点是优化了传入信息和梯度的传递，更易训练。因为每层都直接和输出相联系，提供内在的监督（implicit deep supervision）。

# DenseNets

$x_0$：单个图像

$x_l$：第$l$层的输出

$L$：层数

$H_l(x)$：第$l$层的非线性转化函数，可以是复合函数

### Traditional

$$
x_l = H_l(x_{l - 1})
$$

### ResNets

$$
x_l = H_l(x_{l - 1}) + x_{l - 1}
$$

 ### Dense connectivity

$$
x_l = H_l([x_0, x_1, ..., x_{l - 1}])
$$

$[x_0, x_1, ...,x_{l - 1}]$是前$l-1$层的拼接结果

#### 复合函数

$H_l(x)$可以被定义为三个连续操作的复合，分别是batch Normalization（BN），ReLU，$3\times3$ convolution（Conv）。

#### 转换层

上述$L$层结构被称为一个Dense块，Dense块之间之间有转换层，由卷积和池化层组成。在文章中，转换层由一层BN层，一层$1\times1$ Conv层，一层$2\times2$均值Pool层组成。

#### 成长率$k$

如果每个$H_l$函数生成$k$个通道，那么第$l$层的输入的通道数将会是$k_0 + k\times(l - 1)$，$k_0$是最开始输入层的通道数。

#### 瓶颈层

虽然Dense块中每一层的输出通道都是$k$，但是输入通道却在逐渐增加，所以在$3\times3$ Conv层之前引入$1\times1$ Conv的瓶颈层，例如限制瓶颈层后的输出为$4k$，提高计算效率。

#### 压缩

如果一个Dense块输出$m$个通道，那么转换层将会生成$\lfloor\theta m\rfloor$个通道。

### 一个例子

Input Size：$224\times224$

Input Channel：$3$ (RGB)

Conv: BN-ReLU-Conv

| Layers         | Layer Detail                                                 | Output Size    | Output Channel |
| -------------- | ------------------------------------------------------------ | -------------- | :------------: |
| Convolution    | $7\times7,\ s = 2,\ p = 3$                                   | $112\times112$ |      $2k$      |
| Pooling        | $3\times3,\ s = 2,\ p = 1$                                   | $56\times56$   |      $2k$      |
| Dense          | $\left[\begin{aligned}1\times1 \\ 3\times3\\\end{aligned}\right]\times6$ | $56\times56$   |      $k$       |
| Transition     | $1\times1\ conv;\ 2\times2\ avg\ pool,\ s=2$                 | $28\times28$   |     $k/2$      |
| Dense          | $\left[\begin{aligned}1\times1 \\ 3\times3\\\end{aligned}\right]\times12$ | $28\times28$   |      $k$       |
| Transition     | $1\times1\ conv;\ 2\times2\ avg\ pool,\ s=2$                 | $14\times14$   |     $k/2$      |
| Dense          | $\left[\begin{aligned}1\times 1 \\ 3\times 3\\\end{aligned}\right]\times 24$ | $14\times14$   |      $k$       |
| Transition     | $1\times1\ conv;\ 2\times2\ avg\ pool,\ s=2$                 | $7\times7$     |     $k/2$      |
| Dense          | $\left[\begin{aligned}1\times1 \\ 3\times3\\\end{aligned}\right]\times16$ | $7\times7$     |      $k$       |
| Classification | $7\times7\ avg\ pool$                                        | $1\times 1$    |      $k$       |