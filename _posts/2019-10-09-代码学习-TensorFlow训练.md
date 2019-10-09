---
layout: post
title: 代码学习 TensorFlow训练
subtitle: 看看别人怎么写代码的
date:       2019-10-09
author:     Carno
header-img: 
catalog: true
tags:
- Python
- Neural Network
- DenseNets
---



### 标签分布不平均

由于Positive/Negative标签的分布不均衡，约5:1。所以在训练的时候，每个Batch导入固定数目的Positive和Negative，例如$\frac{2}{3}$的Positive和$\frac{1}{3}$的Negative，降低标签的不均衡分布。



### 平滑标签 Label Smoothing

$$
y_{k}^{LS} = y_k(1 - \alpha) + \frac{\alpha}{K}
$$

其中$\alpha$是超参数（0.01），K是类别数（2）

平滑标签是为了将不连续的OneHot函数变成更加连续的形式，具体可以看[这里](https://zhuanlan.zhihu.com/p/73054583)。

### 评估

每10个epoch开始一次评估，分别导入training、validation、test数据集，计算准确率。



