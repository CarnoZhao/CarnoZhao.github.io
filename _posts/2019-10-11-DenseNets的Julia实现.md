---
layout: post
title: DenseNets的Julia实现
subtitle: 神经网络模型学习（二）
date: 2019-10-11
author: Carno
header-img: img/post-paper-densenets.png
header-mask: 0.5
catalog: true
tags:
- Neural Network
- DenseNets
- Julia
---

## 介绍

在有CUDA加速的情况下，总感觉Julia (Flux)的速度还是要比Python (Pytorch)更快一点，但是Flux包里没有官方的DenseNets，所以从[这里](https://github.com/FluxML/Metalhead.jl/blob/master/src/densenet.jl)找了一个DenseNets打算自己改。

## DenseNets

之前有介绍过[DenseNets](https://carnozhao.github.io/2019/09/27/文献阅读-DenseNets/)的组成，这里直接放上一个例子的表格

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

在这里，每个Conv层都代表BatchNorm-ReLU-Conv层的叠加。

## 代码

#### Dense层

下面这样的结构叫做包含瓶颈层的Dense层
$$
\left[\begin{aligned}1\times1 \\ 3\times3\\\end{aligned}\right]
$$
瓶颈层$1\times1$的作用是将输入的通道数限制为特定数值，这里默认为$4k$，$k$是成长率。

卷积层$3\times3$再将$4k$的输入转化为$k$的输出。

所以这样一个结构的代码是：

```julia
struct Bottleneck
    layer
end

Bottleneck(in_planes, growth_rate) = Bottleneck(
    Chain(BatchNorm(in_planes, relu),
    Conv((1, 1), in_planes => 4 * growth_rate),
    BatchNorm(4 * growth_rate, relu),
    Conv((3, 3), 4 * growth_rate => growth_rate, pad = (1, 1))
    )
)

(b::Bottleneck)(x) = cat(3, b.layer(x), x)
```

`in_planes`是输入的通道数，`growth_rate`是成长率$k$。

所以给出`in_planes`和`growth_rate`就可以定义一个Dense层，Dense层作用在`x`上的输出是计算后的结果和原有数据的组合：`cat(3, b.layer(x), x)`。

#### 转化层

转化层的作用是在Dense块之间完成池化，降低数据大小。
$$
1\times1\ \text{conv};\ 2\times2\ \text{avg pool},\ s=2
$$
这样的结构代码是：

```julia
Transition(chs::Pair{<:Int, <:Int}) = Chain(
    BatchNorm(chs[1], relu),
    Conv((1, 1), chs),
    x -> meanpool(x, (2, 2)))
```

`chs`是转化层的通道设置，传入时默认是`k => 0.5 * k`，0.5代表压缩率。

#### Dense块

下面这样的结构是由6层Dense层组成的Dense块：
$$
\left[\begin{aligned}1\times1 \\ 3\times3\\\end{aligned}\right]\times6
$$
代码是：

```julia
function _make_dense_layers(block, in_planes, growth_rate, nblock)
    local layers = []
    for i in 1:nblock
        push!(layers, block(in_planes, growth_rate))
        in_planes += growth_rate
    end
    Chain(layers...)
end
```

`nblock`是Dense层的数量，例如这里的6。`block`参数是刚才定义的`Bottleneck`。

因为每过一个`Bottleneck`，将会在原有输入上叠加$k$个新通道，作为下一次的输入

所以有`in_planes += growth_rate`。

#### 综合

最后将各个组分和一开始和最后的独立层综合起来：

```julia
function _DenseNet(nblocks; in_chs = 1, block = Bottleneck, growth_rate = 12, reduction = 0.5,num_classes = 1)
    num_planes = 2 * growth_rate
    layers = []
    push!(layers, Conv((7, 7), in_chs => num_planes, stride = (2, 2), pad = (3, 3)))
    push!(layers, BatchNorm(num_planes, relu))
    push!(layers, x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)))
    for i in 1:3
        push!(layers, _make_dense_layers(block, num_planes, growth_rate, nblocks[i]))
        num_planes += nblocks[i] * growth_rate
        out_planes = Int(floor(num_planes * reduction))
        push!(layers, Transition(num_planes => out_planes))
        num_planes = out_planes
    end

    push!(layers, _make_dense_layers(block, num_planes, growth_rate, nblocks[4]))
    num_planes += nblocks[4] * growth_rate
    push!(layers, BatchNorm(num_planes, relu))

    Chain(
        layers...,
        x -> meanpool(x, (7, 7)),
        x -> reshape(x, :, size(x, 4)),
        Dense(num_planes,num_classes))
end
```



一开始有$7\times 7$的Conv-BatchNorm-Pool层，输出通道数为$2k$，假设有四个Dense块，循环叠加Dense块和Transition层，最后加上$7\times 7$的Pool，并转换为全连接。下面的代码可以获得四种DenseNets模板。

```julia
function get_densenet_model(depth)
    if depth == 121
        _DenseNet([6, 12, 24, 16]) |> gpu
    elseif depth == 169
        _DenseNet([6, 12, 32, 32]) |> gpu
    elseif depth == 201
        _DenseNet([6, 12, 48, 32]) |> gpu
    elseif depth == 264
        _DenseNet([6, 12, 64, 48]) |> gpu
    else
        error("No such model is available")
    end
end
```

