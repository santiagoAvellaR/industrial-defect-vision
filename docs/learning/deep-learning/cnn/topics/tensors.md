# Tensors

A tensor is a multidimensional array of numbers. In PyTorch, images, weights, activations, labels, logits and gradients are all represented as tensors.

## From scalar to tensor

```text
scalar:  7
vector:  [1, 4, 8]
matrix:  [[1, 2],
          [3, 4]]
image tensor: [C, H, W]
batch tensor: [B, C, H, W]
```

Our image tensor has shape:

```text
[C, H, W] = [3, 640, 224]
```

A batch of four images has:

```text
[B, C, H, W] = [4, 3, 640, 224]
```

where:

```text
B = batch size
C = channels
H = height
W = width
```

## What is a weight tensor?

A weight tensor is the tensor in which PyTorch stores the learnable parameters of a layer.

For a convolution, all filters are stored together. The weight tensor is not a different concept from the filters; it is the data structure that contains their numeric weights.

For:

```python
nn.Conv2d(3, 16, kernel_size=3)
```

the weight tensor has shape:

```text
[16, 3, 3, 3]
 ↑   ↑  ↑  ↑
 |   |  |  └─ kernel width
 |   |  └──── kernel height
 |   └─────── input channels
 └─────────── output filters / channels
```

So:

```text
one filter       = [3, 3, 3]
all 16 filters   = [16, 3, 3, 3]
```

The weights are initialized before training and then learned through optimization.

## Weight tensor vs activation tensor

This distinction is important:

```text
weight tensor
= parameters learned by the model

activation tensor
= values produced when an input passes through the model
```

For example, after the first convolution:

```text
input image tensor
[B, 3, 640, 224]

      ↓ Conv2d

activation tensor
[B, 16, 640, 224]
```

The first tensor contains data. The convolution has a separate weight tensor. The output activation tensor contains the responses produced by applying those weights to the input.

## Mental model

```text
input tensor
     +
weight tensor
     ↓
operation
     ↓
activation tensor
```

## Questions to be able to answer

1. What is a tensor?
2. What do `B`, `C`, `H` and `W` mean?
3. What is the difference between a weight tensor and an activation tensor?
4. What does `[16, 3, 3, 3]` represent in our first convolution?
5. Why is one filter `[3,3,3]` for an RGB input?
