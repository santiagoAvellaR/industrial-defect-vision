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

## PyTorch is not based on pandas

PyTorch and pandas solve different kinds of problems.

`pandas` is mainly designed for labeled tabular data such as rows and columns in a `DataFrame`.

PyTorch is built around multidimensional `Tensor` objects and numerical kernels designed for large array operations, automatic differentiation and accelerator hardware such as GPUs.

A useful comparison is:

```text
pandas DataFrame
→ tables with named rows/columns
→ data analysis and manipulation

PyTorch Tensor
→ dense multidimensional numerical arrays
→ neural-network computation and gradients
```

PyTorch does not process a convolution by writing a Python loop over every pixel. Operations such as convolution, matrix multiplication, elementwise activations and reductions are implemented by optimized low-level kernels. Those kernels can process many tensor elements efficiently and, depending on the operation and device, in parallel.

For example:

```python
output = convolution(images)
```

looks like one Python statement, but underneath it represents a large tensor operation over batches, channels and spatial positions.

On CPU, PyTorch uses optimized native code and can use multiple CPU threads for supported operations. On GPU, the same tensor abstraction lets large numbers of arithmetic operations be executed by GPU kernels in parallel.

The important mental model is:

> We describe an operation on whole tensors. PyTorch chooses efficient low-level implementations rather than asking us to write a Python loop for each scalar.

This is often called **vectorized/tensor computation**.

It does not mean that every scalar operation literally runs at the exact same instant. `Parallel` here means that the backend can exploit hardware parallelism and optimized batched kernels instead of executing a slow Python loop element by element.

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
6. Why is PyTorch not based on pandas?
7. What does vectorized/tensor computation mean?
8. Why can tensor operations be much faster than Python loops over individual pixels?
