# Pooling and spatial compression

Pooling reduces spatial information in a feature map while keeping a compact summary of local responses.

Our baseline uses:

```python
nn.MaxPool2d(2)
```

## MaxPool

A `2×2` MaxPool keeps the largest activation in each local region.

Example:

```text
input feature map

1  4 | 2  3
5  2 | 0  1
-----+-----
2  8 | 1  4
3  2 | 6  1

MaxPool 2×2

5  3
8  6
```

For one local pooling region $R$ with activation values $a_{i,j}$, MaxPool computes:

```math
y = \max_{(i,j)\in R} a_{i,j}
```

So:

```text
4×4 spatial grid
      ↓
2×2 spatial grid
```

With the default stride of 2:

```text
640 × 224 → 320 × 112
320 × 112 → 160 × 56
160 × 56  → 80 × 28
```

MaxPool changes `H` and `W`, but not channel count.

## What does “summarize spatially” mean?

Inside each local region, MaxPool stops preserving the exact position of every activation and keeps only the strongest response.

Example:

```text
0.1  0.2
0.3  4.8

→ 4.8
```

We lose:

```text
exactly which of the four positions contained the strong response
```

but preserve approximately:

```text
there was a strong response somewhere in this region
```

That is the spatial summary.

For classification, exact pixel location is often less important than knowing that a useful pattern is present. For segmentation, this lost spatial precision matters much more.

MaxPool also reduces computation in later layers and provides some local robustness to small shifts, although it does not guarantee translation invariance.

## MaxPool does not create more features

This distinction is important:

```text
Conv 16→32
→ creates 32 output feature maps

MaxPool
→ keeps 32 feature maps but makes each one spatially smaller
```

For example:

```text
[32, 320, 112]
      ↓ MaxPool
[32, 160, 56]
```

## Global Average Pooling

After the last MaxPool, each image is represented by:

```text
[64, 80, 28]
```

That means 64 feature maps, each with an `80×28` spatial grid.

We use:

```python
features.mean(dim=(2, 3))
```

which averages over `H` and `W`.

For feature map $c$, global average pooling computes:

```math
g_c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} a_{c,i,j}
```

where $a_{c,i,j}$ is the activation of channel $c$ at spatial position $(i,j)$, and $g_c$ is the single value kept for that feature map.

### Tiny numerical example

Suppose one feature map is:

```text
1  3
2  6
```

Then:

```math
\mathrm{GAP} = \frac{1+3+2+6}{4} = 3
```

That entire feature map becomes one number.

With 64 feature maps:

```text
feature map 1  → one mean value
feature map 2  → one mean value
...
feature map 64 → one mean value
```

So:

```text
[64, 80, 28]
      ↓ GAP
[64]
```

and with a batch:

```text
[B, 64, 80, 28]
        ↓
[B, 64]
```

## Does GAP discard the image pixels and keep the features?

By the time we reach GAP, the original RGB pixels are no longer the current representation.

The network has already transformed them repeatedly:

```text
RGB pixels
   ↓ Conv/ReLU/Pool
feature maps
   ↓ Conv/ReLU/Pool
richer feature maps
   ↓ Conv/ReLU/Pool
64 final feature maps
```

GAP operates on those final feature maps.

Useful mental model:

```text
before GAP:
where and how strongly is each learned feature active?

          ↓ GAP

after GAP:
how strongly is each learned feature present overall?
```

It intentionally discards the remaining exact spatial arrangement.

## MaxPool vs Global Average Pooling

```text
MaxPool
= local maximum reduction
= still leaves a spatial grid

Global Average Pooling
= average over the entire remaining spatial map
= removes H and W completely
```

## Flatten vs squeeze

These operations are not the same.

### Flatten

Flatten combines dimensions into one.

```text
[B, 64, 80, 28]
        ↓ flatten from dim 1
[B, 143360]
```

No values are removed; their arrangement is changed into one long vector.

The number of values per image would be:

```math
64\times 80\times 28 = 143{,}360
```

### Squeeze

`squeeze` removes a dimension only if its size is `1`.

Our classifier produces:

```text
[B, 1]
```

After:

```python
.squeeze(1)
```

we get:

```text
[B]
```

Example:

```text
[[ 2.1],
 [-0.7],
 [ 1.4]]

shape [3,1]

↓ squeeze(1)

[2.1, -0.7, 1.4]

shape [3]
```

No prediction is discarded. Only a redundant size-1 dimension is removed.

## Why use GAP instead of flattening here?

Flattening preserves all remaining spatial values, while GAP deliberately collapses each feature map to one average response.

For our final tensor, flattening would create $143{,}360$ values per image before the classifier, whereas GAP gives only $64$ values. This keeps the classification head small and reduces parameter count and overfitting risk.

## Questions to be able to answer

1. What does `MaxPool2d(2)` do?
2. What does it mean to summarize activations spatially?
3. What information does MaxPool discard?
4. Does MaxPool change the number of channels?
5. Why is spatial compression acceptable for classification but more delicate for segmentation?
6. What does global average pooling calculate mathematically?
7. What information does GAP discard?
8. What is the difference between MaxPool and GAP?
9. What is the difference between `flatten` and `squeeze`?
10. Why do we use GAP rather than flattening `[64,80,28]`?
