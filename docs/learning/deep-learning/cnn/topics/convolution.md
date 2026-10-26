# Convolution

A Convolutional Neural Network (CNN) is called **convolutional** because it repeatedly applies learned local filters across spatial positions of an input.

Our first layer is:

```python
nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

It means:

- 3 input channels: RGB;
- 16 output channels;
- kernel spatial size: `3 × 3`;
- default stride: `1`;
- padding: `1` pixel.

## Patch, kernel and filter

These are different concepts.

```text
PATCH
= values from the input currently being read

KERNEL
= learnable weights applied to one spatial patch

FILTER
= complete set of kernels spanning all input channels
  and producing one output feature map
```

For one channel:

```text
image patch             kernel weights

2  5  1                 0.2 -0.1  0.3
4  7  3        ×        0.5  0.1 -0.2
1  0  6                -0.3  0.4  0.2

                 ↓
       weighted sum + bias
                 ↓
          one output value
```

For one output filter $k$ at spatial position $(i,j)$, the convolution can be written conceptually as:

```math
z_{k,i,j}
=
b_k
+
\sum_{c=1}^{C_{\mathrm{in}}}
\sum_{u=1}^{K_h}
\sum_{v=1}^{K_w}
W_{k,c,u,v}\,X_{c,i+u,j+v}
```

where:

- $X$ is the input tensor;
- $W$ contains the learned filter weights;
- $b_k$ is the bias of output filter $k$;
- $c$ indexes input channels;
- $(u,v)$ indexes positions inside the kernel;
- $z_{k,i,j}$ is one output value before the activation function.

The key idea is simpler than the notation: multiply the local patch by the learned weights, sum the contributions from all input channels, add the bias, and obtain one number.

The kernel weights stay the same while the patch changes as the operation moves across the image.

For RGB, one complete filter has shape:

```text
[3, 3, 3]
 ↑  ↑  ↑
 C  H  W
```

It contains one `3×3` kernel for R, one for G and one for B.

Terminology is not perfectly consistent across books and libraries; some sources use *kernel* and *filter* almost interchangeably. The convention above is the one used in these study notes because it makes PyTorch weight shapes easier to reason about.

## Stride

Stride is the spatial step between consecutive kernel positions.

```text
stride = 1  → move one pixel at a time
stride = 2  → move two pixels at a time
```

A useful mental model is:

```text
stride ≈ spatial delta between consecutive positions
```

A larger stride samples fewer positions and usually reduces output spatial resolution.

Ignoring dilation, one spatial output dimension follows:

```math
N_{\mathrm{out}}
=
\left\lfloor
\frac{N_{\mathrm{in}} + 2P - K}{S}
\right\rfloor + 1
```

where $N_{\mathrm{in}}$ is the input size, $K$ the kernel size, $P$ the padding and $S$ the stride.

For our convolutions, $K=3$, $P=1$ and $S=1$, so $N_{\mathrm{out}}=N_{\mathrm{in}}$: the convolution preserves spatial size.

## One filter produces one feature map

Conceptually:

```text
RGB image
   │
   ├── filter 1  → feature map 1
   ├── filter 2  → feature map 2
   ├── ...
   └── filter 16 → feature map 16
```

Every spatial position produces one value. The collection of those values forms one feature map.

Therefore:

```text
one learned filter
        ↓
one feature map
```

Our first convolution uses 16 filters, so it produces 16 output channels / feature maps.

## Why 16 channels?

`16` is an architectural choice, not something determined by RGB.

We could have chosen 8, 32 or another value.

More filters mean more representational capacity, but also:

- more parameters;
- more memory;
- more computation;
- potentially more overfitting.

For a first baseline, `16 → 32 → 64` is small, common and easy to reason about.

## What is learned?

The filter values are trainable parameters.

At initialization they do not already mean “edge”, “scratch” or “defect”. During training, optimization changes the weights so that the network becomes better at classification.

Early filters often respond to relatively simple local patterns such as edges, contrast changes, colors or textures. Deeper layers combine earlier representations and become harder to describe with one human-readable label.

We can inspect filters and feature maps, but there is no guaranteed mapping such as:

```text
filter 7 = scratch detector
filter 12 = defect detector
```

A concept such as `defect` is generally distributed across many activations and layers.

## Channels, spatial size and kernel size

For:

```text
[B, C, H, W]
```

we have:

```text
C     = number of channels / feature maps
H × W = spatial size / spatial resolution
```

Kernel size is separate.

Example:

```text
channels                 = 16
feature-map spatial size = 640 × 224
kernel size              = 3 × 3
```

So `3×3` is not the size of the feature map.

## Why does our convolution preserve spatial size?

Our convolution uses:

```text
kernel = 3×3
stride = 1
padding = 1
```

As shown by the output-size equation above, this combination preserves both spatial dimensions:

```text
640 × 224
   ↓ Conv
640 × 224
```

## Why do channels increase while spatial resolution decreases?

These changes come from different operations:

```text
Conv
→ can increase number of feature maps

MaxPool
→ reduces spatial resolution
```

Our baseline does:

```text
[3, 640, 224]
    ↓ Conv 3→16
[16, 640, 224]
    ↓ MaxPool
[16, 320, 112]
    ↓ Conv 16→32
[32, 320, 112]
    ↓ MaxPool
[32, 160, 56]
    ↓ Conv 32→64
[64, 160, 56]
    ↓ MaxPool
[64, 80, 28]
```

It is more precise to say that we lose **spatial detail/resolution**, not necessarily “quality”. For classification, a coarser but richer learned representation can be more useful than preserving exact pixel location.

For segmentation, spatial detail matters much more because the model must also answer **where** the defect is.

## Interactive demo

`docs/learning/visuals/convolution-kernel-demo.html`

## Questions to be able to answer

1. Why is a CNN called convolutional?
2. What is the difference between a patch, a kernel and a filter?
3. What does `Conv2d(3, 16, kernel_size=3, padding=1)` mean?
4. What does the convolution equation compute at one spatial position?
5. What is stride?
6. How do kernel size, padding and stride determine the output spatial size?
7. Why does one filter produce one feature map?
8. Why do we get 16 feature maps?
9. Can we always assign one human meaning to one learned filter?
10. What is the difference between channels, spatial size and kernel size?
11. Why does our convolution keep `640×224`?
12. Which operation actually reduces spatial size in our baseline?
