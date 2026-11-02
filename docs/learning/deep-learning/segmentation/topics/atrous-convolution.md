# Atrous / dilated convolution

## What it is

**Atrous convolution** and **dilated convolution** are two names for the same idea.

A normal convolution samples neighboring pixels continuously. A dilated convolution inserts gaps between the sampled positions of the kernel.

For a `3×3` kernel:

### Dilation = 1 — ordinary convolution

```text
x x x
x x x
x x x
```

### Dilation = 2

```text
x . x . x
. . . . .
x . x . x
. . . . .
x . x . x
```

### Dilation = 3

```text
x . . x . . x
. . . . . . .
. . . . . . .
x . . x . . x
. . . . . . .
. . . . . . .
x . . x . . x
```

The kernel still has only nine learned positions, but those positions cover a larger spatial region.

## Visual reference

![Dilated convolution animation](https://commons.wikimedia.org/wiki/Special:Redirect/file/Convolution_arithmetic_-_Dilation.gif)

*Source: [Wikimedia Commons — Convolution arithmetic: Dilation](https://commons.wikimedia.org/wiki/File:Convolution_arithmetic_-_Dilation.gif). The animation makes the spacing between sampled input positions explicit.*

A static comparison from the DeepLab literature also shows how increasing the atrous rate enlarges the sampled field of view without enlarging the learned `3×3` kernel itself.

---

## Effective kernel size

For kernel size `k` and dilation `d`, the effective receptive span is:

```math
k_{effective} = 1 + (k - 1)d
```

For a `3×3` kernel:

| Dilation | Learned kernel weights | Effective spatial span |
|---:|---:|---:|
| 1 | 3×3 | 3×3 |
| 2 | 3×3 | 5×5 |
| 3 | 3×3 | 7×7 |
| 6 | 3×3 | 13×13 |

The important point is:

> the receptive field grows without increasing the number of kernel parameters.

A `3×3` convolution with dilation 6 still learns a `3×3` set of weights; it simply samples them farther apart.

---

## Why segmentation needs this idea

Semantic segmentation needs two things at the same time:

1. enough **context** to understand what a region belongs to;
2. enough **spatial resolution** to localize boundaries and small defects.

Repeated pooling or strided convolution increases context but aggressively shrinks the feature map.

```text
large context
        ↑
more downsampling
        ↓
less spatial detail
```

Dilated convolutions provide another option:

```text
larger receptive field
without proportional extra downsampling
```

This is one reason they are important in DeepLab.

---

## Standard convolution versus dilated convolution

Both produce learned feature maps. The difference is **where the kernel samples its input**.

With the same `3×3` learned kernel:

```text
standard conv, dilation=1
→ samples adjacent positions

atrous conv, dilation>1
→ samples positions with gaps
→ wider effective field of view
```

The original DeepLabV3 paper uses exactly this idea to preserve denser feature maps while increasing context.

Paper reference: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587).

---

## Receptive field intuition

Imagine trying to decide whether a tiny bright region is:

- a real defect;
- harmless surface texture;
- part of a longer scratch.

Looking only at immediate neighbors may be insufficient.

A larger receptive field gives the model more surrounding context before deciding how to represent that location.

This matters especially for industrial defects whose local appearance can resemble normal texture unless a wider region is considered.

---

## Does dilation make the convolution free?

No.

It does not increase the kernel parameter count for the same input/output channels and kernel size, but the convolution still has computational cost and memory cost for its feature maps.

Also, larger dilation does not guarantee better features.

Very sparse sampling can miss fine local structure.

---

## Gridding artifact

A possible limitation of large dilation is that the kernel samples a sparse regular grid.

If several layers use poorly chosen dilation patterns, parts of the local neighborhood may be repeatedly skipped. This can produce a **gridding** effect.

One motivation for combining several dilation rates is to collect context at multiple sampling scales rather than relying on one sparse pattern.

That leads directly to ASPP.

---

## Atrous versus pooling

They are not interchangeable.

**Pooling / stride:**

- reduces spatial dimensions;
- lowers computation in later layers;
- increases effective receptive field;
- can discard localization detail.

**Atrous convolution:**

- can expand receptive field while keeping feature-map resolution;
- keeps more spatial positions;
- can cost more compute/memory than aggressively downsampling.

A segmentation architecture chooses a balance between them.

---

## Where it appears in our project

DeepLabV3 uses atrous convolutions in its multiscale context mechanism.

Our current model:

```text
DeepLabV3-ResNet50
```

uses a ResNet50 backbone followed by ASPP, which processes deep features using parallel atrous rates.

---

## Interview explanation

> Atrous or dilated convolution is a convolution where the kernel samples input positions with gaps. It increases the effective receptive field without increasing the kernel size or number of kernel weights. In semantic segmentation this is useful because the model can capture wider context without relying only on additional downsampling, which would reduce localization detail.

## Questions to be able to answer

1. Are atrous convolution and dilated convolution different operations?
2. What is the effective size of a `3×3` kernel with dilation 2?
3. Does increasing dilation increase the number of kernel parameters?
4. Why is dilation useful for semantic segmentation?
5. What trade-off does it have compared with downsampling?
6. What is a gridding artifact?
