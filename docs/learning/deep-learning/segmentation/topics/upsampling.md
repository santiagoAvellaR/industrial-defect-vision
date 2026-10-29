# Upsampling in a segmentation decoder

## What kind of thing is upsampling?

Upsampling is an **operation that increases spatial resolution**.

It is not a complete network and it is not the same thing as a decoder.

A decoder is a subnetwork made from several operations and layers. Upsampling is one operation the decoder can use.

Example:

```text
[B, 128, 80, 28]
        ↓ upsample ×2
[B, 128,160, 56]
```

The goal is to make the spatial grid larger again after the encoder reduced it.

## Why segmentation needs upsampling

An encoder often performs downsampling:

```text
640×224
↓
320×112
↓
160×56
↓
80×28
```

But a pixel-aligned segmentation mask must eventually return to the target resolution:

```text
640×224
```

So the decoder needs a mechanism for increasing `H` and `W`.

## Important limitation

Upsampling increases the number of spatial positions, but it does **not automatically recover information that was lost during downsampling**.

If a small boundary disappeared in the encoder, simply making the tensor larger does not tell the network exactly where that boundary was.

This is one reason U-Net also uses skip connections from higher-resolution encoder stages.

## Common ways to upsample

### 1. Interpolation followed by convolution

A simple approach is:

```text
low-resolution features
↓ interpolation
larger feature grid
↓ convolution
refined features
```

For example in PyTorch, interpolation can double spatial dimensions and ordinary convolutions can then learn how to transform the enlarged features.

Interpolation itself has no learned parameters.

### 2. Transposed convolution

A transposed convolution is a learnable operation that can increase spatial resolution.

It is often called `ConvTranspose2d` in PyTorch.

Unlike simple interpolation, its upsampling transformation contains trainable weights.

It is not literally the mathematical inverse of a normal convolution, despite the common informal phrase "deconvolution".

## Which approach will this project start with?

For the first educational segmentation baseline, the implementation should prefer the simplest option that keeps tensor shapes explicit and understandable.

The exact implementation choice should be made when the first U-Net is written, not assumed in advance.

The important concept is independent of the implementation:

> The encoder reduces spatial resolution to gain context; the decoder must increase spatial resolution so the model can return to pixel-level prediction.

## How upsampling and skip connections work together

Suppose the decoder currently has:

```text
[B,128,80,28]
```

After upsampling:

```text
[B,64,160,56]
```

and the corresponding encoder stage saved a feature map shaped:

```text
[B,64,160,56]
```

Now both have the same `H` and `W`, so U-Net can combine them, commonly by concatenating channels:

```text
decoder [B,64,160,56]
encoder [B,64,160,56]
            ↓ concatenate
         [B,128,160,56]
```

Following convolutions learn how to combine the deep decoder context with the higher-resolution encoder detail.

## Mental model

- downsampling: make the grid smaller and representation more contextual;
- upsampling: make the grid larger again;
- skip connection: bring back higher-resolution information from before stronger compression;
- convolution after combining features: learn how to use that information.

## Questions to be able to answer

1. Is upsampling a layer, a subnetwork or a complete architecture?
2. Why does a segmentation decoder need to increase `H` and `W`?
3. Why does making a tensor larger not automatically restore lost details?
4. What is the difference between interpolation and a transposed convolution?
5. Why do U-Net skips connect stages with matching spatial resolution?
6. What usually happens after decoder and encoder features are combined?
