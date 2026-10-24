# CNN visual guide — baseline classifier

This is the visual companion to `cnn-basics.md`.

Use it when the spatial intuition is more important than the formal definition.

## 1. A convolution kernel is a sliding patch

A `3×3` kernel does not look at the whole image at once. It reads one local `3×3` patch, computes a weighted sum, then moves to the next position.

```text
Image

┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │
├───┼───┼───┼───┼───┤
│ 6 │ 7 │ 8 │ 9 │10 │
├───┼───┼───┼───┼───┤
│11 │12 │13 │14 │15 │
├───┼───┼───┼───┼───┤
│16 │17 │18 │19 │20 │
├───┼───┼───┼───┼───┤
│21 │22 │23 │24 │25 │
└───┴───┴───┴───┴───┘

First 3×3 patch

┏━━━┳━━━┳━━━┓
┃ 1 ┃ 2 ┃ 3 ┃
┣━━━╋━━━╋━━━┫
┃ 6 ┃ 7 ┃ 8 ┃
┣━━━╋━━━╋━━━┫
┃11 ┃12 ┃13 ┃
┗━━━┻━━━┻━━━┛

Then it moves one position to the right when stride = 1.
```

Open the interactive version:

`docs/learning/visuals/convolution-kernel-demo.html`

The demo uses one channel and fixed example weights only to make the mechanism visible. In our real first layer, a filter spans all three RGB channels and its weights are learned.

## 2. What one filter produces

One filter is reused across all spatial positions.

```mermaid
flowchart LR
    I[Input image] --> P1[3×3 patch at position 1]
    I --> P2[3×3 patch at position 2]
    I --> P3[3×3 patch at position ...]
    P1 --> O1[one output value]
    P2 --> O2[one output value]
    P3 --> O3[one output value]
    O1 --> F[feature map]
    O2 --> F
    O3 --> F
```

So:

```text
one learned filter
        ↓
one feature map
```

Our first convolution has 16 filters, therefore it produces 16 feature maps.

## 3. Channels and spatial size are different axes

The first layer changes:

```text
[B, 3, 640, 224]
        ↓ Conv 3→16
[B,16, 640, 224]
```

The spatial size stays the same because we use `kernel_size=3`, `stride=1`, `padding=1`.

The number of channels changes because we ask the layer to learn 16 different filters.

```mermaid
flowchart LR
    RGB[3 input channels] --> C1[filter 1]
    RGB --> C2[filter 2]
    RGB --> C3[...]
    RGB --> C16[filter 16]
    C1 --> F1[feature map 1]
    C2 --> F2[feature map 2]
    C3 --> F3[...]
    C16 --> F16[feature map 16]
```

## 4. ReLU changes values, not shape

```text
before ReLU     after ReLU

 -2.0              0.0
 -0.3              0.0
  0.0              0.0
  1.5              1.5
  4.2              4.2
```

Its shape is:

```text
[B, C, H, W] → [B, C, H, W]
```

The important role is not dimensionality reduction. It is introducing non-linearity between learned transformations.

## 5. MaxPool is a local reduction

For a `2×2` region:

```text
┌───┬───┐
│ 1 │ 5 │
├───┼───┤
│ 2 │ 3 │
└───┴───┘

MaxPool → 5
```

Applied across the whole feature map with stride 2:

```text
H × W
 ↓
H/2 × W/2
```

Channels do not change.

## 6. Our complete shape flow

```mermaid
flowchart TD
    A[Input<br/>B × 3 × 640 × 224]
    B[Conv 3→16<br/>B × 16 × 640 × 224]
    C[ReLU]
    D[MaxPool<br/>B × 16 × 320 × 112]
    E[Conv 16→32<br/>B × 32 × 320 × 112]
    F[ReLU]
    G[MaxPool<br/>B × 32 × 160 × 56]
    H[Conv 32→64<br/>B × 64 × 160 × 56]
    I[ReLU]
    J[MaxPool<br/>B × 64 × 80 × 28]
    K[Global average pooling<br/>B × 64]
    L[Linear 64→1<br/>B × 1]
    M[squeeze<br/>B]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K --> L --> M
```

The broad trend is:

```text
spatial resolution: 640×224 → 320×112 → 160×56 → 80×28
channels:                    3 → 16 → 32 → 64
```

## 7. Classification at the end

After global average pooling, every image is represented by 64 numbers:

```text
[B, 64]
```

The final linear layer combines them into one raw score:

```text
[B, 64]
   ↓ Linear 64→1
[B, 1]
   ↓ squeeze
[B]
```

That raw score is the logit.

```text
large negative logit ───── 0 ───── large positive logit
      likely clean        0.5         likely defective
                         after sigmoid
```

Sigmoid maps the logit to `p(defect)` between 0 and 1.

## 8. Mental model of the whole network

```mermaid
flowchart LR
    A[Pixels] --> B[local patterns]
    B --> C[richer feature maps]
    C --> D[compressed representation]
    D --> E[64-number vector]
    E --> F[one logit]
    F --> G[defect probability]
```

The feature extractor learns *what visual evidence exists*; the classifier head combines that evidence into the binary decision.
