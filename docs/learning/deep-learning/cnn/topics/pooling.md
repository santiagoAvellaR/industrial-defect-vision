# Pooling and spatial compression

Pooling reduces spatial information in feature maps while keeping a compact summary of activations.

Our scratch CNN uses local max pooling between convolution blocks and a global pooling operation before the final classifier. These two uses of pooling solve different problems.

## Local MaxPool

The baseline uses:

```python
nn.MaxPool2d(2)
```

A `2×2` MaxPool keeps the largest activation in each local region.

For one pooling region $R$ with activations $a_{i,j}$:

```math
y = \max_{(i,j)\in R} a_{i,j}
```

With the default stride of 2, spatial resolution is halved:

```text
640 × 224 → 320 × 112
320 × 112 → 160 × 56
160 × 56  → 80 × 28
```

The channel count does not change.

For example:

```text
[32, 320, 112]
      ↓ MaxPool
[32, 160, 56]
```

### Mental model

Before local MaxPool, the feature map tells us where every activation occurred.

After local MaxPool, we preserve approximately:

> there was a strong response somewhere inside this small region

but lose the exact position of that response inside the region.

This spatial compression reduces computation and gives some robustness to small shifts. It is often acceptable for classification, but the lost spatial precision becomes much more important for segmentation.

## Final feature tensor in our baseline

After the three convolution blocks, one image is represented by:

```text
[64, 80, 28]
```

For a batch:

```text
[B, 64, 80, 28]
```

At this point the original RGB pixels are no longer the current representation. The CNN has transformed them into 64 learned feature maps.

The final classifier needs a fixed-size vector, so we must summarize the remaining `H×W` dimensions.

## Global Average Pooling — GAP

Global Average Pooling averages every spatial position of each feature map.

In PyTorch:

```python
average = features.mean(dim=(2, 3))
```

For feature map $c$:

```math
g_c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} a_{c,i,j}
```

So:

```text
[B, 64, 80, 28]
        ↓ GAP
[B, 64]
```

### Mental model

GAP asks something like:

> How strongly is this learned feature present across the image overall?

It keeps global evidence but can dilute a feature that is strongly active in only a very small spatial region.

Suppose a defect-sensitive feature map has a few strong activations around a tiny scratch and low values everywhere else. GAP averages those few strong responses together with thousands of low responses.

That does not mean GAP is wrong. It means its summary emphasizes distributed evidence rather than the single strongest location.

## Global Max Pooling — GMP

Global Max Pooling keeps only the strongest spatial activation of each feature map.

In PyTorch:

```python
maximum = features.amax(dim=(2, 3))
```

For feature map $c$:

```math
g_c = \max_{i,j} a_{c,i,j}
```

Again:

```text
[B, 64, 80, 28]
        ↓ GMP
[B, 64]
```

### Mental model

GMP asks:

> Did this learned feature activate very strongly anywhere in the image?

This is attractive for defect classification because a single localized defect is enough to make the whole image positive.

The trade-off is that GMP can also react strongly to an isolated texture fluctuation or noise peak. In other words, it can preserve localized signal but can also amplify localized false evidence.

## GAP versus GMP

The two operations summarize the same feature maps differently.

| Pooling | Main question | Strength | Risk |
| --- | --- | --- | --- |
| GAP | How much evidence exists overall? | global context | localized evidence may be diluted |
| GMP | Is there strong evidence anywhere? | localized peaks | isolated noise can dominate |

Neither operation adds trainable parameters.

The only change is how the spatial feature maps are summarized before classification.

## Combined GAP + GMP

EXP-007 keeps both summaries.

```python
average = features.mean(dim=(2, 3))
maximum = features.amax(dim=(2, 3))
pooled = torch.cat((average, maximum), dim=1)
```

Shapes:

```text
GAP: [B, 64]
GMP: [B, 64]

concatenate
↓
[B, 128]
```

The final linear classifier then learns how much to use the global and localized evidence.

This changes the classifier from:

```text
Linear(64, 1)
```

to:

```text
Linear(128, 1)
```

so only 64 additional classifier weights are introduced.

### Mental model

Combined pooling asks both:

> Is this feature broadly present?

and:

> Did this feature appear very strongly somewhere?

The classifier receives both answers instead of forcing us to choose one summary in advance.

## Project evidence from M2

The pooling variants were tested using the same dataset split, seed, optimizer, learning rate, `pos_weight=8.47` and 20-epoch training budget.

### Original GAP baseline

Best tested validation operating point by F1:

- threshold `0.80`;
- precision `0.8485`;
- recall `0.5714`;
- F1 `0.6829`;
- TP `28`, FP `5`, FN `21`.

### GMP

Best tested validation F1:

- threshold `0.90`;
- precision `0.9643`;
- recall `0.5510`;
- F1 `0.7013`;
- TP `27`, FP `1`, FN `22`.

A particularly useful comparison is at matched recall. At recall `0.6735`, the original GAP model needed `41` false positives, while GMP needed `20` false positives for the same `33` true positives.

This is evidence that the localized maximum contained useful information that GAP alone was diluting.

### Combined GAP + GMP

Best tested validation F1 was again `0.7013`, so looking only at the maximum F1 would suggest no improvement over GMP.

But matched-recall comparisons reveal additional improvement.

At recall `0.5714`:

| Pooling | Precision | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| GAP | 0.8485 | 28 | 5 | 21 |
| GMP | 0.8000 | 28 | 7 | 21 |
| GAP + GMP | 0.8750 | 28 | 4 | 21 |

At recall `0.6122`:

| Pooling | Precision | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| GMP | 0.6383 | 30 | 17 | 19 |
| GAP + GMP | 0.7692 | 30 | 9 | 19 |

So combined pooling improves several parts of the precision-recall trade-off even though its single best F1 value happens to equal GMP on this validation split.

This is an important evaluation lesson:

> One scalar maximum can hide meaningful differences between models. Compare operating points and the full trade-off, not only the best F1.

## Connection with error analysis

Before changing pooling, error analysis showed:

- median defect area among TP: `1.852%` of the image;
- median defect area among FN: `0.679%`;
- many missed defects were small or highly localized.

That evidence motivated testing whether stronger localized aggregation would help.

The experiment supported the hypothesis, but did not completely solve the problem. High precision can now be obtained with very few false alarms, while recall at those operating points remains around `55–60%`.

This is why the next meaningful step is not to keep inventing pooling variants. A stronger pretrained feature extractor is a more valuable experiment.

## Flatten versus global pooling

Flattening does not summarize values; it rearranges them into one long vector.

```text
[B, 64, 80, 28]
        ↓ flatten
[B, 143360]
```

The number of values per image would be:

```math
64\times 80\times 28 = 143{,}360
```

A dense classifier after this vector would require many more parameters and would preserve exact spatial positions, increasing overfitting risk for this small dataset.

Global pooling deliberately removes `H` and `W` and keeps a compact feature summary.

## Flatten versus squeeze

These operations are also different.

`flatten` combines dimensions.

`squeeze` removes a dimension only when its size is `1`.

Our classifier produces:

```text
[B, 1]
```

and:

```python
.squeeze(1)
```

returns:

```text
[B]
```

No prediction is discarded; only a redundant dimension is removed.

## Interview-level explanation

A concise explanation is:

> Local MaxPool reduces spatial resolution while retaining strong local activations. Before classification, global pooling converts each final feature map into a fixed-size summary. GAP measures average feature presence and GMP keeps the strongest localized response. In our defect dataset, error analysis suggested localized defects were difficult, so I compared GAP, GMP and a concatenation of both under the same training setup. GMP and combined pooling improved the precision-recall trade-off, showing that localized evidence mattered, but the remaining recall limitation motivated moving to transfer learning rather than continuing to hand-tune pooling.

## Questions to be able to answer

1. What does `MaxPool2d(2)` do to `H`, `W` and channel count?
2. What spatial information does local MaxPool discard?
3. Why is spatial compression more delicate for segmentation than classification?
4. What does GAP compute?
5. What does GMP compute?
6. Why can GAP dilute a localized defect signal?
7. Why can GMP create extra false positives?
8. Why does concatenating GAP and GMP produce 128 features from 64 feature maps?
9. Does GAP or GMP add trainable parameters?
10. Why is matched-recall comparison more informative than comparing only maximum F1?
11. What project evidence motivated the pooling experiments?
12. Why did we stop iterating on pooling instead of continuing indefinitely?
