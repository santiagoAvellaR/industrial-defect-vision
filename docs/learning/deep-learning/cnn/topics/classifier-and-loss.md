# Classifier, logits and binary loss

After the convolutional feature extractor and global average pooling, each image is represented by 64 values:

```text
[B, 64]
```

The final classifier is:

```python
nn.Linear(64, 1)
```

It converts those 64 learned features into one raw classification score per image.

## Linear classifier head

The shape flow is:

```text
[B, 64]
   ↓ Linear(64,1)
[B, 1]
   ↓ squeeze(1)
[B]
```

The output is one number per image because our task is binary classification.

## What is a logit?

The raw output of the linear layer is a **logit**.

A logit can be any real number:

```text
-4.2
 0.0
 3.7
```

It is not a probability.

For our convention, positive logits indicate stronger evidence for the positive class (`defective`) and negative logits indicate stronger evidence for the negative class (`clean`).

## From logit to probability: sigmoid

Sigmoid converts a logit $z\in\mathbb{R}$ into a probability-like value between 0 and 1:

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

For our binary classifier:

$$
p = P(y=1\mid x) = P(\text{defect}\mid x) = \sigma(z)
$$

So:

```text
large negative logit → p near 0
logit = 0            → p = 0.5
large positive logit → p near 1
```

For example:

$$
\sigma(2) \approx 0.881,
\qquad
\sigma(-2) \approx 0.119
$$

One output is sufficient because:

$$
P(\text{clean}\mid x) = 1 - P(\text{defect}\mid x)
$$

A two-output softmax classifier could also solve the task, but would add unnecessary complexity for this binary baseline.

## BCEWithLogitsLoss

We train with:

```python
criterion = nn.BCEWithLogitsLoss()
```

It answers:

> How incompatible is the model's raw prediction with the true binary label?

Its inputs are a raw logit $z$ and a true label $y\in\{0,1\}$.

## Binary Cross Entropy

Conceptually, first imagine converting the logit into $p=\sigma(z)$.

Binary Cross Entropy for one sample is:

$$
\mathcal{L}_{\text{BCE}}(y,p)
=
-\left[y\log(p) + (1-y)\log(1-p)\right]
$$

The formula becomes simpler because $y$ is either 0 or 1.

### True label $y=1$ — defective

Substituting $y=1$ gives:

$$
\mathcal{L}_{\text{BCE}} = -\log(p)
$$

For a good prediction such as $p=0.90$:

$$
-\log(0.90) \approx 0.105
$$

For a confident wrong prediction such as $p=0.10$:

$$
-\log(0.10) \approx 2.303
$$

So a correct confident prediction receives small loss and a confident wrong prediction receives large loss.

### True label $y=0$ — clean

Substituting $y=0$ gives:

$$
\mathcal{L}_{\text{BCE}} = -\log(1-p)
$$

If the model assigns only $p=0.10$ probability to defect:

$$
-\log(1-0.10) = -\log(0.90) \approx 0.105
$$

But if it assigns $p=0.90$ probability to defect:

$$
-\log(1-0.90) = -\log(0.10) \approx 2.303
$$

Again, confident wrong predictions are strongly penalized.

## Intuition for the logarithm

The useful behavior is:

```text
correct + confident
→ very small loss

correct but uncertain
→ intermediate loss

wrong + confident
→ large loss
```

For a positive example ($y=1$):

| $P(\text{defect})$ | Approx. loss |
|---:|---:|
| 0.99 | 0.01 |
| 0.90 | 0.11 |
| 0.50 | 0.69 |
| 0.10 | 2.30 |
| 0.01 | 4.61 |

The same behavior is mirrored for negative examples.

## Why `BCEWithLogitsLoss` instead of sigmoid + BCE?

A naive implementation could be:

```python
probabilities = torch.sigmoid(logits)
loss = binary_cross_entropy(probabilities, labels)
```

Conceptually this expresses the same mathematics.

However, very large positive or negative logits can make intermediate probabilities numerically extremely close to 0 or 1. Calculating logarithms of those values can be unstable.

PyTorch's:

```python
nn.BCEWithLogitsLoss()
```

combines sigmoid and BCE into one numerically stable expression. For one sample, an equivalent stable form is:

$$
\mathcal{L}(z,y)
=
\max(z,0) - zy + \log\left(1+e^{-|z|}\right)
$$

This computes the same binary classification objective without explicitly forming a potentially extreme probability first.

Therefore the training flow is:

```text
raw logits
   +
true labels
   ↓
BCEWithLogitsLoss
   ↓
loss
```

Do **not** apply sigmoid first when using `BCEWithLogitsLoss`.

Sigmoid is still useful when we explicitly want probability interpretation during inference or evaluation.

## What happens for a batch?

Our model returns:

```text
logits: [B]
labels: [B]
```

For example:

```text
logits = [ 2.1, -1.4, 0.3, -0.7]
labels = [ 1.0,  0.0, 1.0,  0.0]
```

The loss is calculated for each prediction-label pair. With PyTorch's default `reduction="mean"`, the batch loss is:

$$
\mathcal{L}_{\text{batch}}
=
\frac{1}{B}\sum_{i=1}^{B}\mathcal{L}_i
$$

For a batch of four samples:

$$
\mathcal{L}_{\text{batch}}
=
\frac{\mathcal{L}_1+\mathcal{L}_2+\mathcal{L}_3+\mathcal{L}_4}{4}
$$

So the final `loss` is one scalar for the batch.

## The loss does not update the model

This boundary is essential:

```text
model predictions
+
labels
    ↓
loss function
    ↓
scalar loss
```

At this point we have only **measured the error**.

No filter has changed yet.

The next block explains how that scalar becomes learning:

```text
loss
 ↓
gradients
 ↓
backpropagation
 ↓
optimizer
 ↓
updated parameters
```

## Preview: gradient with respect to the logit

We will derive gradients later, but one elegant result for one binary example is:

$$
\frac{\partial \mathcal{L}}{\partial z}
=
\sigma(z)-y
$$

This already gives useful intuition:

- if the predicted probability is too high relative to the label, the gradient is positive and gradient descent will tend to push the logit downward;
- if the predicted probability is too low relative to the label, the gradient is negative and gradient descent will tend to push the logit upward.

This is only a preview; gradients and backpropagation deserve their own note rather than being treated as a black box here.

## Questions to be able to answer

1. What does `Linear(64,1)` do?
2. What is a logit?
3. Why is a logit not a probability?
4. Why is one output enough for binary classification?
5. What does sigmoid do mathematically?
6. What does Binary Cross Entropy measure?
7. Why is a confident wrong prediction strongly penalized?
8. Why does `BCEWithLogitsLoss` expect raw logits?
9. Why should sigmoid not be applied before `BCEWithLogitsLoss`?
10. What does PyTorch return for the loss of a batch by default?
11. Does calculating the loss change any weights?
12. What concepts are still needed to turn loss into learning?
