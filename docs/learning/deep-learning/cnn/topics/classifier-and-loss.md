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

Sigmoid converts a logit `z` into a number between 0 and 1:

```text
p = sigmoid(z) = 1 / (1 + exp(-z))
```

For us:

```text
p = P(defect)
```

So:

```text
large negative logit → p near 0
logit = 0            → p = 0.5
large positive logit → p near 1
```

Examples:

```text
z =  2 → sigmoid(z) ≈ 0.881
z = -2 → sigmoid(z) ≈ 0.119
```

One output is sufficient because:

```text
P(clean) = 1 - P(defect)
```

A two-output softmax classifier could also solve the task, but would add unnecessary complexity for this binary baseline.

## BCEWithLogitsLoss

We train with:

```python
criterion = nn.BCEWithLogitsLoss()
```

It answers:

> How incompatible is the model's raw prediction with the true binary label?

Its inputs are:

```text
logit z
+
true label y ∈ {0,1}
```

## Binary Cross Entropy

Conceptually, first imagine converting the logit into:

```text
p = P(defect)
```

Binary Cross Entropy is:

```text
BCE = -[y log(p) + (1-y) log(1-p)]
```

The formula becomes simpler because `y` is either 0 or 1.

### True label `y = 1` — defective

Then:

```text
BCE = -log(p)
```

Examples:

```text
p = 0.90 → loss ≈ 0.105
p = 0.10 → loss ≈ 2.303
```

So a correct confident prediction receives small loss and a confident wrong prediction receives large loss.

### True label `y = 0` — clean

Then:

```text
BCE = -log(1-p)
```

Examples:

```text
p(defect) = 0.10 → loss ≈ 0.105
p(defect) = 0.90 → loss ≈ 2.303
```

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

For a positive example (`y=1`):

| P(defect) | Approx. loss |
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

combines the sigmoid-related calculation and binary cross entropy into a numerically stable formulation.

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

The loss is calculated for each prediction-label pair:

```text
2.1  ↔ 1 → loss₁
-1.4 ↔ 0 → loss₂
0.3  ↔ 1 → loss₃
-0.7 ↔ 0 → loss₄
```

By default, PyTorch returns the mean of those losses:

```text
(loss₁ + loss₂ + loss₃ + loss₄) / 4
```

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

```text
d(loss) / d(logit) = sigmoid(logit) - label
```

Intuitively:

```text
prediction too high
→ gradient pushes the logit downward

prediction too low
→ gradient pushes the logit upward
```

This is only a preview; gradients and backpropagation deserve their own note rather than being treated as a black box here.

## Questions to be able to answer

1. What does `Linear(64,1)` do?
2. What is a logit?
3. Why is a logit not a probability?
4. Why is one output enough for binary classification?
5. What does sigmoid do?
6. What does Binary Cross Entropy measure?
7. Why is a confident wrong prediction strongly penalized?
8. Why does `BCEWithLogitsLoss` expect raw logits?
9. Why should sigmoid not be applied before `BCEWithLogitsLoss`?
10. What does PyTorch return for the loss of a batch by default?
11. Does calculating the loss change any weights?
12. What concepts are still needed to turn loss into learning?
