# Activation functions — ReLU

After each convolution in the baseline we use:

```python
nn.ReLU()
```

ReLU is:

```text
ReLU(x) = max(0, x)
```

So:

```text
before ReLU     after ReLU

-2.0               0.0
-0.3               0.0
 0.0               0.0
 1.5               1.5
 4.2               4.2
```

ReLU changes values, not tensor shape.

## When exactly is ReLU applied?

The convolution first produces **pre-activations**.

Example:

```text
-1.2   0.7   2.1
 0.3  -0.8   1.5
-2.4   3.0  -0.2
```

ReLU is then applied element-by-element:

```text
 0.0   0.7   2.1
 0.3   0.0   1.5
 0.0   3.0   0.0
```

The values after the activation function are called **activations**.

Mental model:

```text
patch + learned weights
          ↓
convolution
          ↓
pre-activation z
          ↓
ReLU(z)
          ↓
activation a
```

## Why do we need a non-linear activation?

A convolution or linear layer is an affine transformation. If we compose only affine transformations, the whole stack can still be represented as one affine transformation.

```text
affine → affine → affine

can collapse to

one affine transformation
```

A non-linear activation prevents that global collapse and greatly increases the functions a deep network can represent.

## But ReLU leaves positive values unchanged

Correct.

ReLU is globally non-linear because its rule changes at zero:

```text
x < 0  → 0
x ≥ 0  → x
```

But in a region where all relevant pre-activations are positive:

```text
ReLU(x) = x
```

So for that local input region the network can behave like an affine function.

This does not make the complete network linear. Different inputs cross different ReLU boundaries and activate different subsets of units.

A ReLU network is therefore **piecewise linear**: affine inside local regions, globally non-linear across regions.

ReLU does not need to modify every activation for the overall network to be non-linear.

## Why does ReLU zero negative values?

There is no mathematical law saying negative values are “bad”. ReLU is a design choice with useful optimization properties.

Its derivative is:

```text
x < 0  → derivative 0
x > 0  → derivative 1
```

On the positive side it lets the activation pass unchanged and, during backpropagation, does not shrink the gradient because of the activation itself.

The negative side acts like a gate that turns that response off.

Do not interpret this as:

```text
positive = useful
negative = useless
```

The filters adapt during training, so the representation itself changes.

## Why ReLU instead of another activation?

There is no universal best activation function.

Useful criteria include:

- does it introduce non-linearity?
- what happens to gradients?
- computational cost;
- numerical stability;
- empirical performance;
- simplicity for the current architecture.

### ReLU

Advantages:

- simple;
- very cheap;
- positive side has derivative 1;
- strong historical default for CNNs;
- easy to understand for a baseline.

Main drawback:

- negative side has zero gradient;
- a unit can remain inactive for many inputs (`dying ReLU`).

### LeakyReLU

Keeps a small non-zero slope on the negative side.

### GELU

Smooth activation widely used in Transformers and other modern architectures.

### SiLU / Swish

Smooth activation used successfully in several modern CNNs.

### Sigmoid

Maps values to `(0,1)` and is useful for probability interpretation, but hidden layers can suffer from saturation and small gradients.

## Why we use ReLU here

Not because it is always optimal, but because it is:

```text
simple + standard + cheap + adequate for a baseline
```

Changing the activation later should be an experiment motivated by evidence, not complexity added for its own sake.

## Questions to be able to answer

1. When is ReLU applied in our CNN?
2. What is a pre-activation? What is an activation?
3. Why do affine layers alone not give a deep network enough expressive power?
4. Why is ReLU globally non-linear if positive values pass unchanged?
5. What does *piecewise linear* mean here?
6. Why should negative activations not simply be called “bad features”?
7. What criteria matter when choosing an activation function?
8. Why did we choose ReLU for this baseline?
