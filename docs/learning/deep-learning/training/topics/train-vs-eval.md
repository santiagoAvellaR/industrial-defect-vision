# Training mode vs evaluation mode

Training and validation use the same model parameters, but they do not have the same purpose.

- training uses data to update parameters;
- validation measures how the current parameters behave on data not used for those updates.

## `model.train()`

```python
model.train()
```

puts the module and its submodules in **training mode**.

This call does **not** perform training by itself. It only changes the behavior of layers that have different train/eval behavior.

The main examples are:

- `Dropout`: active during training, disabled during evaluation;
- `BatchNorm`: uses batch statistics during training and stored running statistics during evaluation.

Layers such as `Conv2d`, `ReLU`, `MaxPool2d` and `Linear` do not normally change their forward behavior just because `train()` or `eval()` was called.

Our current `BaselineCNN` has no Dropout or BatchNorm, so switching between `train()` and `eval()` does not materially change its forward computation today. We still use the modes correctly because later architectures may depend on them and because the intent of the code should be explicit.

## `model.eval()`

```python
model.eval()
```

puts the model in **evaluation mode**.

It does not freeze the parameters and it does not disable PyTorch autograd. It only switches the behavior of modules that depend on the training/evaluation flag.

Therefore this is incomplete for normal validation:

```python
model.eval()
logits = model(images)
```

PyTorch may still build a computational graph and track gradients even though the model is in evaluation mode.

## Disabling gradient tracking

During validation we only need predictions and metrics. We do not call `backward()` and do not update parameters.

A common pattern is:

```python
with torch.no_grad():
    logits = model(images)
    loss = criterion(logits, labels)
```

`torch.no_grad()` tells PyTorch not to build the autograd graph for operations executed inside the context.

Benefits include:

- less memory use;
- less unnecessary computation;
- clearer intent: this pass is not for learning.

## `eval()` and `no_grad()` solve different problems

| Tool | What it controls |
| --- | --- |
| `model.train()` | layer behavior for training |
| `model.eval()` | layer behavior for evaluation |
| `torch.no_grad()` | whether autograd records operations for gradients |

For normal validation we therefore use both:

```python
model.eval()

with torch.no_grad():
    for images, masks, labels in val_loader:
        logits = model(images)
        loss = criterion(logits, labels)
```

Important: `model.eval()` does not disable gradients, and `torch.no_grad()` does not switch Dropout or BatchNorm into evaluation behavior.

## What validation does not do

A standard validation pass does not call:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

because validation must not update the model.

We still calculate the loss because we want to measure how well the current model performs on validation data.

## One epoch: training then validation

A common high-level structure is:

```python
for epoch in range(num_epochs):
    model.train()
    for images, masks, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for images, masks, labels in val_loader:
            logits = model(images)
            loss = criterion(logits, labels.float())
```

The training loop changes the parameters batch by batch. The validation loop observes the resulting model without changing it.

## Why validate after an epoch?

Training loss tells us how well the model fits the data it is learning from.

Validation loss tells us how well those learned parameters behave on separate data.

The gap between the two is important for detecting overfitting.

Later we will record validation metrics after each epoch and use them to decide which checkpoint is the best one.

## Questions to be able to answer

1. Does `model.train()` actually train the model?
2. Which kinds of layers behave differently in train and eval modes?
3. Does `model.eval()` disable gradient computation?
4. What does `torch.no_grad()` do?
5. Why do we normally use both `model.eval()` and `torch.no_grad()` during validation?
6. Why do we still calculate validation loss if no weights are updated?
7. Why are `backward()` and `optimizer.step()` absent from validation?
8. Does our current `BaselineCNN` materially change its forward computation between train and eval mode?
