# Training and validation loop

This note connects the concepts studied so far into the real M2 classification loop.

## One training epoch

A training epoch visits every batch in the training loader once. For each batch:

```python
model.train()

for images, _, labels in train_loader:
    optimizer.zero_grad()

    logits = model(images)
    loss = criterion(logits, labels.float())

    loss.backward()
    optimizer.step()
```

The sequence is intentional:

1. clear gradients from the previous standard step;
2. run the forward pass;
3. compare logits with labels through the loss;
4. backpropagate to compute parameter gradients;
5. let the optimizer update the parameters.

The weights can therefore change after every training batch.

## One validation epoch

Validation measures the current model without changing it:

```python
model.eval()

with torch.no_grad():
    for images, _, labels in val_loader:
        logits = model(images)
        loss = criterion(logits, labels.float())
```

There is no `backward()` and no `optimizer.step()`.

## Epoch loss

`BCEWithLogitsLoss()` returns the mean loss of the current batch by default.

To report one mean over the whole epoch, the implementation multiplies each batch mean by its batch size:

```python
total_loss += loss.item() * batch_size
```

and divides by the total number of samples at the end:

```math
L_{\mathrm{epoch}}
=
\frac{\sum_b L_b N_b}{\sum_b N_b}
```

where `L_b` is the mean loss of batch `b` and `N_b` is its number of samples.

This matters because the final batch can be smaller than the others. Averaging batch means directly would give that smaller final batch the same weight as a full batch.

## Binary decision from a logit

For metrics, the baseline predicts `defective` when the logit is at least zero:

```python
prediction = logit >= 0
```

This is equivalent to applying sigmoid and using probability threshold `0.5` because:

```math
\sigma(0)=0.5
```

So sigmoid is not required just to obtain the binary class.

## Why accuracy is not enough

The KSDD2 training split is strongly imbalanced: most images are clean. A model that predicts `clean` for every image can therefore obtain a deceptively high accuracy.

The baseline reports:

- loss;
- accuracy;
- precision;
- recall;
- F1;
- true positives (`tp`);
- true negatives (`tn`);
- false positives (`fp`);
- false negatives (`fn`).

The confusion counts make failure modes visible. In particular, `tp = 0` and `fn` equal to the number of positive samples would reveal a model that has collapsed to predicting everything as clean.

For the positive `defective` class:

```math
\mathrm{precision}
=
\frac{TP}{TP+FP}
```

```math
\mathrm{recall}
=
\frac{TP}{TP+FN}
```

```math
F_1
=
2\frac{\mathrm{precision}\cdot\mathrm{recall}}
{\mathrm{precision}+\mathrm{recall}}
```

## Best checkpoint

The baseline saves the checkpoint with the lowest validation loss.

The model parameters themselves are updated after training batches and remain in memory throughout the run. That is separate from writing a checkpoint file to disk.

The `.pt` file is **not written at every epoch**. After each validation epoch, the script checks:

```python
if val_metrics["loss"] < best_val_loss:
```

Only when validation loss improves does it call `torch.save(...)`. Because the same checkpoint path is reused, each new best checkpoint replaces the previous best file.

Therefore:

- model weights continue evolving in memory every training batch;
- an epoch with no new best validation loss does not create or update the `.pt` file;
- the final `.pt` contains the parameters from the best validation-loss epoch, not necessarily the last epoch;
- it is also not necessarily the epoch with the best F1, recall or precision.

The checkpoint contains:

```text
model_state_dict
optimizer_state_dict
epoch
validation loss
validation metrics
pos_weight
seed
```

Using validation loss gives us a continuous criterion that is independent of choosing a classification threshold for model selection. Precision, recall and F1 are still reported because they are essential for interpreting behavior on this imbalanced dataset.

This creates an important distinction: **best according to validation loss is not automatically best according to the operational metric we care about.**

The checkpoint is written to:

`artifacts/checkpoints/baseline_cnn.pt`

by default. Model-weight files are ignored by Git.

## Why no class weighting yet?

The first run intentionally uses plain `BCEWithLogitsLoss()`.

The class imbalance is known, but adding `pos_weight`, a weighted sampler or another correction before observing the baseline would hide useful evidence. If the model collapses toward the majority class, that failure becomes the empirical reason for the next experiment.

## Implementation

Reusable epoch logic:

`src/industrial_vision/training/classification.py`

Executable training entry point:

`scripts/train_baseline.py`

## Questions to be able to answer

1. Why can parameters change after each batch but not during validation?
2. Why is epoch loss weighted by batch size?
3. Why is `logit >= 0` equivalent to probability `>= 0.5`?
4. Why can accuracy be misleading for KSDD2?
5. What do precision and recall measure for the defective class?
6. Why do we save the best model using validation rather than training loss?
7. Why is writing a checkpoint different from updating model weights?
8. Why can the best validation-loss checkpoint differ from the best-F1 epoch?
