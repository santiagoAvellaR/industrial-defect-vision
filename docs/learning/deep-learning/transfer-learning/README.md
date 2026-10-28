# Transfer learning

This section records the concepts used in M3, where the scratch CNN baseline is compared with a pretrained visual backbone.

## Reading order

1. [Pretraining and freezing](topics/pretraining-and-freezing.md)
2. [Fine-tuning](topics/fine-tuning.md)

## Core mental model

A scratch model starts from random parameters and must learn useful visual features only from the project dataset.

Transfer learning starts from parameters learned on a previous visual task and reuses those representations for a new task.

```mermaid
flowchart LR
    A[Large source dataset] --> B[Pretraining]
    B --> C[Pretrained visual features]
    C --> D[Industrial defect dataset]
    D --> E[Adapt classifier and/or backbone]
    E --> F[Clean vs defective]
```

The source task and target task do not need to have the same labels. What is transferred is mainly the learned representation encoded in the network weights.

## M3 questions

By the end of M3, we should be able to answer:

- What does a pretrained model actually contain?
- Which parameters are reused and which are newly initialized?
- What does freezing a backbone change in backpropagation?
- What is the difference between feature extraction and fine-tuning?
- Why can transfer learning help when the target dataset is relatively small?
- What is domain shift?
- Why can a smaller learning rate be appropriate during fine-tuning?
- How should transfer learning be compared fairly against the scratch baseline?

## Project plan

The controlled comparison is:

1. scratch CNN baseline from M2;
2. pretrained backbone with the backbone frozen and a new binary classifier trained;
3. pretrained backbone with selected or all backbone parameters unfrozen for fine-tuning;
4. comparison using the same train/validation split and the same classification metrics.

The purpose is not to use a large architecture because it is fashionable. The purpose is to test whether pretrained visual representations improve separation between clean and defective samples, especially the difficult localized defects found during M2 error analysis.
