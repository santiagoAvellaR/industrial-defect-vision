# EXP-009 — ResNet18 partial fine-tuning (`layer4`)

## Question

Does adapting the deepest ResNet18 stage to KSDD2 improve the precision-recall trade-off compared with both the frozen pretrained backbone and the scratch CNN baseline?

## Setup

Starting point:

- pretrained ResNet18;
- classifier already trained in EXP-008 with the backbone frozen;
- checkpoint: `artifacts/checkpoints/resnet18_frozen.pt`.

Fine-tuning configuration:

- early backbone stages frozen;
- `layer4` trainable;
- binary classifier trainable;
- ImageNet normalization enabled;
- input resize kept at `640 x 224`;
- `pos_weight=8.47`;
- classifier learning rate: `1e-3`;
- `layer4` learning rate: `1e-4`;
- seed: `42`;
- 10 epochs;
- checkpoint selected by lowest validation loss.

Trainable parameters: `8,394,241`.

## Training result

Validation loss reached its minimum at epoch 5:

| Epoch | Train loss | Train F1 | Val loss | Val precision | Val recall | Val F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.4699 | 0.3430 | 0.9003 | 0.8974 | 0.7143 | 0.7955 |
| 2 | 0.6973 | 0.7099 | 0.6764 | 0.9000 | 0.7347 | 0.8090 |
| 3 | 0.4708 | 0.8045 | 0.6615 | 0.8780 | 0.7347 | 0.8000 |
| 5 | 0.2055 | 0.9163 | **0.6126** | 0.8372 | 0.7347 | 0.7826 |
| 7 | 0.0424 | 0.9776 | 1.0670 | 0.8718 | 0.6939 | 0.7727 |
| 10 | 0.0221 | 0.9949 | 1.2293 | 0.8537 | 0.7143 | 0.7778 |

The saved checkpoint is therefore epoch 5.

The training metrics continue improving toward almost perfect classification after epoch 5 while validation loss gets worse. This is clear overfitting: additional optimization improves fit to the training set but no longer improves generalization.

## Threshold sweep on the saved checkpoint

Best tested validation F1:

- threshold: `0.70` (also `0.75` gives the same counts);
- precision: `0.9000`;
- recall: `0.7347`;
- F1: `0.8090`;
- TP: `36`;
- FP: `4`;
- FN: `13`.

The sweep also shows useful high-recall operating points. For example:

- threshold `0.25`: recall `0.8163`, precision `0.5882`, FP `28`;
- threshold `0.20`: recall `0.8367`, precision `0.4767`, FP `45`;
- threshold `0.10`: recall `0.9184`, precision `0.2528`, FP `133`.

## Matched-recall comparison

A particularly clean comparison is recall `0.7347`, where all three models detect `36/49` validation defects and miss `13`.

| Model | Precision | TP | FP | FN |
|---|---:|---:|---:|---:|
| Scratch CNN, GAP+GMP | 0.5806 | 36 | 26 | 13 |
| ResNet18 frozen | 0.2416 | 36 | 113 | 13 |
| **ResNet18 `layer4` fine-tuned** | **0.9000** | **36** | **4** | **13** |

This isolates the effect very clearly: partial fine-tuning keeps the same defect recall while dramatically reducing false alarms.

## Interpretation

EXP-008 showed that ImageNet features used unchanged were not sufficiently aligned with the industrial domain. EXP-009 shows that the pretrained representation is still valuable once its deeper stage is allowed to adapt.

The result supports the domain-shift interpretation:

- generic pretrained visual features are useful as initialization;
- a completely frozen representation is too rigid for KSDD2;
- adapting the deepest ResNet stage strongly improves class separation;
- full-backbone fine-tuning is not required to demonstrate the value of transfer learning here.

The later training epochs also show why checkpoint selection matters. Without keeping the best validation checkpoint, the final epoch would retain a model that fits train almost perfectly but generalizes worse.

## Conclusion

Partial fine-tuning is the strongest classification model tested so far.

The best tested validation operating point reaches `F1=0.8090`, and at recall `0.7347` it reduces false positives from `26` for the best scratch CNN to only `4`.

M3 has now demonstrated both required transfer-learning regimes:

1. frozen feature extraction;
2. unfreezing and target-domain adaptation.

Because partial fine-tuning already produces a large improvement while later epochs show strong overfitting, the next useful step is not to keep increasing classification capacity. The project should move to semantic segmentation, where the available masks allow the model to learn both whether a defect exists and where it is located.
