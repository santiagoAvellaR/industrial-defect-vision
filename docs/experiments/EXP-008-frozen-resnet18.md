# EXP-008 — Frozen pretrained ResNet18

## Question

Can ImageNet-pretrained ResNet18 features separate clean and defective KSDD2 images better than the scratch CNN when the pretrained backbone is kept completely frozen?

## Setup

- model: torchvision ResNet18 with pretrained ImageNet weights;
- input size: `(640, 224)` in PyTorch `(H, W)` order;
- ImageNet channel normalization enabled;
- original ResNet classifier replaced by `Linear(512, 1)`;
- backbone frozen, including BatchNorm running statistics;
- trainable parameters: 513;
- weighted BCE with `pos_weight=8.47`;
- Adam on the new classifier;
- learning rate: `1e-3`;
- epochs: 10;
- seed: 42;
- same train/validation split used throughout M2;
- checkpoint selected by minimum validation loss.

## Training result

The best-loss checkpoint was epoch 10:

- validation loss: `0.8579`;
- at threshold `0.50`: precision `0.3793`, recall `0.6735`, F1 `0.4853`;
- TP `33`, FP `54`, FN `16`.

The validation loss continued to improve slowly, but the fixed-threshold classification metrics fluctuated because samples moved around the decision threshold.

## Threshold sweep

Best tested F1 occurred at threshold `0.60`:

- precision: `0.6078`;
- recall: `0.6327`;
- F1: `0.6200`;
- TP: `31`;
- FP: `20`;
- FN: `18`.

At higher recall the frozen model required many false positives. For example:

- threshold `0.45`: recall `0.7143`, precision `0.3043`, FP `80`;
- threshold `0.40`: recall `0.7347`, precision `0.2416`, FP `113`.

## Comparison with the scratch CNN

The best M2 scratch model with combined GAP+GMP reached a best tested validation F1 of `0.7013`, compared with `0.6200` for frozen ResNet18.

At a similar recall around `0.63`:

- scratch GAP+GMP: precision about `0.74`;
- frozen ResNet18: precision about `0.61`.

Therefore the frozen pretrained representation does not outperform the project-specific scratch CNN on this dataset.

## Interpretation

The result does not mean pretraining is useless. It means the ImageNet representation, when used without adapting the backbone, is not sufficiently aligned with the industrial texture domain.

This is evidence of domain shift: KSDD2 consists of narrow industrial surfaces, microtextures and localized defects that differ substantially from the source pretraining domain.

The frozen model still achieves meaningful separation, so the pretrained features contain useful visual information. The next controlled question is whether adapting the deepest residual stage improves that representation without discarding the lower-level pretrained features.

## Next action

Start from the trained frozen checkpoint and fine-tune only ResNet18 `layer4` plus the binary classifier.

Use a smaller learning rate for the pretrained stage than for the classifier so that the deep features adapt conservatively rather than being overwritten aggressively.
