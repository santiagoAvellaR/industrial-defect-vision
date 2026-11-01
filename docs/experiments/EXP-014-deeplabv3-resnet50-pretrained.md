# Experiment EXP-014 — Pretrained DeepLabV3-ResNet50

## Question

Does a modern pretrained semantic-segmentation model outperform the SimpleUNet M4 baseline on the same KSDD2 split and segmentation objective?

## Hypothesis

DeepLabV3-ResNet50 starts from a much stronger representation than SimpleUNet trained from scratch. Its pretrained ResNet50 backbone and atrous spatial pyramid pooling should improve defect localization, especially when the dataset is small and foreground pixels are rare.

The main risk is that the model is substantially larger and its pretrained representation comes from natural images rather than industrial surfaces. Better capacity therefore does not guarantee better validation Dice.

## Setup

Keep the successful EXP-013 training setup wherever it remains meaningful:

- Dataset: KSDD2
- Split: same train/validation split
- Seed: 42
- Model: torchvision DeepLabV3-ResNet50
- Pretrained weights: `DeepLabV3_ResNet50_Weights.DEFAULT` (`COCO_WITH_VOC_LABELS_V1`)
- Output head: replace the 21-class classifier output with one binary defect logit map
- Auxiliary classifier: disabled after loading pretrained weights
- Image resize: same `640 x 224` pipeline
- Preprocessing: ImageNet normalization required by the pretrained weights
- Augmentation: none
- Optimizer: Adam
- LR: `3e-4`
- Batch size: 4
- Loss: BCE on all images + soft Dice only on non-empty target masks
- Epochs: 20
- Evaluation threshold during training: 0.5
- Checkpoints: lowest validation loss and highest validation Dice
- Runtime target: Google Colab GPU
- DataLoader runtime: `num_workers=2`, `pin_memory=True`
- DeepLab training loader: `drop_last=True` to prevent a final singleton batch from entering BatchNorm; validation keeps every sample

## Controlled comparison with EXP-013

Held constant:

- dataset and split;
- seed;
- resize;
- no augmentation;
- BCE + positive-mask-only Dice objective;
- Adam optimizer;
- learning rate;
- batch size;
- epoch budget;
- validation metrics;
- checkpoint selection rules;
- initial threshold of 0.5 followed by a validation threshold sweep.

Changed as one model package:

- SimpleUNet trained from scratch -> pretrained DeepLabV3-ResNet50;
- unnormalized input -> ImageNet normalization expected by the pretrained model;
- DeepLab training uses `drop_last=True` because its ASPP BatchNorm cannot train on a final batch containing only one image.

The preprocessing difference is intentional and is part of using the pretrained representation correctly, so this is a controlled model-family comparison rather than a strict one-variable ablation. `drop_last=True` is an architecture/runtime requirement rather than a modeling hypothesis; the number of omitted training samples per epoch is recorded in the checkpoint.

## Baseline to beat

EXP-013 best validation-Dice checkpoint at threshold `0.5`:

- Dice: `0.4201`
- IoU: `0.2659`
- Precision: `0.6702`
- Recall: `0.3060`
- Epoch: 17

## First GPU smoke test

The first Colab smoke test confirmed that batch size 4 fits on the Tesla T4 and the pretrained weights load correctly. The run failed near the end of the training epoch, not because of CUDA memory, but because the final training batch contained one image.

DeepLabV3's ASPP pooling branch produces a `1 x 1` feature map followed by BatchNorm. With a singleton batch the tensor shape was `[1, 256, 1, 1]`, giving BatchNorm only one value per channel and triggering:

`ValueError: Expected more than 1 value per channel when training`

Fix: the DeepLab training DataLoader now uses `drop_last=True`. The validation DataLoader remains unchanged and evaluates the complete validation set.

## Second GPU smoke test

After the `drop_last=True` fix, the one-epoch smoke test completed end to end on the Colab Tesla T4 with batch size 4, `num_workers=2` and pinned memory.

Runtime evidence:

- `train_drop_last=True`;
- one training sample omitted in the incomplete final batch;
- forward, loss, backward, optimizer step and validation all completed;
- both best-loss and best-Dice smoke checkpoints were saved.

One-epoch validation metrics at threshold `0.5`:

- validation loss: `0.2223`;
- Dice: `0.4288`;
- IoU: `0.2729`;
- precision: `0.5678`;
- recall: `0.3445`.

These values are **not the final EXP-014 result**. The smoke run is disposable and exists to validate the pipeline before the controlled 20-epoch experiment. The fact that one epoch already reaches a Dice near the EXP-013 best checkpoint is encouraging but should not be used to select the final model.

## Run command

```bash
python scripts/train_deeplabv3.py /content/ksdd2 \
  --epochs 20 \
  --batch-size 4 \
  --num-workers 2 \
  --pin-memory \
  --learning-rate 3e-4 \
  --threshold 0.5 \
  --dice-positive-only \
  --checkpoint artifacts/checkpoints/EXP-014-deeplabv3-resnet50-pretrained_best_loss.pt \
  --dice-checkpoint artifacts/checkpoints/EXP-014-deeplabv3-resnet50-pretrained_best_dice.pt
```

## Evaluation after training

1. Compare the best-Dice checkpoint against EXP-013 at the common threshold `0.5`.
2. Run the validation threshold sweep on the fixed best-Dice checkpoint.
3. Generate qualitative predictions using the selected validation threshold.
4. Compare quantitative and qualitative evidence before selecting the M5 candidate.

## Results

Pending full 20-epoch training.

## Interpretation

Pending full 20-epoch training.

## Conclusion

Pending full 20-epoch training.

## MLflow run

Not tracked yet.
