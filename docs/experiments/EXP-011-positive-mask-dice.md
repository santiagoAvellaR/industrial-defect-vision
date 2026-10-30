# Experiment EXP-011 — BCE + Dice on positive masks only

## Question

Does excluding empty target masks from the Dice term reduce the tendency toward all-background predictions and improve foreground recall and Dice?

## Hypothesis

BCE should continue supervising every pixel in every image, including clean images. Restricting the overlap-based Dice term to masks that actually contain foreground may reduce unnecessary pressure toward zero foreground probabilities on the many clean KSDD2 images.

Expected effect:

- validation recall should increase;
- Dice and IoU should improve if the hypothesis is correct;
- precision may decrease somewhat as the model becomes less conservative;
- epoch-to-epoch all-background collapses should become less frequent or disappear.

## Setup

Keep the EXP-010 setup unchanged except for the Dice scope:

- Dataset: KSDD2
- Split: same train/validation split as EXP-010
- Seed: 42
- Model: SimpleUNet
- Resolution: dataset default preprocessing
- Augmentation: none
- Optimizer: Adam
- LR: `1e-3`
- Batch size: 4
- Loss: BCE on all images + soft Dice only on non-empty target masks
- Epochs: 10
- Hardware: CPU
- Evaluation threshold during training: 0.5
- Checkpoints: lowest validation loss and highest validation Dice

### Controlled variable

EXP-010:

`BCE(all images) + Dice(all images)`

EXP-011:

`BCE(all images) + Dice(positive-mask images only)`

No other training variable changed.

## Results

The best validation-loss checkpoint occurred at epoch 6. The best validation-Dice checkpoint occurred at epoch 9.

| Metric | Best validation loss checkpoint | Best validation Dice checkpoint |
|---|---:|---:|
| Epoch | 6 | 9 |
| Loss | 0.2275 | 0.2281 |
| Dice | 0.2685 | 0.3789 |
| IoU | 0.1551 | 0.2337 |
| Precision | 0.2267 | 0.5228 |
| Recall | 0.3293 | 0.2971 |

Compared with the highest validation Dice observed in EXP-010 (`0.3469`), EXP-011 increased the best observed Dice to `0.3789` at the same training threshold of `0.5`. The corresponding IoU increased from `0.2098` to `0.2337` and recall from `0.2632` to `0.2971`.

Unlike EXP-010, EXP-011 did not produce validation epochs with zero Dice. However, validation remained unstable: for example, epoch 3 reached recall `0.4200` but precision only `0.0130`, corresponding to millions of false-positive pixels.

### Validation threshold sweep

The epoch-9 best-Dice checkpoint was evaluated from threshold `0.10` to `0.90` using the validation split only.

At the training threshold:

- threshold: `0.50`
- Dice: `0.3789`
- IoU: `0.2337`
- precision: `0.5228`
- recall: `0.2971`

The highest validation Dice in the sweep was:

- threshold: `0.90`
- Dice: `0.3877`
- IoU: `0.2405`
- precision: `0.6361`
- recall: `0.2788`

Increasing the threshold mainly traded recall for precision. The Dice improvement from `0.50` to `0.90` was modest (`+0.0088` absolute), so threshold tuning helps this checkpoint but does not explain the larger improvement from EXP-010 to EXP-011.

The selected `0.90` threshold is validation-derived and checkpoint-specific. It must not be re-selected using the test split.

## Qualitative observations

Representative validation predictions were inspected at threshold `0.90` using the epoch-9 best-Dice checkpoint.

- Best overlap (`12287.png`): Dice `0.9334`, with `2630` ground-truth pixels and `2382` predicted pixels. The predicted region closely matches the visible compact defect and demonstrates that the model can learn accurate localization and shape on some samples.
- Partial defect (`10421.png`): Dice `0.5488`, with `3090` ground-truth pixels and `1739` predicted pixels. The model detects the correct elongated defect region but predicts only part of its full extent, confirming a real under-segmentation failure mode.
- False negative (`10574.png`): ground-truth area `5242` pixels with zero predicted foreground pixels. This is not simply a tiny-defect failure: the target is relatively large, yet the model does not produce a meaningful response at the defect location.
- False positive (`10291.png`): clean target with `4918` predicted foreground pixels. The model produces a strong localized response on background texture, indicating that some normal surface patterns can be confused with defects.
- Clean correct (`10001.png`): both target and prediction are empty, showing that the model can also suppress foreground correctly on clean samples.

The probability maps show qualitatively different regimes rather than a single thresholding problem: some defects receive a strong, well-localized response; some receive only partial activation; some defects are missed almost entirely; and some clean texture regions receive high-confidence false activations.

## Failure modes

- Validation predictions remain unstable across epochs even though all-background collapse disappeared.
- Some epochs strongly over-predict foreground, producing very high false-positive pixel counts.
- Foreground recall remains limited: even the best-Dice checkpoint detects less than one third of defect pixels globally.
- Qualitative inspection confirms under-segmentation on some defects, complete misses on others, and localized false positives on clean texture.
- The false-negative example has a non-trivial target area, so missed defects cannot be explained only by defects being extremely small.
- Lowest validation loss is not a reliable checkpoint-selection criterion for the final segmentation objective. Epoch 6 had slightly lower loss than epoch 9 but substantially worse Dice.

## Conclusion

The experiment supports the empty-mask hypothesis. Restricting Dice to non-empty target masks removed the zero-Dice collapse pattern and improved the best observed validation Dice, IoU and recall relative to EXP-010. The improvement is meaningful but does not fully stabilize optimization. The best-Dice checkpoint should therefore be the primary segmentation checkpoint, while validation loss remains useful as a training diagnostic rather than the sole model-selection criterion.

Threshold tuning provides a smaller additional gain and selects `0.90` for the epoch-9 checkpoint on validation data. Qualitative inspection shows that the remaining error is not primarily caused by one poorly chosen threshold: the model can segment some defects very accurately but alternates between partial coverage, complete misses and strong false activations on clean texture.

## Next action

Run a controlled learning-rate experiment while keeping the EXP-011 loss formulation, architecture, data split, seed, batch size, epoch count and thresholding procedure unchanged. Reduce Adam LR from `1e-3` to `3e-4` and compare training stability plus best validation Dice/IoU/precision/recall. If the lower LR merely slows learning without reducing instability, investigate longer training as a separate follow-up rather than changing both variables at once.

## MLflow run

Not tracked yet. MLflow will be introduced after the segmentation baseline is sufficiently stable for meaningful experiment comparison.

## Relevant commit

To be linked after EXP-011 qualitative analysis is finalized.
