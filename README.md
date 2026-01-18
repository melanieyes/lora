# Limitations of LoRA Observed in Digit Classification

This project demonstrates a key limitation of LoRA when applied to classification tasks with imbalanced or single-class fine-tuning: catastrophic degradation in unrelated classes. While LoRA excels in parameter-efficient adaptation for large models, its low-rank constraint can amplify biases in narrow training distributions.

## Core Issue: Single-Class Fine-Tuning Bias
- Fine-tuned only on digit 9 batches (100 batches), ignoring other classes during LoRA updates.
- Result: Digit 9 errors drop sharply (81 → 14 wrong counts), but digits like 4 explode (24 → 112), 1 (28 → 71), and 3 (64 → 90).
- Why? Low-rank ΔW = BA shares directions across all classes; optimizing solely for 9 distorts boundaries for visually similar digits (e.g., 4's loops overlap 9's in feature space).

## Low-Rank Subspace Constraints
- LoRA rank r (e.g., 8-16 here) limits updates to r × (in + out) params vs. full in × out, efficient (~9% param increase) but insufficient for multi-class rebalancing.
- Zero-init B + Gaussian A + α = r scaling follows the paper, yet still causes collateral harm without full-class data.
- Applied to final linear layers only—good for efficiency, but amplifies logit warping without earlier feature regularization.

## Comparison of Accuracies
| Model          | Overall Accuracy | Digit 4 Errors | Digit 9 Errors | Param Increase |
|----------------|------------------|----------------|---------------|---------------|
| Baseline      | 0.962           | 24             | 81            | 0%           |
| LoRA (9-only) | 0.941           | 112            | 14            | ~9%          |

## Broader Implications
- Not suitable for extreme task shifts without safeguards: LoRA assumes downstream task aligns with pretrained capabilities; single-class ignores class interactions.
- Overfitting risk high in few-shot/low-data regimes (here, just 100 batches); no shown validation halts or L2 reg exacerbate it.
- Visual similarity amplifies harm: Digits 4/9/0 share curves; LoRA's subspace can't decouple them without higher rank or multi-class sampling.

## Recommendations to Mitigate
- Train on full dataset or balanced subsets (e.g., oversample weak classes like 4).
- Use higher rank (r > 32), tuned α, or QLoRA variant for quantization efficiency.
- Add class-aware losses (focal loss) or orthogonal init for A.
- Monitor per-class metrics during training, not just overall accuracy.
