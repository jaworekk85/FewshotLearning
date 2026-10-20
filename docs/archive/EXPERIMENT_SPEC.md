> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Experiment Spec Reconstructed From `access_v4.pdf`

This file maps the paper protocol to code. The goal is to make every paper-style single run reproducible while keeping the pieces reusable for new ablations.

## Directly Stated Protocol

- Datasets: CIFAR-10, Fashion-MNIST, SVHN, STL-10.
- Classes: 10 for every dataset.
- Image representation: 32 x 32 RGB tensors.
- Fashion-MNIST: grayscale images are replicated to RGB.
- STL-10: images are resized to 32 x 32.
- Validation split: 100 images per class are reserved from the official training split before label-budget sampling.
- Training budgets: `N in {1, 2, 5, 10, 20, 50, 100, 200}` images per class from the remaining official training images.
- Unused official training images are discarded.
- Test split: official test split.
- No pretrained weights, external data, pseudo-labels, unlabeled target data, semi-supervised learning, transduction, or base classes.
- Architectures: compact CNN and small ResNet.
- Embedding size: 128.
- Classifier: linear 10-class head after the embedding.
- Optimizer: Adam, learning rate `1e-3`, batch size `60`.
- Primary grid: 500 optimizer updates, 10 paired seeds.
- v4 controls: first 5 paired seeds.
- v4 schedules:
  - `CE` and `CE+Aug`: 2000 updates.
  - `CE+Aug-4V` and `CE+Aug-4V+L2`: 1000 updates.
- Pairing: same subset and initialization seed across methods for a dataset, architecture, budget, and seed.
- Weak augmentation:
  - reflected random crop,
  - horizontal flip except for SVHN.
- Stronger metric views:
  - weak augmentation,
  - contrast and brightness jitter,
  - additive noise.
- Metric tuples: anchors and positives from the same class, negatives from a different class, sampled only from the training subset after validation has been reserved.
- Analysis:
  - pooled paired differences by label budget,
  - wins,
  - two-sided exact sign test,
  - two-sided 95% percentile bootstrap confidence interval for paired mean,
  - 100,000 bootstrap resamples with a fixed analysis seed.

## Implemented Method Ladder

The method ids live in `configs/v4_protocol.json`. Treat these as presets: each can be repeated exactly or used as a starting point with command-line overrides.

- `ce_500`: CE, no augmentation, 500 updates.
- `ce_aug_500`: CE, one weakly augmented view, 500 updates.
- `ce_aug_4v_500`: CE averaged over four weakly augmented views, 500 updates.
- `ce_aug_4v_l2_500`: preceding method with L2-normalized embedding, 500 updates.
- `triplet_500`: CE plus triplet loss, strong views, no L2 normalization, 500 updates.
- `triplet_l2_500`: CE plus triplet loss, strong views, L2-normalized embedding, 500 updates.
- `supcon_joint_500`: CE plus supervised contrastive loss, two strong views, L2-normalized embedding, 500 updates.
- `ce_2000`: v4 long CE control.
- `ce_aug_2000`: v4 long CE+Aug control.
- `ce_aug_4v_1000`: v4 longer four-view CE control.
- `ce_aug_4v_l2_1000`: v4 longer four-view normalized CE control.

## Explicit Assumptions

The PDF does not specify every low-level hyperparameter needed for executable code. These defaults are intentionally centralized in `configs/v4_protocol.json`:

- Triplet margin: `0.2`.
- CE loss weight in joint objectives: `1.0`.
- Metric loss weight in joint objectives: `1.0`.
- SupCon temperature: `0.07`.
- Strong augmentation jitter:
  - brightness `0.25`,
  - contrast `0.25`,
  - Gaussian noise std `0.03`.
- CNN and ResNet exact layer layouts are practical small 32x32 encoders with 128-dimensional embeddings, not claimed to be the hidden original implementation.

Change these values in the config when matching an older internal codebase or running ablations.

## Composable Blocks

One run is defined by:

- dataset,
- architecture,
- label budget per class,
- seed,
- objective,
- augmentation mode,
- number of transformed views,
- embedding normalization,
- optimizer settings,
- loss settings.

The CLI supports both styles:

- select `--method <preset>` to repeat a named preset,
- select `--method <preset>` and override individual fields,
- omit `--method` and compose the method directly from flags.

This makes the original v3/v4 grids a reproducibility fixture rather than the only intended way to use the code.

## Leakage Checks

The code follows this ordering:

1. Load labels from the official training split.
2. Reserve `100` validation examples per class.
3. Sample `N` training examples per class from the remaining images.
4. Build metric tuples only from the budgeted training subset.
5. Evaluate on validation and official test splits without augmentation.

This matches the corrected v3/v4 protocol described in the paper and avoids the earlier exploratory leakage issue mentioned in the limitations.

## Optional Unlabeled Pool

The original paper protocol is label-only, so unused training images are discarded. For semi-supervised ablations, this code can instead expose those unused official training images as an unlabeled pool.

For a fixed budget `N`, the split becomes:

1. Reserve `100` validation examples per class.
2. Reserve `N` labeled training examples per class.
3. Put the remaining official training examples into `unlabeled`.
4. Never use validation or test images as unlabeled training data.

This preserves the labeled budget while changing the information setting. Such runs should be described as semi-supervised ablations, not as label-only reproductions.

Available unlabeled losses:

- `pseudo_ce`: predict a hard pseudo-label from one unlabeled view and train on the other view or views only when confidence exceeds `pseudo_label_threshold`.
- `consistency`: match the probability distribution from one unlabeled view across other augmented views.
- `simclr`: use multiple augmented views of the same unlabeled image as positives in an instance-contrastive loss.
- `pseudo_triplet`: use one unlabeled view as anchor, another view of the same image as positive, and either a random or pseudo-label-based batch item as negative.

Triplet loss itself still uses labeled data only unless you intentionally implement pseudo-labeled triplet mining. Without labels or pseudo-labels, an unlabeled example cannot define class-positive and class-negative pairs.

For `pseudo_triplet`, the unlabeled batch is separate from the labeled batch. Given unlabeled images `u_i`, the transform returns at least two views:

- `a_i = t_1(u_i)`,
- `p_i = t_2(u_i)`.

The model predicts pseudo-labels from the anchor view:

- `q_i = softmax(f_theta(a_i))`,
- `yhat_i = argmax q_i`,
- `c_i = max q_i`.

The negative is chosen within the same unlabeled batch. Three strategies are implemented:

- `random`: sample `j(i) != i` uniformly from the unlabeled batch. With roughly balanced 10-class data, this has about 90% probability of selecting a true different-class negative.
- `pseudo_label_hard`: use only confident pseudo-labels, then choose `j(i) = argmin_j d(z(a_i), z(a_j))` subject to `j != i`, `c_i >= threshold`, `c_j >= threshold`, and `yhat_j != yhat_i`.
- `pseudo_label_or_random`: use `pseudo_label_hard` when a candidate exists, otherwise fall back to `random`.

Then:

- `L_u = mean_i max(d(z(a_i), z(p_i)) - d(z(a_i), z(a_j(i))) + margin_u, 0)`.

The full objective is still:

- `L_total = L_labeled + unlabeled_weight * L_u`.
