# Training Overview

## Configuration
- validation_mode: holdout-kfold
- split_unit_requested: clip
- split_unit_effective: clip
- augment_mode: none
- augment_factor: 0.0
- loss_function: categorical_crossentropy
- focal_alpha_mode: fixed
- focal_alpha_cap: 4.0
- focal_gamma: 2.0
- focal_alpha: 0.25
- num_folds: 5
- test_size: 0.2

## Dataset
- total_samples: 200
- class_distribution: {'class_0': 59, 'class_1': 40, 'class_2': 48, 'class_3': 53}

## CV Aggregate
- accuracy: mean=0.2875, std=0.0234
- macro_precision: mean=0.0723, std=0.0058
- macro_recall: mean=0.2450, std=0.0100
- macro_f1: mean=0.1115, std=0.0071
- weighted_precision: mean=0.0855, std=0.0134
- weighted_recall: mean=0.2875, std=0.0234
- weighted_f1: mean=0.1317, std=0.0184
- macro_auc: mean=0.6037, std=0.0243
- micro_auc: mean=0.5736, std=0.0248
- weighted_auc: mean=0.5801, std=0.0303

## Holdout Metrics
- accuracy: 0.3000
- macro_precision: 0.1472
- macro_recall: 0.2727
- macro_f1: 0.1692
- weighted_precision: 0.1619
- weighted_recall: 0.3000
- weighted_f1: 0.1861
- macro_auc: 0.4342
- micro_auc: 0.5067
- weighted_auc: 0.4329
