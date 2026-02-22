# Training Overview

## Configuration
- validation_mode: holdout-kfold
- split_unit_requested: sample
- split_unit_effective: sample
- num_folds: 2
- test_size: 0.2

## Dataset
- total_samples: 200
- class_distribution: {'class_0': 52, 'class_1': 52, 'class_2': 52, 'class_3': 44}

## CV Aggregate
- accuracy: mean=0.2625, std=0.0000
- macro_precision: mean=0.0656, std=0.0000
- macro_recall: mean=0.2500, std=0.0000
- macro_f1: mean=0.1040, std=0.0000
- weighted_precision: mean=0.0689, std=0.0000
- weighted_recall: mean=0.2625, std=0.0000
- weighted_f1: mean=0.1092, std=0.0000
- macro_auc: mean=0.4832, std=0.0145
- micro_auc: mean=0.5093, std=0.0027
- weighted_auc: mean=0.4751, std=0.0157

## Holdout Metrics
- accuracy: 0.2250
- macro_precision: 0.0563
- macro_recall: 0.2500
- macro_f1: 0.0918
- weighted_precision: 0.0506
- weighted_recall: 0.2250
- weighted_f1: 0.0827
- macro_auc: 0.4879
- micro_auc: 0.4804
- weighted_auc: 0.4755
