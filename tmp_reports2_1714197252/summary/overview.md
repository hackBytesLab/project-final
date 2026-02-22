# Training Overview

## Configuration
- validation_mode: holdout-kfold
- split_unit_requested: clip
- split_unit_effective: clip
- num_folds: 2
- test_size: 0.2

## Dataset
- total_samples: 200
- class_distribution: {'class_0': 56, 'class_1': 47, 'class_2': 47, 'class_3': 50}

## CV Aggregate
- accuracy: mean=0.2500, std=0.0125
- macro_precision: mean=0.0625, std=0.0031
- macro_recall: mean=0.2500, std=0.0000
- macro_f1: mean=0.1000, std=0.0040
- weighted_precision: mean=0.0627, std=0.0062
- weighted_recall: mean=0.2500, std=0.0125
- weighted_f1: mean=0.1002, std=0.0090
- macro_auc: mean=0.4410, std=0.0234
- micro_auc: mean=0.4942, std=0.0081
- weighted_auc: mean=0.4305, std=0.0193

## Holdout Metrics
- accuracy: 0.3000
- macro_precision: 0.0750
- macro_recall: 0.2500
- macro_f1: 0.1154
- weighted_precision: 0.0900
- weighted_recall: 0.3000
- weighted_f1: 0.1385
- macro_auc: 0.3334
- micro_auc: 0.4790
- weighted_auc: 0.3143
