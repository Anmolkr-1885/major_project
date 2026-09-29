# Alzheimer MRI classification - subject split

- images 6400, classes ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
- patients: {'method': 'slice-similarity graph (corr > 0.98)', 'n_groups': 200, 'group_size_min': 32, 'group_size_median': 32.0, 'group_size_max': 32, 'groups_with_mixed_labels': 0, 'nn_same_label_fraction': 1.0, 'nn_corr_median': 0.9937969446182251}
- patients shared by train and test: 0
- model: conxnet, 128px, seed 42

## Test, slice level

| model | accuracy | balanced_accuracy | f1_macro | roc_auc_ovr | binary_accuracy_demented_vs_non | binary_auc |
|---|---|---|---|---|---|---|
| CNN (ConXNet, selected) | 0.4870 | 0.2541 | 0.1894 | 0.6285 | 0.5114 | 0.7295 |

## Test, patient level (main model)

{
  "n": 41,
  "accuracy": 0.4878048780487805,
  "balanced_accuracy": 0.25,
  "precision_macro": 0.125,
  "recall_macro": 0.25,
  "f1_macro": 0.16666666666666666,
  "roc_auc_ovr": 0.6233796296296297,
  "binary_accuracy_demented_vs_non": 0.5121951219512195,
  "binary_sensitivity": 0.047619047619047616,
  "binary_specificity": 1.0,
  "binary_auc": 0.769047619047619
}

```
                  precision    recall  f1-score   support

     NonDemented      0.501     0.956     0.657       640
VeryMildDemented      0.300     0.060     0.100       448
    MildDemented      0.000     0.000     0.000       192
ModerateDemented      0.000     0.000     0.000        32

        accuracy                          0.487      1312
       macro avg      0.200     0.254     0.189      1312
    weighted avg      0.347     0.487     0.355      1312
```
