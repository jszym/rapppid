# Pre-Trained Weights

This folder contains pre-trained weights for RAPPPID. Each model has a randomly generated name with a time component.

## 1690837077.519848_red-dreamy

This model was trained on the _H. sapiens_ comparatives dataset. Hyperparameters are identical to the ones used in the RAPPPID paper, except the `class_head_name` is set to `mult` rather than `concat`. This doesn't change the performance but helps with order invariance in the inference step.

### Testing Metrics

|AUROC |AUPR |Accuracy |
|------|-----|---------|
|0.801 |0.805|0.710    |
