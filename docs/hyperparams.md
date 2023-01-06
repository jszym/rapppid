# RAPPPID Hyperparameters

## Hyperparameter Tuning

When tuning the hyperparameters for RAPPPID, we searched along the following ranges for each value (Table S1 in [the paper](https://doi.org/10.1093/bioinformatics/btac429))


| Hyperparameter        | `train.py` Argument     | Range        | Increment |
|-----------------------|-------------------------|--------------|-----------|
| # LSTM Layers         | `rnn_num_layers`        | 2 - 3        | 1         |
| Embedding D/O Rate    | `embedding_droprate`    | 0.1-0.4      | 0.1       |
| LSTM D/O Rate         | `lstm_dropout_rate`     | 0.1-0.4      | 0.1       |
| Classifier D/O Rate   | `classhead_dropout_rate`| 0.1-0.4      | 0.1       |
| Learning Rate         | `lr`                    | 10⁻³ - 10⁻²  | 0.009     |


## Chosen Hyperparameters

The chosen hyperparameters for RAPPPID are as follows (Table S2 in [the paper](https://doi.org/10.1093/bioinformatics/btac429))

| Hyperparam / Experiment | C1  | C2  | C3  |
|-------------------------|-----|-----|-----|
| # LSTM Layers           | 3   | 2   | 2   |
| Embedding D/O Rate      | 0.1 | 0.3 | 0.3 |
| LSTM D/O Rate           | 0.1 | 0.3 | 0.3 |
| Classifier D/O Rate     | 0.1 | 0.2 | 0.2 |
| Learning Rate           | 10⁻²| 10⁻²| 10⁻²|

