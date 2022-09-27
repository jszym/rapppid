# Training RAPPPID Models

It's possible to train a RAPPPID model using the `train.py` utility.

### 1. Prepare your Data

You'll first need a dataset in a format that RAPPPID understands. There are two options.

1. If you wish to you use the datasets from the [Szymborski & Emad](https://doi.org/10.1101/2021.08.13.456309) manuscript, you can read the "Szymborski & Emad Datasets" heading in the [data.md](data.md) docs.

2. To prepare a new dataset, read the "Preparing RAPPPID Datasets" header in [data.md](data.md).

### 2. Generate SentencePiece Tokens

First begin by generating a SentencePiece vocabulary using `rapppid/train_seg.py`.
Set the `TRAIN_PATH`, `SEQ_PATH`, and `VOCAB_SIZE` constants to the desired 
values.

This script makes sure that the SentencePiece model is only trained on sequence present in the training dataset to ensure no data leakage.

### 3. Run the `train.py` utility

To train, validate, and test the model, run the `train.py` python file in the 
`rapppid` folder. `train.py` takes the following positional arguments:

* `batch_size: int` The training mini-batch size
* `train_path: Path` The path to the training files. RAPPPID training files can be found in the `data/rapppid` folder
* `val_path: Path` The path to the validation files. RAPPPID training files can be found in the `data/rapppid` folder
* `test_path: Path` The path to the testing files. RAPPPID training files can be found in the `data/rapppid` folder
* `seqs_path: Path` The path to the file containing protein sequences. RAPPPID protein sequences can be found in the `data/rapppid` folder
* `trunc_len: int` Sequences longer than the `trunc_len` will be truncated to this length.
* `embedding_size: int` The size of the token embeddings to use.
* `num_epochs: int` The maximum number of epochs to run. Testing will be run on the epoch with the lowest validation loss.
* `lstm_dropout_rate: float` The rate at which connections are dropped in the LSTM layers (aka DropConnect)
* `classhead_dropout_rate: float` The rate at which activates are dropped at the fully-connected classifier (aka Dropout)
* `rnn_num_layers: int` Number of LSTM layers to use
* `class_head_name: str` The kind of classifier head to use. Use `concat` to replicate the RAPPPID manuscript, other values are poorly supported.
* `variational_dropout: bool` Whether the DropConnect applied on the LSTM layers is done using variational dropout or not.
* `lr_scaing: bool` Whether or not to scale learning rate with sequence length. Set to `False` to replicate RAPPPID manuscript, other values are poorly supported.
* `model_file: str` Path to the SentencePiece .model file generated in Step 2
* `log_path: Path` Where to store logging files (saved weights, tensorboard files, hyper-parameters)
* `vocab_size: int` The size of the sentencepiece vocabulary. 
* `embedding_droprate: float` The rate at which embeddings are dropped (aka Embedding Dropout)
* `optimizer_type: str` The optimizer to use. Valid values are `ranger21` and `adam`. The former is best in our tests.
* `swa: bool` Enable Stochastic Weight Averaging.
* `seed: int` Seed to use for training.

### 4. Check the Output

The path in `log_path` contains model checkpoints, as well as logs, and evaluation metrics. You can monitor loss and various metrics live on tensorboard as well.