# Training RAPPPID Models

It's possible to train a RAPPPID model using the `train.py` utility.

### 1. Prepare your Data

You'll first need a dataset in a format that RAPPPID understands. There are two options.

1. If you wish to you use the datasets from the [Szymborski & Emad](https://doi.org/10.1101/2021.08.13.456309) manuscript, you can read the "Szymborski & Emad Datasets" heading in the [data.md](data.md) docs.

2. To prepare a new dataset, read the "Preparing RAPPPID Datasets" header in [data.md](data.md).

### 2. Generate SentencePiece Tokens

First begin by generating a SentencePiece vocabulary using `rapppid/train_seg.py`.

You can run this script from the CLI. 

```
Usage: train_seg.py [OPTIONS] SEQ_PATH TRAIN_PATH
```

Some more details:

* `SEQ_PATH`: is the location of the sequences Pickle file (see [data.md](data.md))
* `TRAIN_PATH`: is the location of the training pairs Pickle file (see [data.md](data.md))
* `seed: int`: Random seed for determinism.
* `vocab_size: int`: The size of the vocabulary to be generated.
    * **recommended value**: a value of `250` was used in the paper.

This script makes sure that the SentencePiece model is only trained on sequence present in the training dataset to ensure no data leakage.

### 3. Run the `train.py` utility

To train, validate, and test the model, run the `train.py` python file in the 
`rapppid` folder. `train.py` takes the following positional arguments:

* `batch_size: int` The training mini-batch size
    * **recommended value**: a value of `80` was used in the paper on a RTX 2080, with 32 CPU cores clocked at 2.2GHz.
* `train_path: Path` The path to the training files. RAPPPID training files can be found in the `data/rapppid` folder
* `val_path: Path` The path to the validation files. RAPPPID training files can be found in the `data/rapppid` folder
* `test_path: Path` The path to the testing files. RAPPPID training files can be found in the `data/rapppid` folder
* `seqs_path: Path` The path to the file containing protein sequences. RAPPPID protein sequences can be found in the `data/rapppid` folder
* `trunc_len: int` Sequences longer than the `trunc_len` will be truncated to this length.
    * **recommended value**: A value of `1500` was used in the paper, but values as large as `3000` and as small as `1000` have been used during development. A value of `3000` means almost all proteins won't be truncated, while `1500` still only truncates a small proportion of proteins. Larger values lead to vanishing gradients, so if training is unstable, this is a very good parameter to look at.
* `embedding_size: int` The size of the token embeddings to use. This also dictates the number of parameters in the LSTM cells.
    * **recommended value** A value of `64` was used in the paper. `32` has also worked well.
* `num_epochs: int` The maximum number of epochs to run. Testing will be run on the epoch with the lowest validation loss.
    * **recommended value** `train.py` will update the model checkpoints when the validation loss reaches a new low. So in the paper, we set the number of epochs to `100`, and reported the test metrics of the model with the lowest val loss (this is done automatically by `train.py`).
* `lstm_dropout_rate: float` The rate at which connections are dropped in the LSTM layers (aka DropConnect)
    * **recommended value** See [hyperparams.md](hyperparams.md). We tuned this hyperparameter and recommend you do so on new datasets as well.
* `classhead_dropout_rate: float` The rate at which activates are dropped at the fully-connected classifier (aka Dropout)
    * **recommended value** See [hyperparams.md](hyperparams.md). We tuned this hyperparameter and recommend you do so on new datasets as well.
* `rnn_num_layers: int` Number of LSTM layers to use
    * **recommended value** See [hyperparams.md](hyperparams.md). We tuned this hyperparameter and recommend you do so on new datasets as well.
* `class_head_name: str` The kind of classifier head to use. 
    * **recommended value** Use `concat` to replicate the RAPPPID manuscript. 
    * **Update:** We've found using `mult` provides similar performance, reduces the number of parameters, and more deterministic.
* `variational_dropout: bool` Whether the DropConnect applied on the LSTM layers is done using variational dropout or not.
    * **recommended value** `False`.
* `lr_scaing: bool` Whether or not to scale learning rate with sequence length. 
    * **recommended value** Set to `False` to replicate RAPPPID manuscript, other values are poorly supported.
* `model_file: str` Path to the SentencePiece model file generated in Step 2
* `log_path: Path` Where to store logging files (saved weights, tensorboard files, hyper-parameters)
    * **n.b.:** the directory in `log_path` must have the following directories below is:
        * `args`: The `args` folder will hold (in JSON files) all the hyperparameters, as well as the training, validation, and testing metrics. The most useful information is usually here.
        * `chkpts`: Pytorch Lightning Model Checkpoints are stored here. They hold both hyperparameters as well as model weights.
        * `tb_logs`: Holds the tensorboard logs
        * `onnx`: ONNX files are meant to be saved here, but serialization usually fails, so best to use the weights from `chkpts`.
        * `charts`: Quick ROC and PR charts for the testing dataset are generated for each trained model.
* `vocab_size: int` The size of the sentencepiece vocabulary. Use the value set in Step 2.
* `embedding_droprate: float` The rate at which embeddings are dropped (aka Embedding Dropout)
    * **recommended value** See [hyperparams.md](hyperparams.md). We tuned this hyperparameter and recommend you do so on new datasets as well.
* `transfer_path: str` If you wish to load weights that were pre-trained, include the checkpoint file from `/logs/chkpts/yourmodelname.chkpt`
* `optimizer_type: str` The optimizer to use. Valid values are `ranger21` and `adam`.
    * **recommended value** We use `ranger21` in the manuscript, but `adam` also works well.
* `swa: bool` Enable Stochastic Weight Averaging.
    * **recommended value** `True` in the manuscript.
* `seed: int` Seed to use for training.
    * We used [8675309](https://www.youtube.com/watch?v=boaJCrHNRMA), [5353457](https://www.youtube.com/watch?v=L776uPOJgB8), and [1234](https://www.youtube.com/watch?v=6pYMZKVZ9Ws) in the paper. LSTMs suffer from indeterminism when cuDNN is enabled, so while setting the seed helps towards replicable runs, there is always stochasticity in training the network.

### 4. Check the Output

The path in `log_path` contains model checkpoints, as well as logs, and evaluation metrics. You can monitor loss and various metrics live on tensorboard as well.

Training, validation, and testing metrics as well as hyperparameters are all present in the `args` folder in JSON format. Simply look for the file with your model name (unix timestamp + two random words).