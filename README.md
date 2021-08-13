# RAPPPID

***R**egularised **A**utomative **P**rediction of **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

---

## Environment

The conda environment file (`environment.yml`) is available in the root of this
repository.


## Usage

First begin by generating a SentencePiece vocabulary using `rapppid/train_seg.py`.
Set the `TRAIN_PATH`, `SEQ_PATH`, and `VOCAB_SIZE` constants to the desired 
values.

To train, validate, and test the model, run the `cli.py` python file in the 
`rapppid` folder. `cli.py` takes the following positional arguments:

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
* `log_path: Path` Where to store logging files (saved weights, tensorboard files, hyper-parameters)
* `vocab_size: int` The size of the sentencepiece vocabulary. 
* `embedding_droprate: float` The rate at which embeddings are dropped (aka Embedding Dropout)
* `optimizer_type: str` The optimizer to use. Valid values are `ranger21` and `adam`. The former is best in our tests.
* `swa: bool` Enable Stochastic Weight Averaging.
* `seed: int` Seed to use for training.


## License

RAPPPID

***R**egularised **A**utomative **P**rediction of **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

Copyright (C) 2021  Joseph Szymborski

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.