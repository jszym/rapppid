# Transfer Learning
Here are some instructions on how to use RAPPPID for transfer-learning.

## 1. Pre-training
Run RAPPPID as you normally would using the instructions in `train.md`.

Be sure to take note of the randomly generated model name 
(it consists of two words and a timestamp).

## 2. Transfer Learning
Again, follow the instructions in `train.md` for your second dataset,
but this time use the `transfer_path` flag to load the pre-trained weights.

The weights are going to be in `./logs/chkpts/model_name.chkpt` where
`model_name` is the randomly generated model name from Step 1.
