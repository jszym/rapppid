from train import LSTMAWD
from pathlib import Path
from typing import Optional

import sentencepiece as sp
import numpy as np
import torch
from pytorch_lightning.utilities import seed as pl_seed

def load_chkpt(chkpt_path: Path):
    return LSTMAWD.load_from_checkpoint(chkpt_path)

def encode_seq(spp, seq, trunc_len: Optional[int]):
    toks = spp.encode(seq, enable_sampling=False, alpha=0.1, nbest_size=-1)
    
    if trunc_len:
        pad_len = trunc_len - len(toks)
        toks = np.pad(toks, (0, pad_len), 'constant')
        
    return torch.tensor(toks).long()

def process_seqs(spp, input_seqs, trunc_len):
    processed_seqs = []
    for input_seq in input_seqs:
        processed_seqs.append(encode_seq(spp, input_seq, trunc_len))

    return torch.vstack(processed_seqs)

def get_embeddings(model, input_seqs):
    return model(input_seqs)

def predict(model, embedding_one, embedding_two):
    logit = model.class_head(embedding_one, embedding_two).float()
    return torch.sigmoid(logit)