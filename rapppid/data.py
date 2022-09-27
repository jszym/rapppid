from typing import List
import gzip
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as sp
import pytorch_lightning as pl

class RapppidDataset(Dataset):
    def __init__(self, rows, seqs, model_file, trunc_len=1000, vocab_size=2000):
        super().__init__()
        	
        self.rows = rows
        self.seqs = seqs
        self.trunc_len = trunc_len

        self.spp = sp.SentencePieceProcessor(model_file=model_file)

    @staticmethod
    def static_encode(trunc_len: int, spp, seq: str, sp: bool = True, pad: bool = True):
        aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U']

        toks = "".join([aas[r] for r in seq[:trunc_len]])
        
        if sp:
                toks = np.array(spp.encode(toks, enable_sampling=True, alpha=0.1, nbest_size=-1))
        if pad:
             pad_len = trunc_len - len(toks)
             toks = np.pad(toks, (0, pad_len), 'constant')

        return toks

    def encode(self, seq: str, sp: bool = True, pad: bool = True):

        aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U']

        toks = "".join([aas[r] for r in seq[:self.trunc_len]])
        
        if sp:
                toks = np.array(self.spp.encode(toks, enable_sampling=True, alpha=0.1, nbest_size=-1))
        if pad:
             pad_len = self.trunc_len - len(toks)
             toks = np.pad(toks, (0, pad_len), 'constant')

        return toks

    def __getitem__(self, idx):

        p1, p2, label = self.rows[idx]

        p1_seq = self.encode(self.seqs[p1], sp=True, pad=True)

        p2_seq = self.encode(self.seqs[p2], sp=True, pad=True)

        p1_seq = torch.tensor(p1_seq).long()
        p2_seq = torch.tensor(p2_seq).long()
        label = torch.tensor(label).long()

        return (p1_seq, p2_seq, label)

    def __len__(self):
        return len(self.rows)

class RapppidDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, train_path: str, val_path: str, test_path: str, seqs_path: str, trunc_len: int, workers: int, vocab_size: int, model_file: str, seed: int):

        super().__init__()
        
        sp.set_random_generator_seed(seed)
        
        self.batch_size = batch_size
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.seqs_path = seqs_path
        self.vocab_size = vocab_size

        self.dataset_train = None
        self.dataset_test = None
        
        self.trunc_len = trunc_len
        self.workers = workers

        self.model_file = model_file
        
        self.train = []
        self.test = []
        self.seqs = []

    def setup(self, stage=None):

        with gzip.open(self.seqs_path) as f:
            self.seqs = pickle.load(f)

        with gzip.open(self.test_path) as f:
            self.test_pairs = pickle.load(f)

        with gzip.open(self.val_path) as f:
            self.val_pairs = pickle.load(f)  

        with gzip.open(self.train_path) as f:
            self.train_pairs = pickle.load(f)  

        self.dataset_train = RapppidDataset(self.train_pairs, self.seqs, self.model_file, self.trunc_len, self.vocab_size)
        self.dataset_val = RapppidDataset(self.val_pairs, self.seqs, self.model_file, self.trunc_len, self.vocab_size)
        self.dataset_test = RapppidDataset(self.test_pairs, self.seqs, self.model_file, self.trunc_len, self.vocab_size)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, 
        num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, 
        num_workers=self.workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, 
        num_workers=self.workers, shuffle=False)
