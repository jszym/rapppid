import gzip
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as sp
import pytorch_lightning as pl

class RapppidDataset(Dataset):
    def __init__(self, rows, seqs, trunc_len=1000, vocab_size=2000):
        super().__init__()
        	
        self.rows = rows
        self.seqs = seqs
        self.trunc_len = trunc_len

        model_file = f'./sentencepiece_models/smp{vocab_size}.model'

        self.spp = sp.SentencePieceProcessor(model_file=model_file)
        self.aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U']


    def __getitem__(self, idx):

        p1, p2, label = self.rows[idx]

        p1_seq = "".join([self.aas[r] for r in self.seqs[p1][:self.trunc_len]])
        p1_seq = np.array(self.spp.encode(p1_seq, enable_sampling=True, alpha=0.1, nbest_size=-1))

        p2_seq = "".join([self.aas[r] for r in self.seqs[p2][:self.trunc_len]])
        p2_seq = np.array(self.spp.encode(p2_seq, enable_sampling=True, alpha=0.1, nbest_size=-1))

        p1_pad_len = self.trunc_len - len(p1_seq)
        p2_pad_len = self.trunc_len - len(p2_seq)

        p1_seq = np.pad(p1_seq, (0, p1_pad_len), 'constant')
        p2_seq = np.pad(p2_seq, (0, p2_pad_len), 'constant')

        p1_seq = torch.tensor(p1_seq).long()
        p2_seq = torch.tensor(p2_seq).long()
        label = torch.tensor(label).long()

        return (p1_seq, p2_seq, label)

    def __len__(self):
        return len(self.rows)

class RapppidDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, train_path: str, val_path: str, test_path: str, seqs_path: str, trunc_len: int, workers: int, vocab_size: int, seed: int):

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

        self.dataset_train = RapppidDataset(self.train_pairs, self.seqs, self.trunc_len, self.vocab_size)
        self.dataset_val = RapppidDataset(self.val_pairs, self.seqs, self.trunc_len, self.vocab_size)
        self.dataset_test = RapppidDataset(self.test_pairs, self.seqs, self.trunc_len, self.vocab_size)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, 
        num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, 
        num_workers=self.workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, 
        num_workers=self.workers, shuffle=False)
