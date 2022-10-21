from pathlib import Path
import gzip
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as sp
import pytorch_lightning as pl
import tables as tb


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

def get_aa_code(aa):
    # Codes based on IUPAC-IUB
    # https://web.expasy.org/docs/userman.html#AA_codes

    aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U']
    wobble_aas = {
        'B': ['D', 'N'],
        'Z': ['Q', 'E'],
        'X': ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    }

    if aa in aas:
        return aas.index(aa)

    elif aa in ['B', 'Z', 'X']:
        # Wobble
        idx = randint(0, len(wobble_aas[aa])-1)
        return aas.index(wobble_aas[aa][idx])


def encode_seq(seq):
    return [get_aa_code(aa) for aa in seq]


class RapppidDataset2(Dataset):
    def __init__(self, dataset_path, c_type, split, model_file, trunc_len=1000):
        super().__init__()

        self.trunc_len = trunc_len
        self.dataset_path = dataset_path
        self.c_type = c_type
        self.split = split

        self.spp = sp.SentencePieceProcessor(model_file=model_file)
        

    @staticmethod
    def static_encode(trunc_len: int, spp, seq: str, sp: bool = True, pad: bool = True):
        
        seq = seq[:trunc_len]

        if sp:
            toks = np.array(spp.encode(seq, enable_sampling=True, alpha=0.1, nbest_size=-1))
        else:
            toks = encode_seq(seq)

        if pad:
            pad_len = trunc_len - len(toks)
            toks = np.pad(toks, (0, pad_len), 'constant')

        return toks

    def encode(self, seq: str, sp: bool = True, pad: bool = True):

        return self.static_encode(self.trunc_len, self.spp, seq, sp, pad)

    def get_sequence(self, name: str):
        dataset = tb.open_file(self.dataset_path)
        return dataset.root.sequences.read_where(f'name=="{name}"')[0][1].decode('utf8')

    def __getitem__(self, idx):

        dataset = tb.open_file(self.dataset_path)

        p1, p2, label = dataset.root['interactions'][f'c{self.c_type}'][f'c{self.c_type}_{self.split}'][idx]

        p1 = p1.decode('utf8')
        p2 = p2.decode('utf8')

        p1_seq = self.encode(self.get_sequence(p1), sp=True, pad=True)

        p2_seq = self.encode(self.get_sequence(p2), sp=True, pad=True)

        p1_seq = torch.tensor(p1_seq).long()
        p2_seq = torch.tensor(p2_seq).long()
        label = torch.tensor(label).long()

        return (p1_seq, p2_seq, label)

    def __len__(self):
        dataset = tb.open_file(self.dataset_path)
        l = len(dataset.root['interactions'][f'c{self.c_type}'][f'c{self.c_type}_{self.split}'])
        return l


class RapppidDataModule2(pl.LightningDataModule):

    def __init__(self, batch_size: int, dataset_path: Path, c_type: int, trunc_len: int, workers: int, vocab_size: int,
                 model_file: str, seed: int):
        super().__init__()

        sp.set_random_generator_seed(seed)

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size

        self.dataset_train = None
        self.dataset_test = None

        self.trunc_len = trunc_len
        self.workers = workers

        self.model_file = model_file
        self.c_type = c_type

        self.train = []
        self.test = []
        self.seqs = []

    def setup(self, stage=None):
        self.dataset_train = RapppidDataset2(self.dataset_path, self.c_type, 'train', self.model_file, self.trunc_len)
        self.dataset_val = RapppidDataset2(self.dataset_path, self.c_type, 'val', self.model_file, self.trunc_len)
        self.dataset_test = RapppidDataset2(self.dataset_path, self.c_type, 'test', self.model_file, self.trunc_len)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          num_workers=self.workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          num_workers=self.workers, shuffle=False)

class RapppidDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, train_path: Path, val_path: Path, test_path: Path, seqs_path: Path,
                 trunc_len: int, workers: int, vocab_size: int, model_file: str, seed: int):

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
