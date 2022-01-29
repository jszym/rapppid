# RAPPPID Data

## Szymborski & Emad Datasets

[Szymborski & Emad](https://doi.org/10.1101/2021.08.13.456309) describes a C-type dataset based on STRING *H. sapiens* protein pairs, and randomly sampled negative examples. For details, refer to the manuscript.

The data can be downloaded from the Internet Archive at [this link](https://archive.org/details/rapppid_dataset), from [this mirror](https://dl.sphericalcow.xyz/rapppid/rapppid_dataset.tar.gz) hosted at OVH, or using the Academic Torrents [BitTorrent file](https://academictorrents.com/details/34079b029c6a8230f196593164e3fab8956e9ee5).

You can also download the datasets used for the BioLip transfer learning analysis [at the Internet Archive](https://archive.org/details/rapppid_transfer_learning_dataset), from [this mirror](https://dl.sphericalcow.xyz/rapppid/rapppid_transfer_learning_dataset.zip) hosted at OVH, or using [this torrent](https://archive.org/download/rapppid_transfer_learning_dataset/rapppid_transfer_learning_dataset_archive.torrent).

## Prepare Datasets for RAPPPID

RAPPPID expects four files when training a new dataset.

1. `train.pkl.gz` - Contains training pairs of protein IDs and a binary interaction label.
2. `val.pkl.gz` - Contains validation pairs of protein IDs and a binary interaction label.
3. `test.pkl.gz` - Contains testing pairs of protein IDs and a binary interaction label.
4. `seqs.pkl.gz` - Which is a map between protein IDs and their amino acid sequences.

Each of these files are expected to be a [gzipped](https://docs.python.org/3/library/gzip.html) [pickle](https://docs.python.org/3/library/pickle.html) file.

`{train, val, test}.pkl.gz` are pickled arrays of 3-tuples (tripels). The first two elements are two protein ids, and the last element is either a 1 (known interaction), or a 0 (no known interaction).

Here's an example of python code that creates a valid `train.pkl.gz` comprised of two pairs. `val.pkl.gz` and `test.pkl.gz` are created identically.

```python
import gzip
import pickle

train_pairs = [
    #(protein_id_a, protein_id_b, label)
    ("PROTEIN_ID_ONE", "PROTEIN_ID_TWO", 1),
    ("PROTEIN_ID_THREE", "PROTEIN_ID_FOUR", 0)
]


with gzip.open('train.pkl.gz', mode='wb') as f:
    pickle.dump(train_pairs, f)
```

The `seqs.pkl.gz` file is created similarly. A dictionary of protein ID keys and encoded amino acid sequences. A function below is provided to encode the amino acid sequences. 

Special amino acid codes `X`, `Z`, `B` are replaced with a randomly selected amino acid from the pool the represent. See the `encode_seq` function for more details.

```python
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

seqs = {
    # protein_id: encoded amino acid seqs
    'PROTEIN_ID_ONE': encode_seq('MANL'),
    'PROTEIN_ID_TWO': encode_seq('MANG'),
    'PROTEIN_ID_THREE': encode_seq('MANW'),
    'PROTEIN_ID_FOUR': encode_seq('MANA'),
}

with gzip.open('seqs.pkl.gz', mode='wb') as f:
    pickle.dump(seqs, f)
```

With the above examples, you should be able to create your own datasets quite easily for RAPPPID.