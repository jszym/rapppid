import gzip
import pickle
import sentencepiece as spm

spm.set_random_generator_seed(8675309)

# Path to the seqs file
SEQS_PATH = '../data/9606/uniref_sequences_rapppid_subset[rapppid_proteins_[common_string_9606.protein.links.detailed.v11.5_uniref.csv]_8675309.json].pkl.gz'
VOCAB_SIZE = 250

with gzip.open(SEQS_PATH) as f:
    seqs = pickle.load(f)

aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U']

print('making seqs file')
with open('seqs.txt', 'w') as f:
    for pid in seqs:

        seq = seqs[pid]
        seq = "".join([aas[r] for r in seq])
        f.write(seq+'\n')

print('sentencepiece training')

spm.SentencePieceTrainer.train(input='seqs.txt', 
model_prefix=f'smp{VOCAB_SIZE}', 
vocab_size=VOCAB_SIZE,
character_coverage=1.,
bos_id=-1,
eos_id=-1,
pad_id=0,
unk_id=1,
)
