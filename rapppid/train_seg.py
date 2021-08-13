import gzip
import pickle
import sentencepiece as spm

spm.set_random_generator_seed(5353456)

TRAIN_PATH = 'train_pairs.pkl.gz'
SEQS_PATH = 'seqs.pkl.gz'
VOCAB_SIZE = 250

with gzip.open(TRAIN_PATH) as f:
    train_pairs = pickle.load(f)

with gzip.open(SEQS_PATH) as f:
    seqs = pickle.load(f)

aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U']

print('making seqs file')
with open('seqs.txt', 'w') as f:
    for prot_id_a, prot_id_b, y in train_pairs:

        seq = seqs[prot_id_a]
        seq = "".join([aas[r] for r in seq])
        f.write(seq+'\n')

        seq = seqs[prot_id_b]
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
