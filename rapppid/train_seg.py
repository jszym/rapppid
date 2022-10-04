import click
import gzip
import pickle
import sentencepiece as spm
import os


@click.command()
@click.option('--seed', default=8675309, help='Random seed to use.')
@click.option('--vocab_size', default=250, help='The size of the SentencePiece vocab.')
@click.argument('seq_path')
@click.argument('train_path')
def seq_train(seed, vocab_size, seq_path, train_path):
    spm.set_random_generator_seed(seed)

    with gzip.open(train_path) as f:
        pairs = pickle.load(f)

    proteins = set()

    for pair in pairs:
        p1, p2, y = pair
        proteins.add(p1)
        proteins.add(p2)

    with gzip.open(seq_path) as f:
        seqs = pickle.load(f)

    aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
           'O', 'U']

    with open('seqs.txt', 'w') as f:
        for pid in seqs:
            if pid in proteins:
                seq = seqs[pid]
                seq = "".join([aas[r] for r in seq])
                f.write(seq + '\n')

    spm.SentencePieceTrainer.train(input='seqs.txt',
                                   model_prefix=f'smp{vocab_size}',
                                   vocab_size=vocab_size,
                                   character_coverage=1.,
                                   bos_id=-1,
                                   eos_id=-1,
                                   pad_id=0,
                                   unk_id=1,
                                   )

    os.unlink('seqs.txt')


if __name__ == '__main__':
    seq_train()
