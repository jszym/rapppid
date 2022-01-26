## Infer with RAPPPID

```python
from infer import *

seed = 8675309
pl_seed.seed_everything(seed, workers=True)

# Path to the RAPPPID chkpt
chkpt_path = '/path/to/chkpt/1627273298.6560657_unbent-curse.ckpt'
model = load_chkpt(chkpt_path)

# Path to the SentencePiece Model
model_file = '/path/to/spm/250.model'
spp = sp.SentencePieceProcessor(model_file=model_file)

seqs = [
    'LVYTDCTESGQNLCLCEGSNVCGQGNKCILGSDGEKNQCVTGEGTPKPQSHNDGDFEEIPEEYLQ',
    'QVQLKQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSINKDNSKSQVFFKMNSLQSNDTAIYYCARALTYYDYEFAYWGQGTLVTVSAASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSPKSCDKTHTCPPCPAPELLGGP'
]

toks = process_seqs(spp, seqs, 1500)

out = model(toks)
embedding_one = out[0].unsqueeze(0)
embedding_two = out[1].unsqueeze(0)
print(predict(model, embedding_one, embedding_two).item())
```