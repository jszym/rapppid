import torch
from torch import nn
import torch.nn.functional as F

# Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
# https://arxiv.org/abs/1908.08681v1
# implemented for PyTorch / FastAI by lessw2020 (Less Wright)
# github: https://github.com/lessw2020/mish
# https://github.com/lessw2020/mish/blob/master/mish.py
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))