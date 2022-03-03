import torch 
import torchaudio 
torch.nn as nn
import numpy as np 
import numpy as np as np


# class for melspec 

class LogMelSpec(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, win_length=win_length,
            n_mels-n_mels, hop_length=hop_length
        )
    
    def forward(self. x):
        x = self.transform(x)





