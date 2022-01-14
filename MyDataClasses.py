#  custom Torch Dataset objects for each dataset

import torch 
import torchaudio 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt     




# Global variables 

SAMPLE_RATE = 16000
N_FFT = int(0.025 * SAMPLE_RATE)   # 25 ms 
HOP_LENGTH = int(0.01 * SAMPLE_RATE)
N_MELS = 13
