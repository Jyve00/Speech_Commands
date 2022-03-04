# recreating Librispeechd dataset object from source code 
# source code: https://pytorch.org/audio/stable/_modules/torchaudio/datasets/librispeech.html



import os 
import torchaudio
from torch import Tensor 
from torch.utils.data import Dataset  
