#  custom Torch Dataset objects for each dataset
import pandas as pd 
import torch 
import torchaudio 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt     




# Global variables 

SAMPLE_RATE = 16000
N_FFT = int(0.025 * SAMPLE_RATE)   # 25 ms 
HOP_LENGTH = int(0.01 * SAMPLE_RATE)  # 10 ms
N_MELS = 13


# Dataset Paths 

# ASVP-ESD 
ASVP_dir = '/Users/stephen/Desktop/Speech_Recognition/Data/ASVP-ESD_UPDATE/Audio/'
ASVP_metadata = '/Users/stephen/Desktop/Speech_Recognition/Data/ASVP-ESD_UPDATE/metadata.csv'

# Google Speech Commands 

# VOicES 

# Libspech 






##### DATA CLASSES ######


# ASVP Dataset class 

class ASVPDataset(Dataset):
    # constructor 
    def __init__(self, annotations_file, audio_dir, target_sample_rate, num_samples, device, transformation=None): 
        self.annotations_file = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir 
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples 
        self.device = device 
        self.transformation = transformation

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        # if a transformation is stated, apply to audio tensor 
        if self.transformation: 
            signal = self.transformation(signal).to(self.device)
        return signal, label 

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal 

    def _right_pad_if_necessary(self, signal): 
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:    # compare each file with desired length 
            num_missing_samples = self.num_samples - length_signal 
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal 

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal 

    def _mix_down_if_necessary(self, signal):  # if file is Stereo, convert to mono by using the mean 
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal 

    def _get_audio_sample_path(self, index):
        #fold = f"fold{self.annotations.iloc[index, 2]}"    # grab from "Folder" column 
        #path = os.path.join(self.audio_dir, fold, self.annotations_file.iloc[index, 1]) # combine to make full PATH
        fold = self.annotations_file.iloc[index, 2]
        path = os.path.join(self.audio_dir + fold + self.annotations_file.iloc[index, 1])
        return path 
    
    def _get_audio_sample_label(self, index):
        return self.annotations_file.iloc[index, 0]  # grab from "label" column  


if __name__ == '__main__':
    ANNOTATIONS_FILE = ASVP_metadata
    AUDIO_DIR = ASVP_dir
    TARGET_SAMPLE_RATE = SAMPLE_RATE
    NUM_SAMPLES = SAMPLE_RATE*20

    if torch.cuda.is_available():
        device - "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    asvp = ASVPDataset(
        ANNOTATIONS_FILE, 
        AUDIO_DIR, 
        TARGET_SAMPLE_RATE, 
        NUM_SAMPLES, 
        device)


    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]