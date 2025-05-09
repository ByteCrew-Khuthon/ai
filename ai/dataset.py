import os
import torch
import torchaudio
import torchaudio.transforms as T
import random
from torch.utils.data import Dataset

class AudioMelDataset(Dataset):
    def __init__(self, file_list, sample_rate=44100, duration=3, num_mels=64):
        self.file_list = file_list
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.num_mels = num_mels

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=num_mels
        )
        self.to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        waveform, sr = torchaudio.load(path)

        # Resample
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        original_len = waveform.shape[1]

        # Random crop or random pad
        if original_len > self.target_length:
            start = random.randint(0, original_len - self.target_length)
            waveform = waveform[:, start:start + self.target_length]
        elif original_len < self.target_length:
            pad_total = self.target_length - original_len
            pad_left = random.randint(0, pad_total)
            pad_right = pad_total - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))

        # Mel Spectrogram & log scale
        mel = self.mel_transform(waveform)  # (1, n_mels, T)
        mel_db = self.to_db(mel)
        mel_db = mel_db.squeeze(0)[:, :-1]
        return mel_db
