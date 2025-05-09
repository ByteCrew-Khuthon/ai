import sounddevice as sd
import numpy as np
import time
import random
import torchaudio.transforms as T
import torch
import torchaudio

S_RATE = 44100
DURATION = 0.5

mel = T.MelSpectrogram(
        sample_rate = S_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
        )
to_db = T.AmplitudeToDB()

def get_mel_spec(audio_data) :
    mel_spec = mel(audio_data)
    mel_spec_db = to_db(mel_spec)
    mel_spec_db = (mel_spec_db + 80) / 80.0
    return mel_spec_db


def process_audio() :
    audio = sd.rec(int(DURATION * S_RATE), samplerate=S_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = torch.from_numpy(audio).float().squeeze().unsqueeze(0)
    mel_spec_db = get_mel_spec(audio)
    return mel_spec_db

def get_wav_spec(filename) :
    waveform, sr = torchaudio.load(filename)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    original_len = waveform.shape[1]

    pad_total = int(S_RATE * DURATION) - original_len
    pad_left = random.randint(0, pad_total)
    pad_right = pad_total - pad_left
    waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
    return get_mel_spec(waveform)


if __name__ == '__main__' :
    mel = process_audio()
    print(mel.shape)
