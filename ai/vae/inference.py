import torch
from torch.utils.data import DataLoader

import torchaudio
Resample = torchaudio.transforms.Resample(44100, 48000, resampling_method='kaiser_window')

if torch.cuda.is_available():
    device = 'cuda:0'
    my_cuda = 1
else:
    device = 'cpu'
    my_cuda = 0

Resample = Resample.to(device)

from pathlib import Path

import librosa
import librosa.display

import IPython.display as display

from model import VAE
from dataset import TestDataset, ToTensor


sampling_rate = 44100
sr = sampling_rate

hop_length = 128

segment_length = 1024
n_units = 256
latent_dim = 8

batch_size = 256


state = torch.load(Path(r'ai/vae/model/ckpt'), map_location=torch.device(device))

if my_cuda:
    model = VAE(segment_length, n_units, latent_dim).to(device)
else:
    model = VAE(segment_length, n_units, latent_dim)

model.load_state_dict(state['state_dict'])
model.eval()


test_audio_path = ""
test_audio, fs = librosa.load(test_audio_path, sr=None)

display.Audio(test_audio, rate=fs)


test_dataset = TestDataset(test_audio, segment_length = segment_length, sampling_rate = sampling_rate, transform=ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)


def raw_to_z_dist(test_dataloader, raw_model, device):
    init_test = True
    for iterno, test_sample in enumerate(test_dataloader):
        with torch.no_grad():
            test_sample = test_sample.to(device)
            test_mu, test_logvar = raw_model.encode(test_sample)

        if init_test:
            test_z_mu = test_mu
            test_z_logvar = test_logvar
            init_test = False

        else:
            test_z_mu = torch.cat((test_z_mu, test_mu ),0)
            test_z_logvar = torch.cat((test_z_logvar, test_logvar ),0)
    return test_z_mu, test_z_logvar

test1_z_mu, test1_z_logvar = raw_to_z_dist(test_dataloader, model, device)
