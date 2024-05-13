import os
import torch
from torch.utils.data import DataLoader

from dataset import AudioDataset, TestDataset, ToTensor

import numpy as np
import librosa
import soundfile as sf

def init_test_audio(workdir, sampling_rate, segment_length):
  # Create a set samples to test the network as it trains

  # Create a folder called reconstructions
  audio_log_dir = workdir / 'audio_logs'
  os.makedirs(audio_log_dir, exist_ok=True)

  with open("filelists/test.txt", "r", encoding="utf-8") as test_file:
      test_files = test_file.read().splitlines()

  init = True
  for test in test_files:
      
    audio_full, _ = librosa.load(test, sr=sampling_rate)

    if init:
      test_dataset_audio = audio_full
      init = False
    else:
      test_dataset_audio = np.concatenate((test_dataset_audio, audio_full ),axis=0)
  
  # Create a dataloader for test dataset
  test_dataset = TestDataset(test_dataset_audio, segment_length = segment_length, sampling_rate = sampling_rate, transform=ToTensor())
  
  sf.write(audio_log_dir.joinpath('test_original.wav'), test_dataset_audio, sampling_rate)
  return test_dataset, audio_log_dir



def create_dataset(file_path, segment_length, sampling_rate, hop_length, batch_size):
    with open(file_path, "r", encoding="utf-8") as file:
        files = file.read().splitlines()
    
    audio_array = []
    new_loop = True

    for f in files: 
        new_array, _ = librosa.load(f, sr=sampling_rate)

        if new_loop:
            audio_array = new_array
            new_loop = False
        else:
            audio_array = np.concatenate((audio_array, new_array), axis=0)

    total_frames = len(audio_array) // segment_length
    print('Total number of audio frames: {}'.format(total_frames))

    dataset = AudioDataset(audio_array, segment_length=segment_length, sampling_rate=sampling_rate, hop_size=hop_length, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, len(dataset)


def generate_latent_data(model, dataloader, device='cuda:0'):
    model.eval()
    latent_data = []

    with torch.no_grad():
        for data in dataloader:
            _, mu, _ = model.forward(data.to(device))
            latent_data.append(mu.cpu().detach())

    latent_data = torch.cat(latent_data, dim=0)
    return latent_data.numpy()
