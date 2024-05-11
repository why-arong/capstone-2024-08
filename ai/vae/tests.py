import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from spectralvae.dataset import TestDataset, ToTensor

import numpy as np
import librosa
from pathlib import Path
import soundfile as sf

def init_test_audio(workdir, sampling_rate, segment_length):
  # Create a set samples to test the network as it trains

  # Create a folder called reconstructions
  audio_log_dir = workdir / 'audio_logs'
  os.makedirs(audio_log_dir, exist_ok=True)

  with open("filelists/train.txt", "r", encoding="utf-8") as test_file:
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