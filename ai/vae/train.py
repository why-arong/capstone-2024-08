# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import DataLoader

from torch import optim

from model import VAE, loss_function
from tests import init_test_audio
from dataset import AudioDataset, ToTensor

import numpy as np

import os, sys, argparse, time
from pathlib import Path

import librosa
import soundfile as sf
import configparser

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
args = parser.parse_args()


# Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
try: 
  config.read(config_path)
except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()


# Import audio configs 
sampling_rate = config['audio'].getint('sampling_rate')
hop_length = config['audio'].getint('hop_length')
segment_length = config['audio'].getint('segment_length')


# Dataset
dataset = Path(config['dataset'].get('data'))

run_number = config['dataset'].getint('run_number')

generate_test = config['dataset'].get('generate_test')    


# Training configs
epochs = config['training'].getint('epochs')
learning_rate = config['training'].getfloat('learning_rate')
batch_size = config['training'].getint('batch_size')
checkpoint_interval = config['training'].getint('checkpoint_interval')
save_best_model_after = config['training'].getint('save_best_model_after')


# Model configs
latent_dim = config['VAE'].getint('latent_dim')
n_units = config['VAE'].getint('n_units')
n_hidden_units = config['VAE'].getint('n_hidden_units')
kl_beta = config['VAE'].getfloat('kl_beta')
device = config['VAE'].get('device')


# etc
example_length = config['extra'].getint('example_length')
normalize_examples = config['extra'].getboolean('normalize_examples')
plot_model = config['extra'].getboolean('plot_model')

start_time = time.time()
config['extra']['start'] = time.asctime( time.localtime(start_time) )

device = torch.device(device)
device_name = torch.cuda.get_device_name()
print('Device: {}'.format(device_name))
config['VAE']['device_name'] = device_name


# Create workspace
run_id = run_number
while True:
    try:
        run = Path('run')
        workdir = run / '{:03d}'.format(run_id)
        os.makedirs(workdir)

        break
    except OSError:
        if workdir.is_dir():
            run_id = run_id + 1
            continue
        raise

config['dataset']['workspace'] = str(workdir.resolve())

print("Workspace: {}".format(workdir))


# Create the dataset
print('creating the dataset...')

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

# Usage example:
train_dataloader, train_dataset_len = create_dataset("filelists/train.txt", segment_length, sampling_rate, hop_length, batch_size)
validation_dataloader, val_dataset_len = create_dataset("filelists/val.txt", segment_length, sampling_rate, hop_length, batch_size)


print("saving initial configs...")
config_path = workdir / 'config.ini'
with open(config_path, 'w') as configfile:
  config.write(configfile)


# Train
model_dir = workdir / "model"
checkpoint_dir = model_dir / 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

log_dir = workdir / 'logs'
os.makedirs(log_dir, exist_ok=True)

if generate_test:

  test_dataset, audio_log_dir = init_test_audio(workdir, sampling_rate, segment_length)
  test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)


# Neural Network
state = torch.load(Path(r'model/ckpt_00500'), map_location=torch.device(device))

model = VAE(segment_length, n_units, n_hidden_units, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.load_state_dict(state['state_dict'])


# Some dummy variables to keep track of loss situation

train_loss_prev = 1000000
best_loss = 1000000
final_loss = 1000000

for epoch in range(epochs):
  
  print('Epoch {}/{}'.format(epoch, epochs - 1))
  print('-' * 10)

  model.train()
  train_loss = 0
  
  for i, data in enumerate(train_dataloader):
    
    # data, = data
    data = data.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
  
  print('====> Epoch: {} - Total loss: {} - Average loss: {:.9f}'.format(
          epoch, train_loss, train_loss / train_dataset_len))
  
  if epoch % checkpoint_interval == 0 and epoch != 0: 
    print('Checkpoint - Epoch {}'.format(epoch))
    state = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
    }
    
    # Validation
    model.eval()
    validation_loss = 0
    with torch.no_grad():
      for data in validation_dataloader:
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        validation_loss += loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length).item()

    print('Validation loss: {:.9f}'.format(validation_loss / val_dataset_len))
    
    # Save checkpoint
    torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epoch)))
  
    # Save best model
    if (train_loss < train_loss_prev) and (epoch > save_best_model_after):
      save_path = workdir.joinpath('model').joinpath('best_model.pt')
      torch.save(model, save_path)
      print('Epoch {:05d}: Saved {}'.format(epoch, save_path))
      config['training']['best_epoch'] = str(epoch)
      best_loss = train_loss

    elif (train_loss > train_loss_prev):
      print("Average loss did not improve.")
  
  final_loss = train_loss

print('Last Checkpoint - Epoch {}'.format(epoch))
state = {
  'epoch': epoch,
  'state_dict': model.state_dict(),
  'optimizer': optimizer.state_dict()
}

if generate_test:
      
  init_test = True
  
  for iterno, test_sample in enumerate(test_dataloader):
    with torch.no_grad():
      test_sample = test_sample.to(device)
      test_pred = model(test_sample)[0]
  
    if init_test:
      test_predictions = test_pred
      init_test = False
    
    else:
      test_predictions = torch.cat((test_predictions, test_pred), 0)
    
  audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format(epochs))
  test_predictions_np = test_predictions.view(-1).cpu().numpy()
  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('Audio examples generated: {}'.format(audio_out))

  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('Last Audio examples generated: {}'.format(audio_out))


# Save the last model as a checkpoint dict
torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epochs)))

if train_loss > train_loss_prev:
  print("Final loss was not better than the last best model.")
  print("Final Loss: {}".format(final_loss))
  print("Best Loss: {}".format(best_loss))
  
  # Save the last model using torch.save 
  save_path = workdir.joinpath('model').joinpath('last_model.pt')
  torch.save(model, save_path)
  print('Training Finished: Saved the last model')

else:
  print("The last model is the best model.")

with open(config_path, 'w') as configfile:
  config.write(configfile)