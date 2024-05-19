import torch
from torch.utils.data import DataLoader

import librosa
import librosa.display

import configparser
import  sys, argparse
sys.path.append('../vae')

from dataset import AudioDataset, ToTensor
from model import VAE
from vae.utils import generate_latent_data

def get_cond(wav, path):
    config_path = path + 'config.ini'

    config = configparser.ConfigParser(allow_no_value=True)
    try: 
        config.read(config_path)
        print('Config File Found at {}'.format(config_path))
    except FileNotFoundError:
        print('Config File Not Found at {}'.format(config_path))
        sys.exit()

    sampling_rate = config['audio'].getint('sampling_rate')
    hop_length = config['audio'].getint('hop_length')
    segment_length = config['audio'].getint('segment_length')

    latent_dim = config['VAE'].getint('latent_dim')
    n_units = config['VAE'].getint('n_units')
    n_hidden_units = config['VAE'].getint('n_hidden_units')
    batch_size = config['training'].getint('batch_size')


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path = path + 'best_model.pt'

    model = VAE(segment_length, n_units, n_hidden_units, latent_dim).to(device)

    model = torch.load(model_path, map_location=torch.device(device))
    model.to(device) 
    model.eval()


    test_dataset = AudioDataset(wav, segment_length = segment_length, sampling_rate = sampling_rate, hop_size=hop_length, transform=ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    latent_data = generate_latent_data(model, test_dataloader, device)

    return latent_data