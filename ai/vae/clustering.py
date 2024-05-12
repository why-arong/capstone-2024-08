import torch
import torchaudio
from pathlib import Path

Resample = torchaudio.transforms.Resample(44100, 48000, resampling_method='kaiser_window')

if torch.cuda.is_available():
    device = 'cuda:0'
    my_cuda = 1
else:
    device = 'cpu'
    my_cuda = 0

Resample = Resample.to(device)

from model import VAE
from utils import create_dataset

from sklearn.cluster import KMeans


# Generate latent space data
def generate_latent_data(model, dataloader):
    model.eval()
    latent_data = []

    with torch.no_grad():
        for data in dataloader:
            recon_batch, mu, logvar = model(data)
            latent_data.append(mu)

    latent_data = torch.cat(latent_data, dim=0)
    return latent_data.numpy()


segment_length = 1024
n_units = 256
n_hidden_units = 64
latent_dim = 8

batch_size = 256
sampling_rate = 44100
hop_length = 128

state = torch.load(Path(r'ai/vae/model/ckpt'), map_location=torch.device(device))

if my_cuda:
    model = VAE(segment_length, n_units, n_hidden_units, latent_dim).to(device)
else:
    model = VAE(segment_length, n_units, n_hidden_units, latent_dim)

model.load_state_dict(state['state_dict'])
model.eval()


test_dataloader, test_dataset_len = create_dataset("filelists/test.txt", segment_length, sampling_rate, hop_length, batch_size)


# Generate latent space data
latent_data = generate_latent_data(model, test_dataloader)

# Apply K-means clustering
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(latent_data)

# Get cluster centroids
cluster_centers = kmeans.cluster_centers_

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Print cluster centroids and labels
print("Cluster Centroids:")
print(cluster_centers)
print("Cluster Labels:")
print(cluster_labels)