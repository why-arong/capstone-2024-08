import torch
from pathlib import Path
from sklearn.cluster import KMeans
from model import VAE
from utils import create_dataset
import numpy as np


# Function to generate latent space data using the VAE model
def generate_latent_data(model, dataloader):
    model.eval()
    latent_data = []

    with torch.no_grad():
        for data in dataloader:
            recon_batch, mu, logvar = model.forward(data)
            latent_data.append(mu)

    latent_data = torch.cat(latent_data, dim=0)
    return latent_data.numpy()

# Function to load VAE model from checkpoint
def load_model_from_checkpoint(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=torch.device(device))
    model = VAE(state['segment_length'], state['n_units'], state['n_hidden_units'], state['latent_dim']).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model

# Function to perform K-means clustering
def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    return cluster_centers, cluster_labels

if __name__ == "__main__":
    segment_length = 1024
    n_units = 256
    n_hidden_units = 64
    latent_dim = 8
    batch_size = 256
    sampling_rate = 44100
    hop_length = 128
    num_clusters = 10

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_checkpoint_path = Path('ai/vae/model/ckpt')
    model = load_model_from_checkpoint(model_checkpoint_path, device)

    test_dataloader, test_dataset_len = create_dataset("filelists/test.txt", segment_length, sampling_rate, hop_length, batch_size)

    latent_data = generate_latent_data(model, test_dataloader)

    cluster_centers, cluster_labels = kmeans_clustering(latent_data, num_clusters)

    # Print cluster labels and number of data points in each cluster
    for cluster_id in range(num_clusters):
        num_points_in_cluster = sum(cluster_labels == cluster_id)
        print(f"Cluster {cluster_id}: {num_points_in_cluster} data points")

    print("Cluster Centroids:")
    print(cluster_centers)
    print("Cluster Labels:")
    print(cluster_labels)
    
    output_file = model_checkpoint_path / "cluster_results.txt"
    np.savetxt(output_file, cluster_centers)
    with open(output_file, "a") as f:
        f.write("\n")
        np.savetxt(f, cluster_labels)