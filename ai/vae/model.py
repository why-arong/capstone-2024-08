import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE(nn.Module):
  def __init__(self, segment_length, n_units, n_hidden_units, latent_dim):
      super(VAE, self).__init__()

      self.segment_length = segment_length
      self.n_units = n_units
      self.latent_dim = latent_dim

      self.fc1 = nn.Linear(segment_length, n_units)
      self.fc2 = nn.Linear(n_units, n_hidden_units)
      self.fc3 = nn.Linear(n_hidden_units, latent_dim)
      self.fc4 = nn.Linear(n_hidden_units, latent_dim)

      self.fc5 = nn.Linear(latent_dim, n_hidden_units)
      self.fc6 = nn.Linear(n_hidden_units, n_units)
      self.fc7 = nn.Linear(n_units, segment_length)

  def encode(self, x):
      h1 = F.relu(self.fc1(x))
      h2 = F.relu(self.fc2(h1))
      return self.fc3(h2), self.fc4(h2)

  def reparameterize(self, mu, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + eps * std

  def decode(self, z):
      h3 = F.relu(self.fc5(z))
      h4 = F.relu(self.fc6(h3))
      return F.tanh(self.fc7(h4))

  def forward(self, x):
      mu, logvar = self.encode(x.view(-1, self.segment_length))
      z = self.reparameterize(mu, logvar)
      return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, kl_beta, segment_length):
  recon_loss = F.mse_loss(recon_x, x.view(-1, segment_length))

  # see Appendix B from VAE paper:
  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  # https://arxiv.org/abs/1312.6114
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

  return recon_loss + ( kl_beta * KLD)