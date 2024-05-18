import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000000

        self.encoder_conv_layers = self._build_encoder_conv_layers()
        self.fc_mu = nn.Linear(self.encoder_conv_output_size, self.latent_space_dim)
        self.fc_logvar = nn.Linear(self.encoder_conv_output_size, self.latent_space_dim)
        self.decoder_fc = nn.Linear(self.latent_space_dim, self.encoder_conv_output_size)
        self.decoder_conv_layers = self._build_decoder_conv_layers()

    def _build_encoder_conv_layers(self):
        layers = []
        in_channels = self.input_shape[0]
        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.encoder_conv_output_size = self._get_conv_output_size(layers)
        return nn.Sequential(*layers)

    def _build_decoder_conv_layers(self):
        layers = []
        in_channels = self.conv_filters[-1]
        for out_channels, kernel_size, stride in zip(reversed(self.conv_filters[:-1]), reversed(self.conv_kernels[:-1]), reversed(self.conv_strides[:-1])):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        layers.append(nn.ConvTranspose2d(in_channels, self.input_shape[0], self.conv_kernels[0], self.conv_strides[0], padding=1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, layers):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_output = nn.Sequential(*layers)(dummy_input)
            return dummy_output.view(1, -1).size(1)

    def encode(self, x):
        conv_out = self.encoder_conv_layers(x)
        mu = self.fc_mu(conv_out)
        logvar = self.fc_logvar(conv_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, self.conv_filters[-1], self.input_shape[1] // (2 ** (len(self.conv_filters) - 1)), self.input_shape[2] // (2 ** (len(self.conv_filters) - 1)))
        x = self.decoder_conv_layers(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def loss_function(recon_x, x, mu, logvar, reconstruction_loss_weight):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss_weight * BCE + KLD
