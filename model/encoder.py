import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Encoder(nn.Module):
    def __init__(self, config, img_channels, latent_dim, img_size):
        super(Encoder, self).__init__()
        latent_config = config["latent"]
        architecture = latent_config["encoder_architecture"]
        kernel_size = latent_config["kernel_size"]
        padding = (kernel_size - 1) // 2
        layers = []

        in_channels = img_channels
        for out_channels in architecture:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        architecture = latent_config["encoder_architecture"]
        encoded_img_size = img_size // (2 ** len(architecture))  # Adjusted based on the number of layers
        self.fc = nn.Sequential(
            nn.Linear(architecture[-1] * encoded_img_size * encoded_img_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, config, img_channels, feature_dim, img_size):
        super(Decoder, self).__init__()
        latent_config = config["latent"]
        architecture = latent_config["encoder_architecture"][::-1]  # Reverse the architecture for decoding
        kernel_size = latent_config["kernel_size"]
        padding = (kernel_size - 1) // 2
        layers = []

        self.encoded_img_size = img_size // (2 ** len(architecture))
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, architecture[0] * self.encoded_img_size * self.encoded_img_size),
            nn.BatchNorm1d(architecture[0] * self.encoded_img_size * self.encoded_img_size),
            nn.ReLU(),
            nn.Unflatten(1, (architecture[0], self.encoded_img_size, self.encoded_img_size))
        )

        in_channels = architecture[0]
        for out_channels in architecture[1:]:
            layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            in_channels = out_channels

        # Last layer to produce the output image
        layers += [
            nn.ConvTranspose2d(in_channels, img_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1),
            nn.Sigmoid()
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = self.decoder(x)
        return x

class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()
        self.config = config
        img_channels=3
        latent_config = config["latent"]
        latent_dim = latent_config["latent_dim"]
        img_size = latent_config["img_size"]
        assert latent_config["kernel_size"] in [3, 5, 7], "Only support kernel size 3, 5, 7"
        self.encoder = Encoder(config, img_channels, latent_dim, img_size)
        self.aux_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, 14),
        )
        self.decoder = Decoder(config, img_channels, latent_dim, img_size)

    def forward(self, x):
        # x: [B, C, H, W]
        latent = self.encoder(x)
        self.aux_vector = self.aux_mlp(latent)
        decoded = self.decoder(latent)
        return decoded, latent

    def save_model(self, base_path):
        torch.save(self.encoder, base_path + "/enc.pth")
        torch.save(self.decoder, base_path + "/dec.pth")
        torch.save(self, base_path + "/full.pth")

class VAE_Encoder(nn.Module):
    def __init__(self, config, img_channels, latent_dim, img_size):
        super(VAE_Encoder, self).__init__()
        latent_config = config["latent"]
        architecture = latent_config["encoder_architecture"]
        kernel_size = latent_config["kernel_size"]
        padding = (kernel_size - 1) // 2
        layers = []

        in_channels = img_channels
        for out_channels in architecture:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        architecture = latent_config["encoder_architecture"]
        encoded_img_size = img_size // (2 ** len(architecture))  # Adjusted based on the number of layers
        self.var_fc = nn.Linear(architecture[-1] * encoded_img_size * encoded_img_size, latent_dim)
        self.mu_fc = nn.Linear(architecture[-1] * encoded_img_size * encoded_img_size, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        x = self.encoder(x)
        latent = torch.flatten(x, start_dim=1)
        mu, logvar = self.mu_fc(latent), self.var_fc(latent)
        z = self.reparameterize(mu, logvar)
        self.mu = mu 
        self.logvar = logvar
        return z 
    
class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config
        img_channels=3
        latent_config = config["latent"]
        latent_dim = latent_config["latent_dim"]
        img_size = latent_config["img_size"]
        assert latent_config["kernel_size"] in [3, 5, 7], "Only support kernel size 3, 5, 7"
        self.encoder = VAE_Encoder(config, img_channels, latent_dim, img_size)
        self.decoder = Decoder(config, img_channels, latent_dim, img_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded, latent
    
    def save_model(self, base_path):
        torch.save(self.encoder, base_path + "/enc.pth")
        torch.save(self.decoder, base_path + "/dec.pth")
        torch.save(self, base_path + "/full.pth")