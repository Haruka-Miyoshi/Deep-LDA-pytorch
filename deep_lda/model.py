import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder

class Model(nn.Module):
    def __init__(self, x_dim:int, h_dim:int, z_dim:int):
        super(Model, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.x_to_z = Encoder(self.x_dim, self.h_dim, self.z_dim)
        self.z_to_x = Decoder(self.x_dim, self.h_dim, self.z_dim)

        self.softmax = nn.Softmax(dim=1)
    
    def reparameterize(self, mu, logvar, mode):
        if mode:
            s = torch.exp(0.5 * logvar)
            e = torch.rand_like(s)
            return e.mul(s).add_(mu)
        else:
            return mu
    
    def encode(self, x, mode):
        mu, logvar = self.x_to_z(x)
        z = self.reparameterize(mu, logvar, mode)
        theta = self.softmax(z)
        return mu, logvar, z, theta
    
    def decode(self, theta):
        xh = self.z_to_x(theta)
        return xh

    def forward(self, x, mode):
        mu, logvar = self.x_to_z(x)
        z = self.reparameterize(mu, logvar, mode)
        theta = self.softmax(z)
        xh = self.z_to_x(theta)
        return mu, logvar, z, theta, xh