import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SpatialPositionalEmbedding(nn.Module):
    
    def __init__(self, d_model, x_max_len=10, y_max_len=10):
        super().__init__()

        # Compute the positional encodings once in log space.
        x_pe = torch.zeros(x_max_len, d_model//2).float()
        x_pe.require_grad = False

        y_pe = torch.zeros(y_max_len, d_model//2).float()
        y_pe.require_grad = False

        position = torch.arange(0, x_max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model//2, 2).float() * -(math.log(10000.0) / d_model)).exp()
        x_pe[:, 0::2] = torch.sin(position * div_term)
        x_pe[:, 1::2] = torch.cos(position * div_term)

        position = torch.arange(0, y_max_len).float().unsqueeze(1)
        y_pe[:, 0::2] = torch.sin(position * div_term)
        y_pe[:, 1::2] = torch.cos(position * div_term)

        x_pe = x_pe.view(x_max_len, 1, -1).repeat(1, y_max_len, 1).view(x_max_len*y_max_len, -1)
        y_pe = y_pe.view(1, y_max_len, -1).repeat(x_max_len, 1, 1).view(x_max_len*y_max_len, -1)

        pe = torch.cat([x_pe, y_pe], dim=1)
        pe = pe.view(1, x_max_len*y_max_len, -1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]