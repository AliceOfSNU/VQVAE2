import torch
import torch.nn as nn
import numpy as np
import lmdb
import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        # resblock has bottleneck channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )
        self.activ = nn.ReLU(inplace=True)
    def forward(self, input):
        out = self.conv(input)
        out += input
        
        return self.activ(out)
    
# 64x64 -> 32x32
class TopEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_resblocks=2):
        super().__init__()
        self.blocks = [
            nn.Conv2d(in_channels, hidden_dim//2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3, padding=1),
            nn.ReLU()
        ]
        self.blocks += [ResBlock(hidden_dim, hidden_dim//2) for n in range(n_resblocks)]
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        return self.blocks(x)
    
# 256*256 -> 64x64 
class BottomEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_resblocks=2):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channels, hidden_dim//2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        ]
        blocks += [ResBlock(hidden_dim, hidden_dim//2) for n in range(n_resblocks)]
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)

# 32x32->64x64(unsample 2)
class TopDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, n_resblocks=2):
        super().__init__()
        blocks = [nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                  nn.ReLU(inplace=True)]
        blocks += [ResBlock(hidden_dim, hidden_dim//2) for n in range(n_resblocks)]
        blocks.extend([
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1)
        ])
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.blocks(x)

# 64x64->256x256(unsample 4)
class BottomDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, n_resblocks=2):
        super().__init__()
        blocks = [nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                  nn.ReLU(inplace=True)]
        blocks += [ResBlock(hidden_dim, hidden_dim//2) for n in range(n_resblocks)]
        blocks.extend([
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim//2, out_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ])
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.blocks(x)
             

        
class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_resblocks):
        super().__init__()
        # down by 4
        self.bottom_encoder = BottomEncoder(in_channels, hidden_dim, n_resblocks)
        # down by 2
        self.top_encoder = TopEncoder(hidden_dim, hidden_dim, n_resblocks)
        
        # up by 2
        self.top_decoder = TopDecoder(hidden_dim, hidden_dim, hidden_dim, n_resblocks=n_resblocks)
        # up by 4
        self.bottom_decoder = BottomDecoder(hidden_dim, in_channels, hidden_dim, n_resblocks=n_resblocks)
        
    def forward(self, x):
        b_encoded = self.bottom_encoder(x)
        t_encoded = self.top_encoder(b_encoded)
        
        t_decoded = self.top_decoder(t_encoded)
        b_encoded = torch.cat([t_decoded, b_encoded], dim=1)
        
        return t_encoded, b_encoded
        