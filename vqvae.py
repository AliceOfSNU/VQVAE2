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
        
    def forward(self, x):
        out = self.conv(x)
        out = out + x
        return self.activ(out)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_resblocks=2, downsample_ratio=2):
        super().__init__()
        if downsample_ratio == 2:
            blocks = [
                nn.Conv2d(in_channels, hidden_dim//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim//2, hidden_dim, 3, padding=1),
                nn.ReLU()
            ]
        elif downsample_ratio == 4:
            blocks = [
                nn.Conv2d(in_channels, hidden_dim//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim//2, hidden_dim, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True)
            ]
        else:
            raise NotImplementedError("downsample ratio of 2 or 4 supported")
                    
        blocks.extend([ResBlock(hidden_dim, hidden_dim//2) for n in range(n_resblocks)])
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)
    

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, n_resblocks=2, upsample_ratio=2):
        super().__init__()
        blocks = [nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                  nn.ReLU(inplace=True)]
        blocks += [ResBlock(hidden_dim, hidden_dim//2) for n in range(n_resblocks)]
        if upsample_ratio == 2:
            blocks.extend([
                nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1)
            ])
        elif upsample_ratio == 4:
            blocks.extend([
                nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim//2, out_channels, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ])
        else:
            raise NotImplementedError("upsample ratio of 2 or 4 is supported")
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.blocks(x)

             
class QuantizedEmbedding(nn.Module):
    # much like nn.Embedding, but instead of training, use a custom update rule
    def __init__(self, num_embeddings, embed_dim, gamma=0.99, eps=1e-5):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        
        self.num_embeddings = num_embeddings
        W = torch.randn(num_embeddings, embed_dim)
        
        self.register_buffer("embedW", W)
        self.register_buffer("mt", W.clone())
        self.register_buffer("Nt", torch.zeros(num_embeddings))
        
    # when using inplace operations(_) do not reassign.
    def update_weights(self, ze, zq_onehot, gamma, eps):
        # appendix A.1. from "Neural Discrete Representation Learning" 
        # https://arxiv.org/pdf/1711.00937.pdf
        nt = zq_onehot.sum(dim = 0) #vector of length num_embeddings
        self.Nt = gamma * self.Nt + (1-gamma) * nt
        
        st = zq_onehot.float().transpose(0, 1) @ ze
        self.mt = gamma * self.mt + (1-gamma) * st
        
        # slight modification to counts adapted from 
        # https://github.com/deepmind/sonnet
        n = self.Nt.sum()
        N = (self.Nt + eps) * n/(n + self.num_embeddings*eps)
        
        et = self.mt/N.unsqueeze(-1)#if self.Nt is zero?
        self.embedW.data.copy_(et)
        
    # x must be flattened in advance
    def forward(self, ze):
        B = ze.shape[0]
        ze_flat = ze.flatten(end_dim = -2)
        
        e = self.embedW.unsqueeze(0)
        dist = ze_flat.unsqueeze(1) - e #broadcast
        dist = dist.pow(2).sum(dim=-1)
        
        # find closest vector for each x
        _, zq_idx = dist.min(dim=-1)
        zq_onehot = F.one_hot(zq_idx, self.num_embeddings) #B, ze, V
        zq = self.lookup(zq_idx)
        
        # update
        if self.training:
            self.update_weights(ze_flat.detach(), zq_onehot, self.gamma, self.eps)
            
        zq = zq.view(B, -1, *zq.shape[1:])
        zq_idx = zq_idx.view(B, -1)
        
        dist = (zq.detach() - ze).pow(2).mean() # the ||z_e(x)-sq[e]||^2 term
        zq = ze + (zq-ze).detach() # straight-through trick to pass gradient
        
        return zq, dist, zq_idx
    
    def lookup(self, zq_idx):
        return F.embedding(zq_idx, self.embedW)
        
# VQVAE model class code was borrowed and modified from rosnality/vq-vae-2-pytorch
class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, embed_dim, n_embed, n_resblocks):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        # down by 4
        self.bottom_encoder = Encoder(in_channels, hidden_dim, n_resblocks, downsample_ratio=4)
        # down by 2
        self.top_encoder = Encoder(hidden_dim, hidden_dim, n_resblocks, downsample_ratio=2)
        self.top_quantize_conv = nn.Conv2d(hidden_dim, embed_dim, 1)
        self.top_quantize = QuantizedEmbedding(n_embed, embed_dim)
        
        # up by 2
        self.top_decoder = Decoder(embed_dim, embed_dim, hidden_dim, n_resblocks=n_resblocks, upsample_ratio=2)
        
        self.bottom_quantize_conv = nn.Conv2d(embed_dim + hidden_dim, embed_dim, 1)
        self.bottom_quantize = QuantizedEmbedding(n_embed, embed_dim)
        self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        
        # up by 4 
        # this is also the final decoder
        self.bottom_decoder = Decoder(embed_dim+embed_dim, in_channels, hidden_dim, n_resblocks=n_resblocks, upsample_ratio=4)
        
    def forward(self, x):
        b_encoded = self.bottom_encoder(x)
        t_encoded = self.top_encoder(b_encoded)
        
        t_quantized = self.top_quantize_conv(t_encoded)
        H, W = t_quantized.shape[-2:]
        t_quantized = t_quantized.flatten(-2).transpose(-1, -2)
        t_quantized, t_dists, t_idxs = self.top_quantize(t_quantized)
        t_quantized = t_quantized.transpose(-1, -2).view(-1, self.embed_dim, H, W)
        t_idxs = t_idxs.view(-1, H, W)
        
        t_decoded = self.top_decoder(t_quantized)
        b_encoded = torch.cat([t_decoded, b_encoded], dim=1)
        
        b_quantized = self.bottom_quantize_conv(b_encoded)
        H, W = b_quantized.shape[-2:]
        b_quantized = b_quantized.flatten(-2).transpose(-1, -2)
        b_quantized, b_dists, b_idxs = self.bottom_quantize(b_quantized)
        b_quantized = b_quantized.transpose(-1, -2).view(-1, -1, H, W)
        b_idxs = b_idxs.view(-1, H, W)
        
        latent_loss = b_dists + t_dists
        b_decoded = self.decode(t_quantized, b_quantized)
        return b_decoded, latent_loss
        
    
    def decode(self, top_q, bottom_q):
        upsampled = self.upsample(top_q)
        dec = self.bottom_decoder(torch.cat([upsampled, bottom_q], dim=1))
    
# a single-layer version of VQVAE.
# much similar to the first version of VQVAE
class SingleVQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, embed_dim, n_embed, n_resblocks):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        
        # down by 4
        self.encoder = Encoder(in_channels, hidden_dim, n_resblocks, downsample_ratio=4)
        
        # quantize
        self.quantize_conv = nn.Conv2d(hidden_dim, embed_dim, 1)
        self.quantize = QuantizedEmbedding(n_embed, embed_dim)
        
        # up by 4
        self.decoder = Decoder(embed_dim, in_channels, hidden_dim, n_resblocks=n_resblocks, upsample_ratio=4)
        
    def decode(self, q):
        return self.decoder(q)
    
    def forward(self, x):
        encoded = self.encoder(x)
        
        quantized = self.quantize_conv(encoded)
        quantized, sqdist, quantized_idxs = self.encode(quantized)
        
        decoded = self.decode(quantized)
        
        return decoded, sqdist
    
    # performs encoding action exclusively
    # returns
    #   quantized: embed_dim vector for each pixel
    #   sqdist: single value tensor for the commitment loss
    #   idxs: LongTensor for indexing into the codebook
    def encode(self, x):
        H, W = x.shape[-2:]
        x = x.flatten(-2).transpose(-1, -2)
        quantized, sqdist, idxs = self.quantize(x)
        quantized = quantized.transpose(-1, -2).view(-1, self.embed_dim, H, W)
        idxs = idxs.view(-1, H, W)
        return quantized, sqdist, idxs