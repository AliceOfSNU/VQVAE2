import torch
import torch.nn as nn
import numpy as np
import lmdb
import torch.nn.functional as F

    
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, cond_in_channel=0):
        super().__init__()
        # resblock has bottleneck channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(channel, in_channel, 1), #maybe try change this to 3
            nn.BatchNorm2d(in_channel),
        )
        self.activ = nn.ReLU()
        self.conditioned=False
        
        #if conditioned, put condition into a simple conv
        if cond_in_channel > 0:
            self.conditioned=True
            self.cond_mlp = nn.Sequential(
                nn.ReLU(), #originally an embedded vector
                nn.Linear(cond_in_channel, channel),
            )
            self.cond_conv = nn.Sequential(
                nn.Conv2d(channel*2, channel, 1),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x, cond = None):
        out = self.conv1(x)
        if self.conditioned:
            cond = self.cond_mlp(cond)
            cond = cond[..., None, None].expand_as(out)
            out = self.cond_conv(torch.cat([out, cond], 1))
        out = self.activ(out)
        out = self.conv2(out)
        out = out + x
        return self.activ(out)
    
class Encoder(nn.Module):
    def __init__(self, 
                 in_channels, hidden_dim, n_resblocks=2, 
                 cond_in_channels = 0, downsample_ratio=2):
        super().__init__()
        if downsample_ratio == 2:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, hidden_dim//2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim//2, hidden_dim, 3, padding=1),
                nn.ReLU()
            ])
        elif downsample_ratio == 4:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, hidden_dim//2, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(hidden_dim//2, hidden_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU()
            ])
        else:
            raise NotImplementedError("downsample ratio of 2 or 4 supported")
        
        self.is_conditioned=(cond_in_channels > 0)
        self.resblocks = nn.ModuleList([ResBlock(hidden_dim, hidden_dim//2, cond_in_channel=cond_in_channels) for n in range(n_resblocks)])
    
    def forward(self, x, cond=None):
        x = self.conv(x)
        for block in self.resblocks:
            x = block(x, cond)
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, 
                 cond_in_channels = 0, n_resblocks=2, upsample_ratio=2):
        super().__init__()
        self.is_conditioned=False
        if cond_in_channels > 0:
            self.is_conditioned = True
            
        self.conv = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                  nn.ReLU())

        self.resblocks = nn.ModuleList([ResBlock(hidden_dim, hidden_dim//2, cond_in_channel=cond_in_channels) for n in range(n_resblocks)])
        if upsample_ratio == 2:
            self.out_blocks = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1)
            )
        elif upsample_ratio == 4:
            self.out_blocks = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_dim//2, out_channels, 4, stride=2, padding=1),
            )
        else:
            raise NotImplementedError("upsample ratio of 2 or 4 is supported")
        
    def forward(self, x, cond=None):
        out = self.conv(x)
        for block in self.resblocks:
            out = block(out, cond)
        return self.out_blocks(out)

class ConditionEmbedding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.gender_embed = nn.Embedding(2, 5)
        self.mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
    
    # accepts label and provides an embedding vector for it
    def forward(self, label):
        x = torch.LongTensor([1 if gen=='male' else 0 for gen in label["gender"]]).to(self.gender_embed.weight.device)
        x = self.gender_embed(x)
        ages = ((label["age"].float()-40)/80).to(self.gender_embed.weight.device).unsqueeze(-1)
        x = torch.cat([x, ages], 1)
        return self.mlp(x)
    
             
class QuantizedEmbedding(nn.Module):
    # much like nn.Embedding, but instead of training, use a custom update rule
    def __init__(self, num_embeddings, embed_dim, gamma=0.99, eps=1e-5):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        
        self.num_embeddings = num_embeddings
        W = torch.randn(num_embeddings, embed_dim)
        
        self.register_buffer("embedW", W)
        #self.register_buffer("mt", W.clone()) #should be zeros_like(W) or (1-gamma)*W
        self.register_buffer("mt", torch.zeros_like(W)) #should be zeros_like(W) or (1-gamma)*W
        self.register_buffer("Nt", torch.zeros(num_embeddings))
        
    # when using inplace operations(_) do not reassign.
    def update_weights(self, ze, zq_onehot, gamma, eps):
        # appendix A.1. from "Neural Discrete Representation Learning" 
        # https://arxiv.org/pdf/1711.00937.pdf
        # exponential moving average is same as sonnet.
        nt = zq_onehot.sum(dim = 0) #vector of length num_embeddings
        self.Nt = gamma * self.Nt + (1-gamma) * nt
        
        st = zq_onehot.float().transpose(0, 1) @ ze
        self.mt = gamma * self.mt + (1-gamma) * st
        
        # slight modification to counts adapted from 
        # https://github.com/deepmind/sonnet
        n = self.Nt.sum()
        # normalize proportions again because we are adding many hundreds of eps..
        N = (self.Nt + eps) * n/(n + self.num_embeddings*eps)
        
        et = self.mt/N.unsqueeze(-1)#if self.Nt is zero?
        self.embedW.data.copy_(et)
        
    # x must be flattened in advance
    def forward(self, ze):
        B = ze.shape[0]
        ze_flat = ze.flatten(end_dim = -2)
        
        # this code will run out of gpu
        #e = self.embedW.unsqueeze(0)
        #dist = ze_flat.unsqueeze(1) - e #broadcast
        #dist = dist.pow(2).sum(dim=-1)
        
        # this code works.. ???
        dist = (
            ze_flat.pow(2).sum(1, keepdim=True)
            - 2 * ze_flat @ self.embedW.T
            + self.embedW.T.pow(2).sum(0, keepdim=True)
        )
        # find closest vector for each x
        _, zq_idx = dist.min(dim=-1)
        del dist
        
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
    
class Classifier(nn.Module):
    def __init__(self, in_channels, n_class, criterion_fn):
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, max(in_channels//2, 32), 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(max(in_channels//2, 32), 32, 3, stride=2),
                nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )
        self.criterion_fn = criterion_fn
    
    def forward(self, x, label):
        out = self.conv(x).flatten(-2)
        y = self.mlp(out)
        return self.criterion_fn(y, label)
        
# VQVAE model class code was borrowed and modified from rosnality/vq-vae-2-pytorch
class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, embed_dim, n_embed, n_resblocks, 
                 conditioned=False, aux_tasks =False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.conditioned = conditioned
        if self.conditioned:
            cond_dim = embed_dim
            self.cond_embed = ConditionEmbedding(cond_dim)
        else: cond_dim = 0
        
        self.aux_tasks = {}
        if aux_tasks:
            def gender_loss(y, label):
                label = torch.LongTensor([1 if gen=='male' else 0 for gen in label["gender"]]).to(y.device)
                return F.binary_cross_entropy_with_logits(y, label)
            def age_loss(y, label):
                label = ((label["age"].float()-40)/80).to(y.device).unsqueeze(-1)
                return F.mse_loss(y, label)
            self.aux_tasks = nn.ModuleDict({
                "gender":Classifier(hidden_dim, 2, gender_loss),
                "age": Classifier(hidden_dim, 1, age_loss)
            })
        # down by 4
        self.bottom_encoder = Encoder(in_channels, hidden_dim, 
                                      n_resblocks = n_resblocks, cond_in_channels= cond_dim, downsample_ratio=4)
        # down by 2
        self.top_encoder = Encoder(hidden_dim, hidden_dim, 
                                   n_resblocks=n_resblocks, cond_in_channels=cond_dim, downsample_ratio=2)
        self.top_quantize_conv = nn.Conv2d(hidden_dim, embed_dim, 1)
        self.top_quantize = QuantizedEmbedding(n_embed, embed_dim)
        
        # up by 2
        self.top_decoder = Decoder(embed_dim, embed_dim, hidden_dim, 
                                   n_resblocks=n_resblocks, cond_in_channels=cond_dim, upsample_ratio=2)
        self.bottom_quantize_conv = nn.Conv2d(embed_dim + hidden_dim, embed_dim, 1)
        self.bottom_quantize = QuantizedEmbedding(n_embed, embed_dim)
        self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        
        # up by 4 
        # this is also the final decoder
        self.bottom_decoder = Decoder(embed_dim+embed_dim, in_channels, hidden_dim, 
                                      n_resblocks=n_resblocks, cond_in_channels=cond_dim, upsample_ratio=4)
        
    def forward(self, x, label=None):
        # preocess conditions
        cond = None
        if self.conditioned and label is not None:
            cond = self.embed_conditions(label)
        t_quantized, b_quantized, latent_loss, tcode, bcode = self.encode(x, cond)
        # perform top-level aux tasks
        aux_losses = {}
        if len(self.aux_tasks)>0:
            for task, head in self.aux_tasks.items():
                aux_losses[task] = head.forward(t_quantized, label)
        # decode to image level for reconstr loss
        b_decoded = self.decode(t_quantized, b_quantized, cond)
        return b_decoded, latent_loss, aux_losses, tcode, bcode
        
    def embed_conditions(self, label):
        return self.cond_embed(label)
    
    def encode(self, x, cond=None):
        b_encoded = self.bottom_encoder(x, cond)
        t_encoded = self.top_encoder(b_encoded, cond)
        
        t_quantized = self.top_quantize_conv(t_encoded)
        H, W = t_quantized.shape[-2:]
        t_quantized = t_quantized.flatten(-2).transpose(-1, -2)
        t_quantized, t_dists, t_idxs = self.top_quantize(t_quantized)
        t_quantized = t_quantized.transpose(-1, -2).view(-1, self.embed_dim, H, W)
        t_idxs = t_idxs.view(-1, H, W)
        
        t_decoded = self.top_decoder(t_quantized, cond)
        b_encoded = torch.cat([t_decoded, b_encoded], dim=1)
        
        b_quantized = self.bottom_quantize_conv(b_encoded)
        H, W = b_quantized.shape[-2:]
        b_quantized = b_quantized.flatten(-2).transpose(-1, -2) #(B, H*W, C)
        b_quantized, b_dists, b_idxs = self.bottom_quantize(b_quantized)
        b_quantized = b_quantized.transpose(-1, -2).view(-1, self.embed_dim, H, W) #(B, C, H, W)
        b_idxs = b_idxs.view(-1, H, W)
        
        return t_quantized, b_quantized, b_dists + t_dists, t_idxs, b_idxs
    
    def decode(self, top_q, bottom_q, cond):
        upsampled = self.upsample(top_q)
        return self.bottom_decoder(torch.cat([upsampled, bottom_q], dim=1), cond)
    
    def generate(self, t_idxs, b_idxs, labels=None):
        with torch.inference_mode():
            cond = self.embed_conditions(labels)
            t_quantized = self.top_quantize.lookup(t_idxs).permute(0, 3, 1, 2)
            b_quantized = self.bottom_quantize.lookup(b_idxs).permute(0, 3, 1, 2)
            decoded = self.decode(t_quantized, b_quantized, cond)#channel first
        return torch.clamp(decoded, 0.0, 1.0)
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
        quantized, sqdist, quantized_idxs = self.encode(x)
        
        decoded = self.decode(quantized)
        
        return decoded, sqdist
    
    # performs encoding action exclusively
    # returns
    #   quantized: embed_dim vector for each pixel
    #   sqdist: single value tensor for the commitment loss
    #   idxs: LongTensor for indexing into the codebook
    def encode(self, x):
        x = self.encoder(x)
        x = self.quantize_conv(x)
        H, W = x.shape[-2:]
        x = x.flatten(-2).transpose(-1, -2)
        quantized, sqdist, idxs = self.quantize(x)
        quantized = quantized.transpose(-1, -2).view(-1, self.embed_dim, H, W)
        idxs = idxs.view(-1, H, W)
        return quantized, sqdist, idxs
    
    def generate(self, code):
        with torch.inference_mode():
            code = self.quantize.lookup(code)
            decoded = self.decode(code.permute(0, 3, 1, 2))#channel first
        return torch.clamp(decoded, 0.0, 1.0)
    