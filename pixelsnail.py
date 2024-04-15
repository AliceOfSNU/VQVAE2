# autoregressive model

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
import utils
# if newest torch, use torch.nn.utils.parametrizations.weight_norm
# https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
from torch.nn.utils import weight_norm

import math

# modeling heavily borrows from https://github.com/rosinality/vq-vae-2-pytorch/
# MIT License 

# very basic building block: ConvNorm
class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                              stride=1, padding=0, bias=True):
        super().__init__()
        # create a convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.conv = weight_norm(self.conv)
    # layernorm
    def forward(self, x):
        return self.conv(x)

# specialization of 1x1 size. for convenience.
class ConvNorm2d11(ConvNorm):
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super().__init__(in_channels, out_channels, 1, stride=stride, padding=0, bias=bias)
        
    def forward(self, x):
        return super().forward(x)
        
# casual building block: CasualConv
class CausalConv2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple,
                              stride=1, mode="downright", bias=True, weight_norm = True):
        super().__init__()
        assert mode in ["right", "downright", "down", "causal"]
        # 4-tuple, uses (left, right, top, bottom)
        ker_v, ker_h = kernel_size
        self.kernel_size = kernel_size
        self.mode = mode
        if mode == "right":
            pad = [ker_h-1, 0, ker_v//1, ker_v//1]
        elif mode == "downright":
            pad = [ker_h-1, 0, ker_v-1, 0]
        elif mode in ["down", "causal"] :
            pad = [ker_h //2 , ker_h //2, ker_v-1, 0]            
            
        self.padding = nn.ZeroPad2d(pad)
        if weight_norm:
            self.conv = ConvNorm(in_channels, out_channels, kernel_size,
                             stride=stride, padding=0, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=0, bias=bias)
            
    def forward(self, x):
        x = self.padding(x)
        if self.mode == "causal":
            # mask the later half of last row of kernel (B mask)
            # modifying data directly is not recommended.. though.
            self.conv.conv.weight_v.data[..., -1, self.kernel_size[1]//2:].zero_()
        return self.conv(x)
    
def shift_right(x, size=1):
    return F.pad(x, [size, 0, 0, 0])[:, :, :, :-1]# add zero to the left and shift right
    
def shift_down(x, size=1):
    return F.pad(x, [0, 0, size, 0])[:, :, :-1, :]# add zero to the top and shift down
    
# gated resblock
class GatedResBlock(nn.Module):
    def __init__(self,
        n_channels, kernel_size, hidden_size, 
        mode=None, cond_channels = 0, dropout=0.1, bias=True
    ):
        super().__init__()
        self.pre_activ = nn.ELU()
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
            
        # instantiate the convolutional layer
        if mode in ["downright", "causal",  "down"]:
            self.conv1 = CausalConv2d(n_channels, hidden_size, kernel_size, mode=mode, bias=bias, weight_norm=True)
            self.conv2 = CausalConv2d(hidden_size, 2*n_channels, kernel_size, mode=mode, bias=bias, weight_norm=True)
        else:
            self.conv1 = ConvNorm(n_channels, hidden_size, kernel_size, 
                                  padding=(kernel_size[0]//2, kernel_size[1]//2), bias=bias)
            self.conv2 = ConvNorm(hidden_size, 2*n_channels, kernel_size, 
                                  padding=(kernel_size[0]//2, kernel_size[1]//2), bias=bias)
        self.activ = nn.ELU()
        
        if cond_channels>0:
            self.cond_conv = ConvNorm2d11(cond_channels, 2*n_channels, bias=False)
            self.is_conditioned = True
        else: 
            self.is_conditioned = False
        # output channel must be double input channels
        # so we can apply glu
        self.dropout = nn.Dropout(dropout)
        
        # glu is implemented as a part of torch.nn module
        # this should be a single-liner
        self.glu = nn.GLU(1)
        
        ## initialize layers
        #for name, param in self.named_parameters():
        #    if "conv" in name:
        #        if "bias" in name:
        #            param.data.fill_(0.0)

        #        if "weight" in name:
        #            param.data.fill_(0.3)
        #            #nn.init.uniform_(param.data, -0.1, 0.1)
                    
    def forward(self, x, cond=None):
        out = self.pre_activ(x)
        out = self.conv1(out)
        out = self.dropout(self.activ(out))
        out = self.conv2(out)
        if self.is_conditioned:
            cond = self.cond_conv(cond)
            out = out + cond #broadcasts if condition is same for all pixels
        out = self.glu(out)
        out = x + out
        return out
       
# specialization with 1x1.
class GatedResBlock11(GatedResBlock):
    def __init__(self,
        n_channels, hidden_size, dropout=0.1, bias=True
    ):
        super().__init__(n_channels, (1, 1), hidden_size,  dropout=dropout, bias=bias)
        
    def forward(self, x):
        return super().forward(x)

@lru_cache(maxsize=32)
def get_causal_mask_as(q_size, kv_size):
    # strictly lower triangualar
    return torch.tril(torch.ones(q_size, kv_size), diagonal=-1)

# a dot product self attention
class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, embed_dim, num_heads, dropout=0.1):
       
        super().__init__()
        self.embed_dim = embed_dim
        
        # create q, k, v matrices, but use weight norm!
        self.query_proj = weight_norm(nn.Linear(query_dim, embed_dim))
        self.key_proj = weight_norm(nn.Linear(key_dim, embed_dim))
        self.value_proj = weight_norm(nn.Linear(value_dim, embed_dim))

        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        assert self.head_dim * num_heads == embed_dim, f"embed_dim({embed_dim}) must be a multiple of num_heads({num_heads})"
        # Initialize the following layers and parameters to perform attention
        self.head_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        N, T, _ = query.shape
        N, S, _ = value.shape
        H = self.num_heads
        D = self.embed_dim
        
        # expected shape is (N, H, T/S, D/H)
        query = self.query_proj(query).view(N, T, H, D//H).transpose(-2, -3)
        key = self.key_proj(key).view(N, S, H, D//H).transpose(-2, -3)
        value = self.value_proj(value).view(N, S, H, D//H).transpose(-2, -3)
        #(N, H, T, D//H) @ (N, H, S, D//H)'
        dot_product = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(D//H)

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            mask = dot_product.new_zeros(attn_mask.shape)
            mask.masked_fill_((attn_mask==0), float("-inf")) #inplace fill
            dot_product = dot_product + mask
            # first row might be full of "-inf". we remove it
            dot_product = dot_product[..., 1:, :]
            attn = dot_product.softmax(-1)
            # and add a row of zeros
            row = attn.new_zeros(*attn.shape[:-2], 1, attn.shape[-1])
            attn = torch.cat([row, attn], -2)# concat first row
        else:
            attn = dot_product.softmax(-1)
        # apply softmax, dropout, and use value
        attn = self.dropout(attn) #(N, H, T, S)
        y = torch.matmul(attn, value).transpose(-3, -2).flatten(-2) #(N, T, H, Ev=D//H)
        
        return y
   
class PositionalEncoding2D(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        H, W = target_size
        coord_y = (torch.arange(H).float() - H / 2) / H
        coord_y = coord_y.view(1, H, 1).expand(1, H, W)
        coord_x = (torch.arange(W).float() - W / 2) / W
        coord_x = coord_x.view(1, 1, W).expand(1, H, W)
        encoding = torch.cat([coord_y, coord_x], dim=0).unsqueeze(0) # add batch dim
        self.register_buffer('encoding', encoding)
        
    def forward(self, x):
        return torch.cat([x, self.encoding.expand(x.shape[0], -1, -1, -1)], dim=1)
        
# joins two branches together with two 1x1 convolutions
class JoinBlock(nn.Module):
    def __init__(self, n_channels, hidden_dim):
        super().__init__()
        self.activ = nn.ELU()
        self.conv_in1 = ConvNorm2d11(n_channels, hidden_dim)
        self.conv_in2 = ConvNorm2d11(n_channels, hidden_dim)
        self.conv_out = ConvNorm2d11(hidden_dim, n_channels)
    
    def forward(self, x1, x2):
        x1 = self.conv_in1(self.activ(x1))
        x2 = self.conv_in2(self.activ(x2))
        out = x1+x2
        out = self.conv_out(self.activ(out))
        return out
    
# block
class PixelSnailBlock(nn.Module):
    def __init__(self, n_channels, n_resblocks, kernel_size, target_size, cond_channels = 0, 
                 attention = True, attn_n_head=8, dropout=0.1):
        super().__init__()
        
        # first res layers
        self.resblocks = [GatedResBlock(n_channels, kernel_size, hidden_size=n_channels,  
                                        mode="causal", cond_channels=cond_channels, dropout=dropout, bias=True) for _ in range(n_resblocks)]
        self.resblocks = nn.ModuleList(self.resblocks)
        self.positional_encoding = PositionalEncoding2D(target_size)
        
        # self attention 
        self.has_attn = attention
        if attention:
            # key&value: cat resblock output to input.
            self.kv_resblock11 = GatedResBlock11(
                # output, input, positional encoding
                n_channels * 2 + 2,
                hidden_size = n_channels, dropout=dropout
            )
            # query: use resblock outputs
            self.query_resblock11 = GatedResBlock11(
                # output, positional_encoding
                n_channels + 2,
                hidden_size= n_channels, dropout=dropout
            )

            attn_hidden_dim = n_channels
            self.attention = MultiHeadAttentionLayer(
                query_dim = n_channels + 2,
                key_dim = n_channels * 2 + 2,
                value_dim = n_channels * 2 + 2,
                embed_dim= attn_hidden_dim,
                num_heads= attn_n_head, dropout=dropout
            )
            
            # combine conv + attn results
            self.join_attn = JoinBlock(n_channels, n_channels)
        else:
            self.out_block = ConvNorm2d11(n_channels + 2, n_channels)
            

    def forward(self, x, cond=None):
        N, _, H, W = x.shape
        out = x
        for block in self.resblocks:
            out = block(out, cond)
            
        if self.has_attn:
            # concatenate at channel dim, and shuffle!
            kv = self.positional_encoding(torch.cat([out, x], dim=1))
            kv = self.kv_resblock11(kv)
            q = self.positional_encoding(out)
            q = self.query_resblock11(q)
            
            # channel_last layout, flattened
            kv = kv.flatten(-2).transpose(-1, -2) 
            q = q.flatten(-2).transpose(-1, -2)
            causal_mask = get_causal_mask_as(q.shape[-2], kv.shape[-2]).to(q.device)
            attn = self.attention(q, kv, kv, causal_mask)
            
            # back to channel_first, 2 dim layout
            attn = attn.reshape(N, H, W, -1)
            attn = attn.permute(0, 3, 1, 2) #channel_first layout
            out = self.join_attn(out, attn)
        else:
            out = self.positional_encoding(out)
            out = self.out_block(out)
            
        return out
    
class ConditionEmbedding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.gender_embed = nn.Embedding(2, 5)
        self.mlp = nn.Sequential(
            nn.Linear(6, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, out_dim)
        )
    
    # accepts label and provides an embedding vector for it
    def forward(self, label):
        x = torch.LongTensor([1 if gen=='male' else 0 for gen in label["gender"]]).to(self.gender_embed.weight.device)
        x = self.gender_embed(x)
        ages = ((label["age"].float()-40)/80).to(self.gender_embed.weight.device).unsqueeze(-1)
        x = torch.cat([x, ages], 1)
        return self.mlp(x)
    
class PixelSnail(nn.Module):
    def __init__(self, n_class, hidden_dim, n_layers, n_resblocks, n_output_layers, target_size,
                 n_cond_img = 0,  # channels for image type condition
                 n_cond_embed = 0,  # embedding dim for label type condition
                 n_cond_resblocks=0,  # number of resblocks for image type condition
                 attention=True, down_kernel=(2, 5), downright_kernel=(2, 3), hidden_kernel=(3, 3)
                 ):
        super().__init__()
        
        self.n_class = n_class
        
        self.down0 = CausalConv2d(
            n_class, hidden_dim, down_kernel, mode='down'
        )
        self.downright0 = CausalConv2d(
            n_class, hidden_dim, downright_kernel, mode='downright'
        )
        
        if n_cond_img > 0: #image conditions
            if n_cond_resblocks > 0:
                self.cond_net = [ConvNorm(n_class, n_cond_img, (3, 3), padding=1)]
                self.cond_net.extend([GatedResBlock(n_cond_img , (3, 3), n_cond_img) for _ in range(n_cond_resblocks)])
                self.cond_net = nn.Sequential(*self.cond_net)
            else:
                self.cond_net = ConvNorm(n_class, n_cond_img , (3, 3), padding=1)
            cond_channels = n_cond_img
            
        if n_cond_embed > 0: #label(discrete conditions)
            self.cond_embed = ConditionEmbedding(n_cond_embed)
            if n_cond_img > 0: cond_channels += n_cond_embed
            else: cond_channels = n_cond_embed
        
        self.layers = [PixelSnailBlock(hidden_dim, n_resblocks, hidden_kernel, target_size, 
                                       attention=attention, cond_channels=cond_channels) for _ in range(n_layers)]
        self.layers = nn.ModuleList(self.layers)
        # 1x1 convolutions
        if n_output_layers > 1:
            self.output_layers = [GatedResBlock11(hidden_dim, hidden_size=hidden_dim) for _ in range(n_output_layers-1)]
            self.output_layers.append(nn.ELU())
        else: self.output_layers = [nn.ELU()]
        self.output_layers.append(ConvNorm2d11(hidden_dim, n_class))
        self.output_layers = nn.Sequential(*self.output_layers)
        
    def forward(self, x, img_cond=None, label_cond=None):
        x = (
            F.one_hot(x, self.n_class).permute(0, 3, 1, 2).float()
        )
        x_down = shift_down(self.down0(x))
        x_downright = shift_right(self.downright0(x))
        
        out = x_down + x_downright
        
        cond = None
        if img_cond is not None:
            cond = (
                F.one_hot(img_cond, self.n_class).permute(0, 3, 1, 2).float()
            )
            cond = self.cond_net(cond) #img size is half(32)
            cond = F.interpolate(cond, scale_factor=2)
            cond = cond[:, :, :out.shape[-2], :]
            
        if label_cond is not None:
            label_cond = self.cond_embed(label_cond) #label
            label_cond = label_cond[..., None, None]
            # concat to image condition
            if img_cond is not None:
                label_cond = label_cond.expand_as(cond)
                cond = torch.cat([cond, label_cond], 1)
            else:
                cond = label_cond 
        
        for layer in self.layers:
            out = layer(out, cond)
        
        out = self.output_layers(out)
        
        return out
