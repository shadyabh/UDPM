import torch.nn.functional
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from EDM_nets import Conv2d, Linear, GroupNorm, UNetBlock, AttentionOp, PositionalEmbedding
import numpy as np
# from spectrum import spectrum_transform, spectrum_transform_after_mean
from collections import OrderedDict

class DiscBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(), init_attn=None,
    ):
        super().__init__()
        norm = spectral_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = norm(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, bias=True, **init))
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = norm(Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, bias=True))

        self.act = torch.nn.LeakyReLU(0.2, inplace=False)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        x = self.conv0(self.act(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = self.act(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = self.act(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class DDGAN_Discriminator(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, dropout=0, num_classes=None, n_scales=5):
        super(DDGAN_Discriminator, self).__init__()
        norm = lambda x: x

        self.dropout = dropout
        self.map_noise = PositionalEmbedding(num_channels=num_feat)

        # Input:
        self.conv0 = nn.Conv2d(num_in_ch, min(num_feat, 512), kernel_size=1, stride=1, padding=0)

        body = []
        for s in range(n_scales):
            body.append(UNetBlock(in_channels=min(num_feat * 2 ** s, 512), out_channels=min(num_feat * 2 ** (s+1), 512), 
                                  emb_channels=num_feat, down=True if s > 0 else False, dropout=dropout, init_zero=dict(init_weight=1)))
        
        self.body = nn.ModuleList(body)

        self.final_conv = nn.Conv2d(in_channels=min(num_feat * 2 ** (s+1),512) + 1, out_channels=min(num_feat * 8, 512), kernel_size=3, padding=1)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.num_classes = num_classes
        if num_classes is not None and num_classes > 1:
            self.map_label = Linear(in_features=num_classes, out_features=num_feat)

        # Output:
        self.out_lin = nn.Linear(min(num_feat*8, 512), 1)

    def forward(self, x, t, cls=None, dp=None):
        # Input:
        emb = self.map_noise(t)
        if self.num_classes is not None and cls is not None:
            c_emb = self.map_label(torch.nn.functional.one_hot(cls, self.num_classes).to(x.dtype))
            emb = emb + c_emb
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=False) 

        for l in self.body:
            x0 = F.leaky_relu(l(x0, emb, dp=dp), negative_slope=0.2, inplace=False)

        x5 = x0
        
        batch, channel, height, width = x5.shape
        group = min(batch, self.stddev_group)
        stddev = x5.view(
                group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
            )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.sum([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([x5, stddev], 1)

        out = F.leaky_relu(self.final_conv(out), negative_slope=0.2, inplace=False)

        # Output
        out = torch.nn.functional.dropout(self.out_lin(out.view(out.shape[0], out.shape[1], -1).sum(2)), 
                                            p=dp if dp is not None else 0, training=self.training)

        return out

def get_disc(args):
    n_scales = int(np.log2(args.img_size//(args.diffusion_scale ** (args.diffusion_steps-1)))) + 1
    disc = DDGAN_Discriminator(num_in_ch=3, num_feat=args.disc_channels, dropout=args.disc_dropout[0] if len(args.disc_dropout) == 1 else 0,
                                num_classes=args.classes_num if args.class_disc else 0, n_scales=n_scales)
    
    return disc