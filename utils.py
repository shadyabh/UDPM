import torch
import torch.nn.functional as F
import torch.fft
import numpy as np
import lpips

def flip(x):
    return x.flip([-2, -1])


def flip_circ(x):
    x_ = torch.flip(torch.roll(x, ((x.shape[2] // 2), (x.shape[3] // 2)), dims=(-2, -1)), dims=(-2, -1))
    return torch.roll(x_, (- (x_.shape[2] // 2), -(x_.shape[3] // 2)), dims=(-2, -1))


def fft_circ(x, s=None, zero_centered=True):
    # s = (Ny, Nx)
    H, W = x.shape[-2:]
    if s == None:
        s = (H, W)
    if zero_centered:
        x_ = torch.roll(x, ((H // 2), (W // 2)), dims=(-2, -1))
    else:
        x_ = x
    x_pad = torch.nn.functional.pad(x_, (0, s[1] - W, 0, s[0] - H))
    x_pad_ = torch.roll(x_pad, (- (H // 2), -(W // 2)), dims=(-2, -1))
    return torch.fft.fftn(x_pad_, dim=(-2, -1))

def rfft_circ(x, s=None, zero_centered=True):
    # s = (Ny, Nx)
    H, W = x.shape[-2:]
    if s == None:
        s = (H, W)
    if zero_centered:
        x_ = torch.roll(x, ((H // 2), (W // 2)), dims=(-2, -1))
    else:
        x_ = x
    x_pad = torch.nn.functional.pad(x_, (0, s[1] - W, 0, s[0] - H))
    x_pad_ = torch.roll(x_pad, (- (H // 2), -(W // 2)), dims=(-2, -1))
    return torch.fft.rfftn(x_pad_, dim=(-2, -1))


def build_flt(f, size):
    is_even_x = not size[1] % 2
    is_even_y = not size[0] % 2

    grid_x = np.linspace(-(size[1] // 2 - is_even_x * 0.5), (size[1] // 2 - is_even_x * 0.5), size[1])
    grid_y = np.linspace(-(size[0] // 2 - is_even_y * 0.5), (size[0] // 2 - is_even_y * 0.5), size[0])

    x, y = np.meshgrid(grid_x, grid_y)

    h = f(x, y)
    h = np.roll(h, (- (h.shape[0] // 2), -(h.shape[1] // 2)), (0, 1))

    return torch.tensor(h).float().unsqueeze(0).unsqueeze(0)


def shift_by(H, shift):
    k_x = np.linspace(0, H.shape[-1] - 1, H.shape[-1])
    k_y = np.linspace(0, H.shape[-2] - 1, H.shape[-2])

    k_x[((k_x.shape[0] + 1) // 2):] -= H.shape[-1]
    k_y[((k_y.shape[0] + 1) // 2):] -= H.shape[-2]

    exp_x, exp_y = np.meshgrid(np.exp(-1j * 2 * np.pi * k_x * shift / H.shape[3]),
                               np.exp(-1j * 2 * np.pi * k_y * shift / H.shape[2]))

    exp_x_torch = (torch.tensor(np.real(exp_x), device=H.device, dtype=H.dtype) +
                   1j * torch.tensor(np.imag(exp_x), device=H.device, dtype=H.dtype)).unsqueeze(0).unsqueeze(0)
    exp_y_torch = (torch.tensor(np.real(exp_y), device=H.device, dtype=H.dtype) +
                   1j * torch.tensor(np.imag(exp_y), device=H.device, dtype=H.dtype)).unsqueeze(0).unsqueeze(0)

    return H * exp_x_torch * exp_y_torch


def get_box(supp, size=None):
    if size == None:
        size = (supp[0] * 2, supp[1] * 2)

    is_odd_x = supp[1] % 2
    is_odd_y = supp[0] % 2
    h = np.zeros(size)

    h[0:supp[0] // 2 + is_odd_y, 0:supp[1] // 2 + is_odd_x] = 1
    h[0:supp[0] // 2 + is_odd_y, -(supp[1] // 2):] = 1
    h[-(supp[0] // 2):, 0:supp[1] // 2 + is_odd_x] = 1
    h[-(supp[0] // 2):, -(supp[1] // 2):] = 1

    return torch.tensor(h).float().unsqueeze(0).unsqueeze(0)


def get_box_odd(supp, size=None):
    if size == None:
        size = (supp[0] * 2, supp[1] * 2)

    h = np.zeros(size)

    h[0:supp[0] // 2 + 1, 0:supp[1] // 2] = 1
    h[0:supp[0] // 2 + 1, -(supp[1] // 2):] = 1
    h[-(supp[0] // 2):, 0:supp[1] // 2 + 1] = 1
    h[-(supp[0] // 2):, -(supp[1] // 2) + 1:] = 1

    return torch.tensor(h).float().unsqueeze(0).unsqueeze(0)


def fft_Filter_(x, h):
    H = fft_circ(h, s=x.shape[-2:])
    X_fft = torch.fft.fftn(x, dim=(-2, -1))
    HX = H * X_fft
    return torch.fft.ifftn(HX, dim=(-2, -1))


def zero_SV(H, eps):
    abs_H = torch.abs(H)
    H[abs_H / abs_H.max() <= eps] = 0
    return H


def fft_Down_(x, h, alpha):
    X_fft = torch.fft.fftn(x, dim=(-2, -1))
    H = fft_circ(h, s=x.shape[-2:])
    HX = H * X_fft
    margin = (alpha - 1) // 2
    y = torch.fft.ifftn(HX, dim=(-2, -1))[:, :, margin:HX.shape[-2] - margin:alpha, margin:HX.shape[-1] - margin:alpha]
    return y


def rfft_Down_(x, h, alpha):
    X_fft = torch.fft.rfftn(x, dim=(-2, -1))
    H = rfft_circ(h, s=x.shape[-2:])
    HX = H * X_fft
    margin = (alpha - 1) // 2
    y = torch.fft.irfftn(HX, dim=(-2, -1))[:, :, margin:HX.shape[-2] - margin:alpha, margin:HX.shape[-1] - margin:alpha]
    return y


def fft_Up_(y, h, alpha):
    shape = y.shape
    x = torch.zeros(*shape[:-2], y.shape[-2] * alpha, y.shape[-1] * alpha, device=y.device)
    H = fft_circ(h, s=x.shape[-2:])
    start = alpha // 2
    x[:, :, start::alpha, start::alpha] = y
    X = torch.fft.fftn(x, dim=(-2, -1))
    HX = H * X
    return torch.fft.ifftn(HX, dim=(-2, -1))


@torch.no_grad()
def EMA_update(acc_state_dict, new_state_dict, EMA_weight):
    for key, val in acc_state_dict.items():
        acc_state_dict[key] = EMA_weight * val + (1 - EMA_weight) * new_state_dict[key]

    return acc_state_dict


class LPIPS(lpips.LPIPS):
    def forward(self, in0, in1, retPerLayer=False, normalize=False, p=2.0):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = torch.abs(feats0[kk]-feats1[kk])**p

        if(self.lpips):
            if(self.spatial):
                res = [lpips.upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [lpips.spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [lpips.upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [lpips.spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val
