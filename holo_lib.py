# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Code for holographic propagation
# Adapted from Holotorch (https://github.com/facebookresearch/holotorch/), 
# used under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import torch
from torch.nn.functional import pad
import lib


def ft2(field, norm='ortho'):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm=norm, dim=(-2, -1)), dim=(-2, -1))

def ift2(field, norm='ortho'):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm=norm, dim=(-2, -1)), dim=(-2, -1))


# ASM propagation
def compute_pad_size(H, W, circ_conv=True):
    if circ_conv == True:
        padding_x = 0
        padding_y = 0
    else:
        padding_x = 1    
        padding_y = 1    
        
    padW = int( (1 + padding_x) * W)
    padH = int( (1 + padding_y) * H)
    
    return padW, padH

def create_frequency_grid(H, W, circ_conv=True):

    padW, padH = compute_pad_size(H,W,circ_conv)

    ky = torch.linspace(-1/2, 1/2, padH, dtype=lib.tdtype, device=lib.device)
    kx = torch.linspace(-1/2, 1/2, padW, dtype=lib.tdtype, device=lib.device)        
    Ky, Kx = torch.meshgrid(ky, kx, indexing='ij')
    return Kx, Ky

def get_asm_kernel(h, w, spacing_y, spacing_x, wavelength, zs, circ_conv=True):
    Kx, Ky = create_frequency_grid(h, w, circ_conv)

    dx = spacing_x
    dy = spacing_y

    Kx = 2 * np.pi * Kx / dx
    Ky = 2 * np.pi * Ky / dy

    K2 = Kx**2 + Ky**2 

    K_lambda = 2 * np.pi / wavelength
    K_lambda_2 = K_lambda ** 2

    zs_exp = zs[:, None, None]
    K2_exp = K2[None, :, :]

    ang = zs_exp * torch.sqrt(K_lambda_2 - K2_exp)
    ang = torch.nan_to_num(ang)

    # size of the field
    # Total field size on the hologram plane
    length_x = h * dx 
    length_y = w * dy

    # band-limited ASM - Matsushima et al. (2009)
    f_y_max = 2*np.pi / torch.sqrt((2 * zs_exp * (1 / length_x) ) ** 2 + 1) / wavelength
    f_x_max = 2*np.pi / torch.sqrt((2 * zs_exp * (1 / length_y) ) ** 2 + 1) / wavelength

    H_filter = torch.zeros_like(ang)
    H_filter[ ( torch.abs(Kx[None, :, :]) < f_x_max) & (torch.abs(Ky[None, :, :]) < f_y_max) ] = 1

    kernels = H_filter * torch.exp(1j * H_filter * ang)
    return kernels

def to_tensor(var):
    if torch.is_tensor(var):
        return var

    if torch.is_tensor(var[0]):
        return torch.stack(var, dim=0)
    
    return torch.tensor(var, device=lib.device, dtype=lib.tdtype)

def sep(var):
    if torch.is_tensor(var):
        return var.detach().clone()
    else:
        return torch.tensor(var, device=lib.device, dtype=lib.tdtype)

class ASMProp:
    def __init__(self, h, w, spacing_y, spacing_x, wavelength, dists=None, circ_conv=True):
        self.h = h
        self.w = w
        self.spacing_y = spacing_y
        self.spacing_x = spacing_x
        self.wavelength = wavelength
        self.circ_conv = circ_conv
        if torch.is_tensor(dists):
            dists = dists.detach().clone()
        else:
            dists = torch.tensor(dists, device=lib.device, dtype=lib.tdtype)
        self.kernels = get_asm_kernel(self.h, self.w, self.spacing_y, self.spacing_x, self.wavelength, dists, circ_conv=self.circ_conv)

    def forward(self, field, kernel_distort = None):
        ondims = len(field.shape)
        if ondims == 2:
            field = field[None, :, :]
        
        if not self.circ_conv:
            size_diff = (self.kernels.shape[-2] - field.shape[-2], self.kernels.shape[-1] - field.shape[-1])
            pad_y = int(size_diff[0]/2)
            pad_x = int(size_diff[1]/2)
            field = pad(field, (pad_x,pad_x,pad_y, pad_y), mode='constant', value=0)

            if kernel_distort is not None:
                kernel_distort_sp = ift2(kernel_distort)
                kernel_distort_sp = pad(kernel_distort_sp, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)
                kernel_distort = ft2(kernel_distort_sp)

        field_f = ft2(field)

        ckernels = self.kernels
        if kernel_distort is not None:
            ckernels = ckernels * kernel_distort
        field_f_k = ckernels[None, :, :, :] * field_f[:, None, :, :]

        field_k = ift2(field_f_k)

        if not self.circ_conv:
            field_k = field_k[..., pad_y:-pad_y, pad_x:-pad_x]

        if ondims == 2:
            field_k = field_k[0]
        return field_k


class FourierProp:
    def __init__(self, focal_length = None):
        self.focal_length = sep(focal_length)

    def update_kernel(self, nfocal_length):
        self.focal_length = sep(nfocal_length)
    
    def forward(self, front_field, nfocal_length = None):
        if nfocal_length is not None:
            self.update_kernel(nfocal_length)

        back_field = ft2(front_field)

        return back_field