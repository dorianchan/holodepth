# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Base models for holographic projectors, including naive and etendue-expanded.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import numpy as np
import lib
import holo_lib
from units import *



class Proj:
    def __init__(self, slm_pixel_size = 8 * um, slm_res = (1200, 1920), wavelength = 530 * nm):
        self.slm_pixel_size = slm_pixel_size
        self.slm_res = slm_res
        self.wavelength = wavelength

class FourierProj(Proj):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # A naive holographic projector based on the Fourier transform property of a lens.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    def __init__(self, zs, \
                 fourier_focal_length = 75 * mm, slm_pixel_y = 8 * um, slm_pixel_x = 8 * um, slm_res = (1200, 1920), wavelength = 530 * nm, circ_conv=True):
        super().__init__(slm_pixel_y, slm_res, wavelength)

        print("NAIVE")
        self.fourier_focal_length = fourier_focal_length

        self.front_to_back = holo_lib.FourierProp(self.fourier_focal_length)
        back_spacing_y = (wavelength * fourier_focal_length / slm_pixel_y) / slm_res[0]
        back_spacing_x = (wavelength * fourier_focal_length / slm_pixel_x) / slm_res[1]

        print("FOV:", back_spacing_y * slm_res[0], back_spacing_x * slm_res[1])

        self.zs = zs
        self.back_to_zs = holo_lib.ASMProp(slm_res[0], slm_res[1], back_spacing_y, back_spacing_x, wavelength, self.zs, circ_conv=circ_conv)


    def forward(self, slm_field):
        back_field = self.front_to_back.forward(slm_field)
        out_fields = self.back_to_zs.forward(back_field)
        return out_fields

class LensArrayProj(Proj):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # A Fourier holographic projector with etendue expanded by a lens array.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    def __init__(self, zs, lens_offset=(0, 0), lens_pitch=(1.4 * mm, 1.0 * mm), lens_focal_length=4.7 * mm, lens_pixel_y=8 * um / 4.5, lens_pixel_x=8 * um / 3.2, fourier_focal_length=75 * mm, slm_pixel_size=8 * um, slm_res=(1200, 1920), \
        wavelength=530 * nm, circ_conv=True):
        super().__init__(slm_pixel_size, slm_res, wavelength)
        print("LENS ARRAY")
        lens_scale_y = slm_pixel_size / lens_pixel_y
        lens_scale_x = slm_pixel_size / lens_pixel_x
        lens_res = (int(slm_res[0] * lens_scale_y), int(slm_res[1] * lens_scale_x))
        slm_lens_y = np.arange(lens_res[0]) * lens_pixel_y - lens_offset[0]
        slm_lens_x = np.arange(lens_res[1]) * lens_pixel_x - lens_offset[1]
        slm_lens_yy, slm_lens_xx = np.meshgrid(slm_lens_y, slm_lens_x, indexing='ij')


        lens_cy = (slm_lens_yy // lens_pitch[0]) * lens_pitch[0] + 0.5 * lens_pitch[0]
        lens_cx = (slm_lens_xx // lens_pitch[1]) * lens_pitch[1] + 0.5 * lens_pitch[1]

        lens_phase = -2 * np.pi / wavelength * 1 / (2 * lens_focal_length) * ((slm_lens_yy - lens_cy) ** 2 + (slm_lens_xx - lens_cx) ** 2)
        self.lens_mod = torch.exp(1j * torch.tensor(lens_phase, dtype=lib.tdtype, device=lib.device))

        self.upsample = torch.nn.Upsample(size=lens_res, mode='nearest')

        self.fourier_focal_length = fourier_focal_length
        self.front_to_back = holo_lib.FourierProp(self.fourier_focal_length)
        self.back_spacing_y = (wavelength * fourier_focal_length / lens_pixel_y) / lens_res[0]
        self.back_spacing_x = (wavelength * fourier_focal_length / lens_pixel_x) / lens_res[1]

        print("FOV:", self.back_spacing_y * lens_res[0], self.back_spacing_x * lens_res[1])

        self.zs = zs
        self.back_to_zs = holo_lib.ASMProp(lens_res[0], lens_res[1], self.back_spacing_y, self.back_spacing_x, wavelength, self.zs, circ_conv=circ_conv)

    def forward(self, slm_field):
        slm_field_arr = lib.complex_to_arr(slm_field, coord='rect')
        front_field = lib.arr_to_complex(self.upsample(slm_field_arr), coord='rect')  * self.lens_mod

        back = self.front_to_back.forward(front_field)

        out = self.back_to_zs.forward(back)
        return out


    