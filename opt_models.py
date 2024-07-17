# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Code for optimizable models, including our proposed model in Eq. (8)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import holo_lib
import base_models

import torch
import numpy as np
import lib

import matplotlib.pyplot as plt
from units import *


cmapr = plt.get_cmap('hsv')
cmap = lambda img: cmapr(img)[..., :3]
def vis_phase(phase):
    return cmap(lib.normalize(phase % (2 * np.pi), 0, 2 * np.pi).detach().cpu().numpy())

class ComplexMod(torch.nn.Module):
    def __init__(self, res, init_abs = None, init_phase = None, init_abs_method='ones', init_phase_method='zeros', mode='mult'):
        super().__init__()
        if init_abs is None:
            if init_abs_method == 'ones':
                self.magnitude = torch.nn.Parameter(torch.ones(res, dtype=lib.tdtype, device=lib.device))
            else:
                self.magnitude = torch.nn.Parameter(torch.zeros(res, dtype=lib.tdtype, device=lib.device))
        else:
            self.magnitude = torch.nn.Parameter(torch.tensor(init_abs, dtype=lib.tdtype, device=lib.device))
        
        if init_phase is None:
            if init_phase_method == 'rand':
                self.angle = torch.nn.Parameter(torch.rand(res[0], res[1], dtype=lib.tdtype, device=lib.device) * 2 * np.pi)
            elif init_phase_method == 'zeros':
                self.angle = torch.nn.Parameter(torch.zeros(res, dtype=lib.tdtype, device=lib.device))
            else:
                print(f"Unrecognized init method {init_phase_method}.")
                exit()
        else:
            self.angle =  torch.nn.Parameter(torch.tensor(init_phase, dtype=lib.tdtype, device=lib.device))

        self.mode = mode

    def get_mod(self):
        mod = self.magnitude * torch.exp(1j * self.angle)
        return mod
    def forward(self, field):
        mod = self.get_mod()
        if self.mode == 'mult':
            return field * mod
        elif self.mode == 'add':
            return field + mod
        else:
            print("Unrecognized, exiting...")
            exit()
    
    def vis(self):
        mod = self.get_mod()

        mag = torch.abs(mod)
        angle = mod.angle()

        mag_vis = lib.normalize(mag.detach().cpu().numpy())
        angle_vis = vis_phase(angle)
        return {"magnitude": mag_vis, "phase": angle_vis}



class OptProj(base_models.Proj, torch.nn.Module):
    def __init__(self, slm_pixel_size, slm_res, wavelength):
        torch.nn.Module.__init__(self)
        super().__init__(slm_pixel_size, slm_res, wavelength)

    def vis(self):

        vis_out = {}
        for name, param in self.named_children():
            if isinstance(param, ComplexMod):
                print(name)
                vis_data = param.vis()

                for pname in vis_data:
                    vis_out[f"{name}_{pname}"] = vis_data[pname]
            elif torch.is_tensor(param):
                vis_out[name] = lib.normalize(param)
            else:
                print("Can't vis:", name)
        return vis_out
    

class OptPhaseMaskProj(OptProj):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Our proposed model, with learnable modulations and convolutions for 
    # various parts of the propagation process.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    def __init__(self, slm_to_mask_distance, zs, fourier_focal_length, slm_pixel_size, slm_res, \
        wavelength, mask_scale, circ_conv=True, init_aperture=None, init_phase=None):
        super().__init__(slm_pixel_size, slm_res, wavelength)

        # compute mask params
        self.mask_scale = mask_scale
        mask_pixel_size = slm_pixel_size / mask_scale
        mask_res = (slm_res[0] * mask_scale, slm_res[1] * mask_scale)
        self.mask_pixel_size = mask_pixel_size
        self.mask_res = mask_res

        self.slm_distort = ComplexMod(mask_res)


        self.kernel_distort = ComplexMod(mask_res)
        self.slm_to_mask = holo_lib.ASMProp(mask_res[0], mask_res[1], mask_pixel_size, mask_pixel_size, wavelength, [slm_to_mask_distance], \
            circ_conv=circ_conv)


        self.phase_mask = ComplexMod(mask_res, init_abs=init_aperture, init_phase=init_phase, init_phase_method='zeros')
        self.back_distort = ComplexMod(mask_res)
        self.mask_to_back = holo_lib.ASMProp(mask_res[0], mask_res[1], mask_pixel_size, mask_pixel_size, wavelength / 1.49, [1 * mm], \
            circ_conv=circ_conv)
            
        self.back_mask = ComplexMod(mask_res)


        self.additive = ComplexMod(mask_res, init_abs_method='zeros', init_phase_method='zeros', mode='add')
        self.mult_back = ComplexMod(mask_res)


        self.upsample = torch.nn.Upsample(scale_factor=mask_scale, mode='nearest')

        self.fourier_focal_length = fourier_focal_length
        self.front_to_back = holo_lib.FourierProp(self.fourier_focal_length)
        self.back_spacing_y = (wavelength * fourier_focal_length / mask_pixel_size) / mask_res[0]
        self.back_spacing_x = (wavelength * fourier_focal_length / mask_pixel_size) / mask_res[1]

        self.zs = zs
        if (len(zs) == 1) and zs[0] == 0:
            self.back_to_zs = None
        else:
            self.back_to_zs = holo_lib.ASMProp(mask_res[0], mask_res[1], self.back_spacing_y, self.back_spacing_x, wavelength, self.zs, circ_conv=True)

    def forward(self, slm_field):
        slm_field_arr = lib.complex_to_arr(slm_field, coord='rect')
        slm_field_up = self.slm_distort(lib.arr_to_complex(self.upsample(slm_field_arr), coord='rect'))

        field_pm = self.slm_to_mask.forward(slm_field_up, kernel_distort = self.kernel_distort.get_mod())[0]

        front_mask_field = self.phase_mask(field_pm)
        back_mask_before_field = self.mask_to_back.forward(front_mask_field, kernel_distort = self.back_distort.get_mod())[0]
        back_mask_field = self.back_mask(back_mask_before_field)

        back = self.additive(self.mult_back(self.front_to_back.forward(back_mask_field)))

        if self.back_to_zs is not None:
            out = self.back_to_zs.forward(back)
        else:
            out = back[None]
        return out


