# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# A Camera class that models the incoherent integration performed by camera pixels.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import torch

class Camera(torch.nn.Module):
    def __init__(self, cam_res):
        super().__init__()
        self.cam_res = cam_res
        self.downsample = lambda x: torch.nn.functional.interpolate(x, size=cam_res, mode='area')


    def forward(self, fields):
        mags = torch.abs(fields).pow(2)

        out = []
        for o_p in mags:
            osig = self.downsample(o_p[None, None])[0, 0]
            out.append(osig)
        return torch.stack(out, dim=0)