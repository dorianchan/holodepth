# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# General helper functions.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import torch
tdtype = torch.float32
cdtype = torch.complex64
device = torch.device("cuda")


def complex_to_arr(field, coord, pad=None):
    if coord == 'rect':
        field_proc = torch.stack((field.real, field.imag), dim=0)
    elif coord == 'polar':
        field_proc = torch.stack((field.abs(), field.angle()), dim=0)

    if pad is not None:
        field_proc = torch.nn.functional.pad(field_proc, (0, pad[1], 0, pad[0]), mode='constant', value=0)

    return field_proc[None, :, :, :]

def arr_to_complex(arr, coord, crop=None):
    field_proc = arr[0]
    if coord == 'rect':
        out = field_proc[0] + 1j * field_proc[1]
    elif coord == 'polar':
        out = field_proc[0] * torch.exp(1j * field_proc[1])

    if crop is not None:
        out = out[..., :crop[0], :crop[1]]
    return out


def normalize(img, min=None, max=None):
    if min is None:
        min = img.min()
    if max is None:
        max = img.max()
    return (img - min) / (max - min)
