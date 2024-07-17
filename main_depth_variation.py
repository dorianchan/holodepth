import base_models
import patterns
import camera_sim
import numpy as np
import cv2
from units import *

wavelength = 530 * nm
microlens_f = 4.7 * mm
microlens_pitch = (1.4 * mm, 1.0 * mm)
microlens_offset = (0, 0)
fourier_focal_length = 75 * mm
exp_ratio = (4.5, 3.2)
slm_pixel_size = 8 * um
h, w = (1200, 1920)
wavelength = 530 * nm

def load_img(path, h, w):
    img = cv2.imread(path)[..., 0] / 255.0
    
    return cv2.resize(img, (w, h))

def save_stack(tag, imgs):
    for i in range(len(imgs)):
        cv2.imwrite(tag + f"{i:03d}.png", np.clip(imgs[i], 0, 1) * 255)

if __name__ == "__main__":

    cam = camera_sim.Camera((h, w))
    
    # Testing etendue-expanded vs naive resized system
    print("Testing etendue expansion of depth variation for a fixed FOV...")
    zs = np.array([-0.0075, -0.0025])
    pixel_y = slm_pixel_size/exp_ratio[0]
    pixel_x = slm_pixel_size/exp_ratio[1]
    targets = np.array([load_img("images/canyon.jpg", h, w), load_img("images/train.jpg", h, w)])

    naive_proj = base_models.FourierProj(zs, fourier_focal_length=fourier_focal_length, slm_pixel_y=pixel_y, slm_pixel_x=pixel_x, \
        slm_res=(h, w), wavelength=wavelength)

    _, naive_output = patterns.recover_pattern_SGD_scale(naive_proj, cam, targets, verbose=10)

    save_stack("etendue_test/naive_", naive_output)

    etendue_proj = base_models.LensArrayProj(zs, lens_offset=microlens_offset, lens_pitch=microlens_pitch, lens_focal_length=microlens_f, lens_pixel_y=pixel_y, 
                            lens_pixel_x=pixel_x, fourier_focal_length=fourier_focal_length, slm_pixel_size=slm_pixel_size, slm_res=(h, w), wavelength=wavelength)
    _, etendue_output = patterns.recover_pattern_SGD_scale(etendue_proj, cam, targets, verbose=10)

    save_stack("etendue_test/etendue_", etendue_output)
