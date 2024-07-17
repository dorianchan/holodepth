import base_models
import patterns
import camera_sim
import numpy as np
import cv2
import opt_models
from units import *
import torch
from torch.utils.data import Dataset
import lib

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
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) / 255.0
    
    return cv2.resize(img, (w, h))

def save_stack(tag, imgs):
    for i in range(len(imgs)):
        cv2.imwrite(tag + f"{i:03d}.png", np.clip(imgs[i], 0, 1) * 255)

if __name__ == "__main__":

    cam = camera_sim.Camera((h, w))
    
    # Sample training loop for optimized model

    class CalibDataset(Dataset):
        def __len__(self):
            pass # FIXME

        def __getitem__(self, idx):
            pass # FIXME
            # return torch.tensor(phase, dtype=lib.tdtype, device=lib.device), torch.tensor(cap, dtype=lib.tdtype, device=lib.device)

    calib_params = dict()
    calib_params["slm_distort.magnitude"] = 2e-3
    calib_params["slm_distort.angle"] = 2e-3
    calib_params["kernel_distort.magnitude"] = 2e-3
    calib_params["kernel_distort.angle"] = 2e-3
    calib_params["phase_mask.magnitude"] = 2e-3
    calib_params["phase_mask.angle"] = 2e-3
    calib_params["back_distort.magnitude"] = 2e-3
    calib_params["back_distort.angle"] = 2e-3
    calib_params["back_mask.magnitude"] = 2e-3
    calib_params["back_mask.angle"] = 2e-3
    calib_params["mult_back.magnitude"] = 2e-4
    calib_params["mult_back.angle"] = 2e-3
    calib_params["additive.magnitude"] = 2e-5
    calib_params["additive.angle"] = 2e-5

    epochs = 100
    lr_scale = 1e-2
    lens_dist = 62 * mm
    zs = np.array([0])
    pixel_scale = 5
    model_path = "calib/models"
    vis_path = "calib/vis"
    verbose = 100
    vis_interval = 500
    save_interval = 2000

    dset = CalibDataset()

    proj = opt_models.OptPhaseMaskProj(lens_dist, zs, fourier_focal_length=fourier_focal_length, slm_pixel_size=slm_pixel_size, slm_res=(h,w), wavelength=wavelength, mask_scale=pixel_scale)

    scale = torch.tensor(1, dtype=lib.tdtype, device=lib.device, requires_grad=True)
    opt_params = [{"params": scale, "lr": lr_scale}]
    
    print("opt params:")
    for param_name, param in proj.named_parameters():
        print(param_name)
        if param_name in calib_params:
            print(param_name, "found")
            opt_params.append({"params": param, "lr": calib_params[param_name]})
        else:
            flag = False
            for train_param_name in calib_params:
                if param_name.startswith(train_param_name + "."):
                    opt_params.append({"params": param, "lr": calib_params[train_param_name]})
                    flag = True
                    break
            
            if not flag:
                print(param_name, "NOT FOUND!!!")
            else:
                print("Applying startswith", train_param_name)
    
    optimizer = torch.optim.Adam(opt_params)
    

    def save(epoch, i): # saving only latest data to save disk space.
        torch.save(proj.state_dict(), f"{model_path}/model.pt")
        torch.save(optimizer.state_dict(),  f"{model_path}/opt.pt")
        np.savez_compressed(f"{model_path}/metadata.npz", scale=scale.item())

    def vis_proj(iters):
        vis = proj.vis()
        print("Visualizing...")
        for param_name in vis:
            cv2.imwrite(f"{vis_path}/{param_name}.png", 255 * vis[param_name])


    def mse_loss(sim, targ):
        return torch.nn.functional.mse_loss(sim, targ)

    print("Starting training loop...")
    accum_err = 0
    iters = 0
    for epoch in range(epochs):
        try:
            for i in range(len(dset)):
                phase, cap = dset[i]
                optimizer.zero_grad()
                slm_field = torch.exp(1j * phase)
                
                out_fields = proj.forward(slm_field)
                out_ints = cam(out_fields)

                total_err = mse_loss(scale * out_ints, cap[None])

                total_err.backward()
                optimizer.step()

                with torch.no_grad():
                    accum_err = accum_err + total_err.detach().item()
                    if iters > 0 and (iters % verbose == 0):
                        print(f"{epoch}/{epochs} {i} / {len(dset)} {accum_err}")
                        accum_err = 0
                    if iters % vis_interval == 0:
                        vis_proj(iters)
                    if iters % save_interval == 0:
                        save(epoch, i)

                iters += 1

                    
        except KeyboardInterrupt:
            print("early interrupt.")
            break
    save(epoch, 0)