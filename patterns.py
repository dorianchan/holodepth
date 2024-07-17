# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Code for recovering a SLM pattern that creates a target depth-varying pattern.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import lib
import torch
import numpy as np

def recover_pattern_SGD_scale(holo_proj, cam, targets, iters=1000, lr_slm=2e-1, lr_scale=1e-1, verbose=False, mask=False, mode='even'):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Uses gradient descent, with an optimization scale term.
    # arguments:
    # # holo_proj - object of class base_models.Proj
    # # cam - object of class camera_sim.Camera
    # # targets - the target depth-varying pattern in form [D, H, W]
    # # iters - iterations of gradient descent
    # # lr_slm - learning rate for SLM phase
    # # lr_scale - learning rate for scale term
    # # verbose - whether to print error during training process.
    # # mask - if True, only computes loss over region where target is non-negative
    # # mode - how to select depth planes if the number of depth targets is less than 
    # # # # the number of depths the projector outputs. "even" evenly distributes the depth targets, 
    # # # # "center" concentrates them about the center of the projector depth range.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    if type(verbose) == bool:
        if verbose:
            verbose = 1
        else:
            verbose = None
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets, dtype=lib.tdtype, device=lib.device)
    h, w = holo_proj.slm_res
    init_phase = np.random.uniform(0, 2 * np.pi, size=(h, w))
    opt_phase = torch.tensor(init_phase, dtype=lib.tdtype, device=lib.device, requires_grad=True)
    
    tscale = 1 / (w / 1920 * h / 1200)
    scales = torch.tensor([tscale] * len(targets), dtype=lib.tdtype, device=lib.device, requires_grad=True)
    
    optimizer = torch.optim.Adam(
                        [   
                            {"params" : opt_phase,     "lr" : lr_slm},
                            {"params" : scales, "lr": lr_scale}
                        ]
                    )
    
    mse_loss = torch.nn.MSELoss()
    
    fact = targets.shape[1] * targets.shape[2] / targets.sum()
    def light_loss(output, target):
        return mse_loss(output, target)

    if len(targets.shape) > 2:
        num_target_planes = len(holo_proj.zs)
        if mode == "even":
            samp_ind = np.round(np.linspace(0, num_target_planes - 1, targets.shape[0])).astype(int)
        elif mode == "center":
            samp_ind = np.arange(targets.shape[0]) + (num_target_planes - targets.shape[0]) // 2
        else:
            raise Exception("Sampling mode not handled.")
    else:
        samp_ind = 0

    print("Target inds:", samp_ind)
    for iter in range(iters):
        try:
            optimizer.zero_grad()

            slm_field = torch.exp(1j * opt_phase)
            
            out_fields = holo_proj.forward(slm_field)
            out_ints = cam(out_fields)

            out_ints = out_ints * scales[:, None, None] / fact

            if mask:
                total_err = light_loss(out_ints[samp_ind] * (targets > 0), targets)
            else:
                total_err = light_loss(out_ints[samp_ind], targets)

            total_err.backward()
            optimizer.step()

            if verbose is not None and iter % verbose == 0:
                print(f"{iter}/{iters} {total_err.item()} {out_ints.max().item()}")
        except KeyboardInterrupt:
            print("early interrupt.")
            break

    
    return opt_phase.detach().cpu().numpy().squeeze() % (2 * np.pi), out_ints.detach().cpu().numpy().squeeze()