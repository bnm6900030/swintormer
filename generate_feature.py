import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from print_loss import show_tersor


@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False,
                    temperature=1., score_corrector=None, corrector_kwargs=None, x_T=None,
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]  # batch size
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_T=x_T)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, custom_steps=None, eta=1.0, quantize_x0=False, custom_shape=None,
                              temperature=1., corrector=None,
                              corrector_kwargs=None, x_T=None, ddim_use_x0_pred=False):
    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)
    c = batch[model.cond_stage_key]
    c = rearrange(c, 'b h w c -> b c h w')
    c = c.to(memory_format=torch.contiguous_format).float()

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    with model.ema_scope("Plotting"):
        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                eta=eta,
                                                quantize_x0=quantize_x0, mask=None, x0=z0,
                                                temperature=temperature,
                                                score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_T=x_T, )

        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    # ------------------------For small video memory----------------------

    # size = 128
    # _, _, h, w = c.shape
    # grid_input, idxes = grid(c, h, w, size)
    # parts = []
    # for i in range(0, grid_input.shape[0], 2):
    #     qwe,_,_,_ =grid_input[i:i + 2, :, :, :].shape
    #     sample, _ = convsample_ddim(model, grid_input[i:i + 2, :, :, :], steps=custom_steps, shape=grid_input[i:i + 2, :, :, :].shape,
    #                                             eta=eta,
    #                                             quantize_x0=quantize_x0, mask=None, x0=None,
    #                                             temperature=temperature,
    #                                             score_corrector=corrector, corrector_kwargs=corrector_kwargs,
    #                                             x_T=x_T[:qwe,:,:,:], )
    #     parts.append(sample)
    # restored = torch.cat(parts, dim=0)
    # x_sample = grid_verse(restored, idxes, h*4, w*4, size * 4)

    return sample


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def check_image_size(x, window_size=8):
    _, _, h, w = x.size()

    pad_size = window_size * 8
    mod_pad_h = (pad_size - h % pad_size) % pad_size
    mod_pad_w = (pad_size - w % pad_size) % pad_size
    try:
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    except BaseException:
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
    return x


def get_cond(selected_path):
    example = dict()
    up_f = 4

    c = Image.open(selected_path)
    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    c = check_image_size(c, 16)

    c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], )
    c_up = rearrange(c_up, '1 c h w -> 1 h w c')
    c = rearrange(c, '1 c h w -> 1 h w c')
    c = 2. * c - 1.

    c = c.to(torch.device("cuda"))
    example["LR_image"] = c
    example["image"] = c_up

    return example


def run(model, selected_path, custom_steps, custom_shape):
    example = get_cond(selected_path)

    guider = None
    ckwargs = None
    ddim_use_x0_pred = False
    temperature = 1.
    eta = 1.

    height, width = example["image"].shape[1:3]
    split_input = height >= 128 and width >= 128

    if split_input:
        ks = 128
        stride = 64
        vqf = 4  #
        model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                                    "vqf": vqf,
                                    "patch_distributed_vq": True,
                                    "tie_braker": False,
                                    "clip_max_weight": 0.5,
                                    "clip_min_weight": 0.01,
                                    "clip_max_tie_weight": 0.5,
                                    "clip_min_tie_weight": 0.01}
    else:
        if hasattr(model, "split_input_params"):
            delattr(model, "split_input_params")

    x_T = None

    if custom_shape is not None:
        x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
        x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

    logs = make_convolutional_sample(example, model,
                                     custom_steps=custom_steps,
                                     eta=eta, quantize_x0=False,
                                     custom_shape=custom_shape,
                                     temperature=temperature,
                                     corrector=guider, corrector_kwargs=ckwargs, x_T=x_T,
                                     ddim_use_x0_pred=ddim_use_x0_pred
                                     )
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./options/config.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ldm/bsr_sr/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--custom_steps",
        type=int,
        default=20,
        help="the iteration numbers T",
    )
    parser.add_argument(
        "--custom_shape",
        type=int,
        default=None,
        help="The size of the output tensor."
             "The size of the output tensor does not have to be the size of the image, it can be any size tensor",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    custom_steps = opt.custom_steps
    custom_shape = [1, 3, 512, 512]
    # custom_shape = opt.custom_shape

    device = torch.device("cuda")
    model = model.to(device)
    img_path = '/home/lab/code1/MYIR/datasets/DPDD1/train/C/1P0A0890-1.png'
    save_path = '/home/lab/tmp/save/1P0A0890-1.png'
    with torch.no_grad():
        sample = run(model, img_path, custom_steps, custom_shape)
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2.

        show_tersor(sample)

        sample = np.uint16((sample * 65535).round())
        sample = np.transpose(sample, (0, 2, 3, 1))
        # cv2.imwrite(save_path, cv2.cvtColor(sample[0], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
