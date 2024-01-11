import numpy as np
import os
import argparse
from tqdm import tqdm
import math
import torch
import cv2
from skimage import metrics
from sklearn.metrics import mean_absolute_error
from natsort import natsorted
from glob import glob
from basicsr.archs.eff_arch import Eff as model_r
import lpips
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def MAE(img1, img2):
    mae_0 = mean_absolute_error(img1[:, :, 0], img2[:, :, 0],
                                multioutput='uniform_average')
    mae_1 = mean_absolute_error(img1[:, :, 1], img2[:, :, 1],
                                multioutput='uniform_average')
    mae_2 = mean_absolute_error(img1[:, :, 2], img2[:, :, 2],
                                multioutput='uniform_average')
    return np.mean([mae_0, mae_1, mae_2])


def PSNR(img1, img2):
    mse_ = np.mean((img1 - img2) ** 2)
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)


def SSIM(img1, img2):
    return metrics.structural_similarity(img1, img2, data_range=1, channel_axis=-1)


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def load_img16(filepath):
    return cv2.cvtColor(cv2.imread(filepath, -1), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


alex = lpips.LPIPS(net='alex').cpu()

parser = argparse.ArgumentParser(description='Dual Pixel Defocus Deblurring using Restormer')
parser.add_argument('--input_dir', default='/home/lab/code1/Defocus_Deblurring/Datasets/test/', type=str,help='Directory of validation images')
parser.add_argument('--result_dir',
                    default='/home/lab/code1/MYIR/datasets/DPDD/val2/', type=str,
                    help='Directory for results')
parser.add_argument('--weights',
                    default='/home/lab/code1/IR/experiments/train_MYIR_scratch/models/net_g_20000.pth', type=str,
                    help='Path to weights')
parser.add_argument('--save_images', default=True, help='Save denoised images in result directory')
args = parser.parse_args()

####### Load yaml #######
yaml_file = '/home/lab/code1/IR/options/train/swintormer/train_swintormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
device = torch.device("cuda")
model_restoration = model_r(**x['network_g'])
device_id = torch.cuda.current_device()
checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage.cuda(device_id))
model_restoration.load_state_dict(checkpoint['params'])
model_restoration.eval()

result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesL = natsorted(glob(os.path.join(args.input_dir, 'inputL', '*.png')))
filesR = natsorted(glob(os.path.join(args.input_dir, 'inputR', '*.png')))
filesC = natsorted(glob(os.path.join(args.input_dir, 'target', '*.png')))

indoor_labels = np.array(
    [8, 9, 10, 11, 12, 13, 16, 32, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 59, 60, 61,
     62, 63, 64, 65, 66, 67, 74, 75])
outdoor_labels = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 17, 18, 19, 20, 21, 22, 23,
     24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 57, 58, 68,
     69, 70, 71, 72, 73])


# ---------------------

def grid(x, h, w, size):
    step_h = h if h < size else 220
    step_w = w if w < size else 220
    parts = []
    idxes = []
    i = 0  # 0~h-1
    last_i = False
    while i < h and not last_i:
        j = 0
        if i + size >= h:
            i = h - size
            last_i = True
        last_j = False
        while j < w and not last_j:
            if j + size >= w:
                j = w - size
                last_j = True
            parts.append(x[:, :, i:i + size, j:j + size])
            idxes.append({'i': i, 'j': j})
            j = j + step_w
        i = i + step_h

    parts = torch.cat(parts, dim=0)
    return parts, idxes


def grid_verse(outs, idxes, h, w, size):
    preds = torch.zeros((1, 3, h, w)).to(outs.device)
    count_mt = torch.zeros((1, 1, h, w)).to(outs.device)
    for cnt, each_idx in enumerate(idxes):
        i = each_idx['i']
        j = each_idx['j']
        preds[0, :, i:i + size, j:j + size] += outs[cnt, :, :, :]
        count_mt[0, 0, i:i + size, j:j + size] += 1.
    del outs
    return preds / count_mt


# ---------------------

psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileL, fileR, fileC in tqdm(zip(filesL, filesR, filesC), total=len(filesC)):
        imgL = (np.float32(load_img16(fileL)) / 65535. )
        imgR = (np.float32(load_img16(fileR)) / 65535. )
        imgC = (np.float32(load_img16(fileC)) / 65535. )
        patchL = torch.from_numpy(imgL).unsqueeze(0).permute(0, 3, 1, 2)
        patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0, 3, 1, 2)
        input_ = torch.cat([patchR, patchL], 1)

        model_restoration = model_restoration.cuda()

        size = 256
        _, _, h, w = input_.shape
        grid_input, idxes = grid(input_, h, w, size)

        parts = []
        for i in range(0, grid_input.shape[0],30):
            parts.append(model_restoration(grid_input[i:i+30,:,:,:].cuda()))

        restored = torch.cat(parts, dim=0)
        restored = grid_verse(restored, idxes, h, w, size)

        # restored = model_restoration(input_.cuda())

        restored = torch.clamp(restored, 0, 1)[:, :, :1120, :1680]
        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        # caculate
        panr = PSNR(imgC, restored)
        psnr.append(panr)
        print(panr)
        mae.append(MAE(imgC, restored))
        ssim.append(SSIM(imgC, restored))

        pips.append(alex(torch.from_numpy(imgC).unsqueeze(0).permute(0, 3, 1, 2),
                         torch.from_numpy(restored).unsqueeze(0).permute(0, 3, 1, 2), normalize=True).item())
        if args.save_images:
            save_file = os.path.join(result_dir, os.path.split(fileC)[-1])
            restored = np.uint16((restored * 65535).round())
            # utils.save_img(save_file, restored)
            cv2.imwrite(save_file, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels], mae[indoor_labels], ssim[
    indoor_labels], pips[indoor_labels]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels], mae[outdoor_labels], ssim[
    outdoor_labels ], pips[outdoor_labels]

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae),
                                                                    np.mean(pips)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor),
                                                                    np.mean(mae_indoor), np.mean(pips_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor),
                                                                    np.mean(mae_outdoor), np.mean(pips_outdoor)))
