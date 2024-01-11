# -*- coding: utf-8 -*-
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from torchvision.utils import make_grid

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def show_ternor(tensor):
    img = tensor2img(tensor)
    plt.imshow(img)
    plt.show()

# =====================================================================


iter_pat = re.compile("(?<=iter:)(.*)(?=, lr)")
epoch_pat = re.compile("(?<=\]\[epoch:)(.*)(?=, iter:)")
loss_pat = re.compile("(?<=l_pix:)(.*)")
val_pat = re.compile("(?<=# psnr:)(.*)(?=Best: )")
log_time = re.compile("(?<=_........_)(.*)(?=.log)")
log_day = re.compile("(?<=_)(........)(?=_)")

epoch_vals = []
epoch_vals_dict = OrderedDict()
epoch_loss_dict = OrderedDict()

epoch_now = 0
iter_now = 0
iter_nums = []
loss_nums = []

val_iters = []
val_nums = []
log_path = os.listdir('/home/lab/code1/IR/experiments/Deblurring_Restormer/')
log_path = list(filter(lambda x: x.__contains__('.log'), log_path))


def get_time(name):
    return int(log_time.search(name)[0]) + int(log_day.search(name)[0]) * 1000000


log_path.sort(key=get_time)

epcho_num = 0
for log_f in log_path:
    with open('/home/lab/code1/IR/experiments/train_MYIR_scratch/' + log_f, 'r') as f:
        for line in f:
            if line.__contains__('[epoch'):
                iter_now = int(iter_pat.search(line)[0].replace(',', ''))
                loss = float(loss_pat.search(line)[0].replace(',', ''))
                epoch_now = int(epoch_pat.search(line)[0].replace(',', ''))
                epoch_loss_dict[epoch_now] = loss
                loss_nums.append(loss)
                iter_nums.append(iter_now)

                epcho_num += 1
            elif line.__contains__('# psnr:'):
                val_iters.append(iter_now)
                val_nums.append(float(val_pat.search(line)[0].replace(',', '')))
                epoch_vals_dict[epoch_now] = float(val_pat.search(line)[0].replace(',', ''))


plt.plot(epoch_vals_dict.keys(), epoch_vals_dict.values())
plt.plot(epoch_loss_dict.keys(), epoch_loss_dict.values())
plt.show()



