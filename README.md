# Image Deblurring based on Diffusion Models

The code and pre-trained models of the paper "Image Deblurring based on Diffusion Models" will be released in this
repository.

<hr />

> **Abstract:** *This article introduces a sliding window model for defocus deblurring that achieves the best
performance to date with extremely low memory usage. Named Swintormer, the method utilizes a diffusion model to generate
latent prior features that assist in restoring more detailed images. It also extends the sliding window strategy to
specialized Transformer blocks for efficient inference. Additionally, we have further optimized Multiply-Accumulate
operations (Macs). Compared to the currently top-performing GRL method, our Swintormer model drastically reduces
computational complexity from 140.35 GMACs to 8.02 GMacs, while also improving the Signal-to-Noise Ratio (SNR) for
defocus deblurring from 27.04 dB to 27.07 dB. This new method allows for the processing of higher resolution images on
devices with limited memory, significantly expanding potential application scenarios. The article concludes with an
ablation study that provides an in-depth analysis of the impact of each network module on final performance. The source
code and model will be available at the following website: https://github.com/bnm6900030/swintormer.*
<hr />

## Installation
- Python 3.8.10
- PyTorch 2.0.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'swintormer'.
git clone https://github.com/bnm6900030/swintormer.git
cd swintormer
pip install -r requirements.txt
```

## Training
python basicsr/train.py -opt /home/lab/code1/IR/options/train/swintormer/train_swintormer.yml

## Testing
python basicsr/test.py

## Visual Results

Part visual results are available below. More visual results will come soon.

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR) and [Restormer](https://github.com/swz30/Restormer).

## Contact

If you have any question, please contact chenkang@cau.edu.cn
