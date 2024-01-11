import math
from fairscale.nn import checkpoint_wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torch.nn.init import trunc_normal_

from basicsr.utils.registry import ARCH_REGISTRY


##########################################################################
## Layer Norm

def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    # subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    # compute exponentials
    exp_x = torch.exp(x)
    # compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.body(x)
        x = x.permute(0, 3, 1, 2)
        return x


##########################################################################
#Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, shift_size, window_size):
        super(Attention, self).__init__()
        self.dim = dim
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )
        self.bias = bias
        self.shift_size = shift_size
        self.window_size = window_size

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([
            relative_coords_h,
            relative_coords_w], indexing="ij")).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def window_partition(self, x, window_size):
        """
        Args:
            x: (b,c, h, w, )
            window_size (int): window size

        Returns:
            windows: (num_windows*b, c, window_size, window_size,)
        """
        b, c, h, w, = x.shape
        x = x.view(b, c, h // window_size, window_size, w // window_size, window_size, )
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, window_size, window_size, )
        return windows

    def calculate_mask(self, h, w):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, 1, h, w,))  # 1 1 h w
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w, ] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = set()
        for n, m in self.named_modules():
            if any([kw in n for kw in ("cpb_mlp", "logit_scale", 'relative_position_bias_table')]):
                nod.add(n)
        return nod

    def window_reverse(self, windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*b,c, window_size, window_size, )
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b,c, h, w, )
        """
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, -1, window_size, window_size, )
        x = x.permute(0, 3, 1, 4, 2, 5, ).contiguous().view(b, -1, h, w, )
        return x

    def window_attn(self, qkv, h, w, wb, wh, ww):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c ', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if self.shift_size > 0:
            mask = self.calculate_mask(h, w)
        else:
            mask = None
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(wb // nw, nw, self.num_heads, wh * ww, wh * ww) + mask.unsqueeze(1).unsqueeze(0).to(
                attn.device)
            attn = attn.view(-1, self.num_heads, wh * ww, wh * ww)

        # attn = attn.softmax(dim=-1)
        attn = softmax_one(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=wh, w=ww)

        return out

    def channel_attn(self, qkv, wh, ww):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        attn = softmax_one(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=wh, w=ww)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows
        x = self.window_partition(shifted_x, self.window_size)  # nw*b,c, window_size, window_size,
        # window shape
        wb, wc, wh, ww = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv_window, qkv_ch = torch.split(qkv, c * 3 // 2, dim=1)

        x_win = self.window_attn(qkv_window, h, w, wb, wh, ww)
        x_channel = self.channel_attn(qkv_ch, wh, ww)

        shifted_x = self.project_out(torch.cat([x_win, x_channel], dim=1))
        # shifted_x = self.project_out(x_win)
        shifted_x = self.window_reverse(shifted_x, self.window_size, h, w)  # b c h w

        # reverse cyclic shift
        if self.shift_size > 0:
            out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            out = shifted_x

        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, shift_size, window_size):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, shift_size, window_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.in_c = in_c
        self.proj2 = nn.Conv2d(24, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.proj = nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.proj(x)
        x = self.proj2(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
@ARCH_REGISTRY.register()
class Swintormer(nn.Module):
    def __init__(self,
                 inp_channels=6,
                 dim=48,
                 num_blocks=[2, 4, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 window_size=16,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super().__init__()
        self.window_size = window_size
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, window_size=window_size, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, window_size=window_size,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, window_size=window_size,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, window_size=window_size,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, window_size=window_size,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, window_size=window_size,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, window_size=window_size,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, window_size=window_size,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################
        self.apply(self._init_weights)
        # self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Sequential(*[nn.Conv2d(int(dim * 2 ** 1), 12, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.PixelShuffle(2)])

        self.refinement = checkpoint_wrapper(self.refinement)
        self.decoder_level1 = checkpoint_wrapper(self.decoder_level1)
        self.decoder_level2 = checkpoint_wrapper(self.decoder_level2)
        self.decoder_level3 = checkpoint_wrapper(self.decoder_level3)
        self.latent = checkpoint_wrapper(self.latent)
        self.encoder_level3 = checkpoint_wrapper(self.encoder_level3)
        self.encoder_level2 = checkpoint_wrapper(self.encoder_level2)
        # self.encoder_level1 = checkpoint_wrapper(self.encoder_level1)
        # self.patch_embed = checkpoint_wrapper(self.patch_embed)
        # self.down1_2 = checkpoint_wrapper(self.down1_2)
        # self.down2_3 = checkpoint_wrapper(self.down2_3)
        # self.down3_4 = checkpoint_wrapper(self.down3_4)
        # self.up4_3 = checkpoint_wrapper(self.up4_3)
        # self.up3_2 = checkpoint_wrapper(self.up3_2)
        # self.up2_1 = checkpoint_wrapper(self.up2_1)

    def forward(self, inp_img, C=None):
        # inp_img = inp_img[:,:3,:,:]
        _, _, init_h, init_w = inp_img.shape
        inp_img = 2. * inp_img - 1.
        # C = 2. * C - 1.
        # inp_img = self.check_image_size(inp_img)
        # C = self.check_image_size(C)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img[:, :3, :, :]

        out_dec_level1 = (out_dec_level1 + 1.) / 2.
        return out_dec_level1[:, :, :init_h, :init_w]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_size = self.window_size * 16
        mod_pad_h = (pad_size - h % pad_size) % pad_size
        mod_pad_w = (pad_size - w % pad_size) % pad_size
        # print("padding size", h, w, self.pad_size, mod_pad_h, mod_pad_w)
        try:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        except BaseException:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
        return x


if __name__ == '__main__':
    model = Swintormer()
    model.cpu()
    from torchsummary import summary

    with torch.no_grad():
        print(1)
        # summary(model, [(6, 512, 512), (3, 128, 128)])
        # summary(model, [(6, 768, 768), (3, 128, 128)])
        summary(model, [(6, 256, 256), (3, 128, 128)] ,device='cpu')
    from thop import profile
    with torch.no_grad():
        flops, params = profile(model, [torch.rand((1, 6, 256, 256,)).cpu()])
        print(f"FLOPs: {flops}")
# 445823950848
# 445799989248
# 4368501770400
# 984207958016
# 28219645952
# 4059078307200
# 6289966507680
# 1573643970720
# 54883499307572