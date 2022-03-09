import torch
from torch import nn
from torch.nn import functional as F
from vdvae.model.vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools


class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


class Encoder(HModule):
    def build(self):
        H = self.H
        self.in_conv = get_3x3(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        """ Gets deterministic Bottom Up activations
        """
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        # q(z_i | z <i, x) from activations + input (hence width*2 input dimension, zdim*2 is for mean and variance)
        self.enc = Block(width * 2, cond_width, H.zdim * 2, residual=False, use_3x3=use_3x3)
        # p(z_i | z <i)
        self.prior = Block(width, cond_width, H.zdim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        # 1x1 convolution on z after sampling
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        # resnet applied after everything
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts, get_mean_var=False, decode_from_p=False, use_mode=False):
        """ Computes block output using bottom up activations and q(z).
            Will sample latents from q(z). p(z) is only used for KL-div"""
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        # sample z from q(z) or p(z) or use means
        if use_mode:
            if decode_from_p:
                z = pm
            else:
                z = qm
        else:
            if decode_from_p:
                z = draw_gaussian_diag_samples(pm, pv)
            else:
                z = draw_gaussian_diag_samples(qm, qv)

        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        if get_mean_var:
            return z, x, kl, qm, qv, pm, pv
        else:
            return z, x, kl

    def sample_uncond(self, x, t=None, lvs=None):
        """ Computes block output without bottom up activations and skips q(z).
            Will sample z with temperature t or use passed latent. """
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            # use given latents as z
            z = lvs
        elif t == 0:
            z = pm.clone()
        else:
            if t is not None:
                # adjust variance to given temperature
                pv = pv + torch.ones_like(pv) * np.log(t)
            # sample z from p(z)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, get_latents=False,
                get_mean_var=False, decode_from_p=False, use_mode=False):

        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            # this is some upscaling stuff
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)

        res = self.sample(x, acts, get_mean_var=get_mean_var, decode_from_p=decode_from_p, use_mode=use_mode)

        if get_mean_var:
            z, x, kl, qm, qv, pm, pv = res
        else:
            z, x, kl = res
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        stats = {"kl": kl}

        if get_latents:
            stats["z"] = z.detach()

        if get_mean_var:
            stats["qm"] = qm.detach()
            stats["qv"] = qv.detach()
            stats["pm"] = pm.detach()
            stats["pv"] = pv.detach()

        return xs, stats

    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs


class Decoder(HModule):

    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= H.no_bias_above])
        self.out_net = DmolNet(H)
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False, get_mean_var=False, decode_from_p=False, use_mode=False):
        """ Will use activations and sample using q(z), p(z) will be calculated for KL."""
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}

        if isinstance(decode_from_p, bool):
            decode_from_p = [decode_from_p] * len(self.dec_blocks)

        if isinstance(use_mode, bool):
            use_mode = [use_mode] * len(self.dec_blocks)

        for i, block in enumerate(self.dec_blocks):
            xs, block_stats = block(xs, activations, get_latents=get_latents,
                                    get_mean_var=get_mean_var,
                                    decode_from_p=decode_from_p[i],
                                    use_mode=use_mode[i])
            stats.append(block_stats)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward_uncond(self, n, t=None, y=None):
        """ Will use either given latents of sample using p(z). """
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_uncond(xs, temp)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, (block, lvs) in enumerate(itertools.zip_longest(self.dec_blocks, latents)): # yields None for latents that are not given
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            # will use lvs
            xs = block.forward_uncond(xs, temp, lvs=lvs)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

def get_elbo(x, distortion_per_pixel, stats):
    rate_per_pixel = torch.zeros_like(distortion_per_pixel)
    ndims = np.prod(x.shape[1:])
    for statdict in stats:
        rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
    rate_per_pixel /= ndims
    elbo = (distortion_per_pixel + rate_per_pixel)
    return elbo, rate_per_pixel

class VAE(HModule):
    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def forward(self, x, x_target):
        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        elbo, rate_per_pixel = get_elbo(x, distortion_per_pixel, stats)
        return dict(elbo=elbo.mean(),
                    distortion=distortion_per_pixel.mean(),
                    rate=rate_per_pixel.mean())

    def forward_get_loss_and_latents(self, x, x_target, activations=None, get_mean_var=False, decode_from_p=False, use_mode=False):
        if activations is None:
            activations = self.encoder.forward(x)

        px_z, stats = self.decoder.forward(activations, get_latents=True, get_mean_var=get_mean_var,
                                           decode_from_p=decode_from_p, use_mode=use_mode)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        elbo, rate_per_pixel = get_elbo(x, distortion_per_pixel, stats)
        return dict(elbo=elbo,
                    distortion=distortion_per_pixel,
                    rate=rate_per_pixel
                    ), stats, activations

    def forward_get_latents(self, x, get_mean_var=False):
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True, get_mean_var=get_mean_var)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        # I guess this samples images based on passed latents
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)
