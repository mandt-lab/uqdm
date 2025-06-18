import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.distributions import constraints, TransformedDistribution, SigmoidTransform, AffineTransform
from torch.distributions import Normal, Uniform
from torch.distributions.kl import kl_divergence

# For compression to bits only
from tensorflow_compression.python.ops import gen_ops
import tensorflow as tf

from itertools import islice
from ml_collections import ConfigDict
import numpy as np
import json
import os
from pathlib import Path
from contextlib import contextmanager
import zipfile
from tqdm import tqdm

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

DATASET_PATH = {
    'ImageNet64': 'data/imagenet64/',
}

"""
PyTorch Implementation of 'Progressive Compression with Universally Quantized Diffusion Models', Yang et al., 2025.
Written with focus on readability for a single GPU.

Sections:
Model:   Denoising network
Data:    ImageNet64 data
UQDM:    Diffusion model + codec + simple trainer for the network + saving / loading

Major changes from previous work are highlighted in the class UQDM
"""

"""
Denoising network, Exponential Moving Average (EMA)
"""


class VDM_Net(torch.nn.Module):
    """
    Based on score Net from
    https://github.com/addtt/variational-diffusion-models/blob/main/vdm_unet.py
    which itself is based on
    https://github.com/google-research/vdm/blob/main/model_vdm.py
    and maps parameters via

    vdm_unet         ->        model_vdm
    mcfg.n_attention_heads:    1 (fixed)
    mcfg.embedding_dim:        sm_n_embd
    mcfg.n_blocks:             sm_n_layer
    mcfg.dropout_prob:         sm_pdrop
    mcfg.norm_groups:          32 (fixed, default setting for flax.linen.GroupNorm)

    In addition to predicting the noise, we (optionally) predict backward variances by doubling the output channels.
    """

    @staticmethod
    def softplus_inverse(x):
        """Helper which computes the inverse of `tf.nn.softplus`."""
        import math
        import numpy as np
        return math.log(np.expm1(x))

    def softplus_init1(self, x):
        # Softplus with a shift to bias the output towards 1.0.
        return torch.nn.functional.softplus(x + self.SOFTPLUS_INV1)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mcfg = mcfg = config.model

        attention_params = dict(
            n_heads=mcfg.n_attention_heads,
            n_channels=mcfg.embedding_dim,
            norm_groups=mcfg.norm_groups,
        )
        resnet_params = dict(
            ch_in=mcfg.embedding_dim,
            ch_out=mcfg.embedding_dim,
            condition_dim=4 * mcfg.embedding_dim,
            dropout_prob=mcfg.dropout_prob,
            norm_groups=mcfg.norm_groups,
        )
        if mcfg.use_fourier_features:
            self.fourier_features = FourierFeatures()
        self.embed_conditioning = nn.Sequential(
            nn.Linear(mcfg.embedding_dim, mcfg.embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(mcfg.embedding_dim * 4, mcfg.embedding_dim * 4),
            nn.SiLU(),
        )
        total_input_ch = mcfg.n_channels
        if mcfg.use_fourier_features:
            total_input_ch *= 1 + self.fourier_features.num_features
        self.conv_in = nn.Conv2d(total_input_ch, mcfg.embedding_dim, 3, padding=1)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.down_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params)
                if mcfg.attention_everywhere
                else None,
            )
            for _ in range(mcfg.n_blocks)
        )

        self.mid_resnet_block_1 = ResnetBlock(**resnet_params)
        self.mid_attn_block = AttentionBlock(**attention_params)
        self.mid_resnet_block_2 = ResnetBlock(**resnet_params)

        # Up path: n_blocks+1 blocks with a resnet block and maybe attention.
        resnet_params["ch_in"] *= 2  # double input channels due to skip connections
        self.up_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params)
                if mcfg.attention_everywhere
                else None,
            )
            for _ in range(mcfg.n_blocks + 1)
        )

        output_channels = mcfg.n_channels
        if config.model.get('learned_prior_scale'):
            output_channels *= 2

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=mcfg.norm_groups, num_channels=mcfg.embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(mcfg.embedding_dim, output_channels, kernel_size=3, padding=1)),
        )

        self.SOFTPLUS_INV1 = self.softplus_inverse(1.0)

    def forward(self, z, g_t):
        # Get gamma to shape (B, ).
        g_t = g_t.expand(z.shape[0])  # assume shape () or (1,) or (B,)
        assert g_t.shape == (z.shape[0],)
        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        t = (g_t - self.mcfg.gamma_min) / (self.mcfg.gamma_max - self.mcfg.gamma_min)
        t_embedding = get_timestep_embedding(t, self.mcfg.embedding_dim)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding)

        h = self.maybe_concat_fourier(z)
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        hs = []
        for down_block in self.down_blocks:  # n_blocks times
            hs.append(h)
            h = down_block(h, cond)
        hs.append(h)
        h = self.mid_resnet_block_1(h, cond)
        h = self.mid_attn_block(h)
        h = self.mid_resnet_block_2(h, cond)
        for up_block in self.up_blocks:  # n_blocks+1 times
            h = torch.cat([h, hs.pop()], dim=1)
            h = up_block(h, cond)
        h = self.conv_out(h)

        if self.mcfg.get('learned_prior_scale'):
            # Split the output into a mean and scale component. (B, C, H, W)
            eps_hat, pred_scale_factors = torch.split(h, self.mcfg.n_channels, dim=1)
            pred_scale_factors = self.softplus_init1(pred_scale_factors)  # Make positive.
        else:
            eps_hat = h

        assert eps_hat.shape == z.shape, (eps_hat.shape, z.shape)
        eps_hat = eps_hat + z

        if self.mcfg.get('learned_prior_scale'):
            return eps_hat, pred_scale_factors
        else:
            return eps_hat

    def maybe_concat_fourier(self, z):
        if self.mcfg.use_fourier_features:
            return torch.cat([z, self.fourier_features(z)], dim=1)
        return z


@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    # Sets to zero all the parameters of a module, and returns the module.
    for p in module.parameters():
        nn.init.zeros_(p.data)
    return module


class ResnetBlock(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out=None,
            condition_dim=None,
            dropout_prob=0.0,
            norm_groups=32,
    ):
        super().__init__()
        ch_out = ch_in if ch_out is None else ch_out
        self.ch_out = ch_out
        self.condition_dim = condition_dim
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
        )
        if condition_dim is not None:
            self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)),
        )
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, condition):
        h = self.net1(x)
        if condition is not None:
            assert condition.shape == (x.shape[0], self.condition_dim)
            condition = self.cond_proj(condition)
            condition = condition[:, :, None, None]
            h = h + condition
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        return x + h


def get_timestep_embedding(
        timesteps,
        embedding_dim: int,
        dtype=torch.float32,
        max_timescale=10_000,
        min_timescale=1,
):
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)


class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0 ** freqs_exponent * 2 * torch.pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)


def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # Rescale q and k. This makes them contiguous in memory.
    scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.view(*new_shape)
    k = k.view(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = torch.einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = torch.softmax(weight.float(), dim=-1).to(weight.dtype)  # (B*H, T, T)
    out = torch.einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)


class Attention(nn.Module):
    # Based on https://github.com/openai/guided-diffusion.

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        spatial_dims = qkv.shape[2:]
        qkv = qkv.view(*qkv.shape[:2], -1)  # (B, 3*H*C, T)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, H*C, T)
        return out.view(*out.shape[:2], *spatial_dims)


class AttentionBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),  # (B, 3 * C, H, W)
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x


class UpDownBlock(nn.Module):
    def __init__(self, resnet_block, attention_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
        return x


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Code from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    which is modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py
    and partially based on https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


"""
Data and Checkpoint Loading
"""


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class ToIntTensor:
    # for IMAGENET64
    def __call__(self, image):
        image = torch.as_tensor(image.reshape(3, 64, 64), dtype=torch.uint8)
        return image


class NPZLoader(Dataset):
    """
    Load from a batched numpy dataset.
    Keeps one data batch loaded in memory, so load idx sequentially for fast sampling
    """

    def __init__(self, path, train=True, transform=None, remove_duplicates=True):
        self.path = path
        if train:
            self.files = list(Path(path).glob('*train*.npz'))
        else:
            self.files = list(Path(path).glob('*val*.npz'))
        self.batch_lens = [self.npz_len(f) for f in self.files]
        self.anchors = np.cumsum([0] + self.batch_lens)
        self.removed_idxs = [[] for _ in range(len(self.files))]
        if not train and remove_duplicates:
            removed = np.load(os.path.join(path, 'removed.npy'))
            self.removed_idxs = [
                removed[(removed >= self.anchors[i]) & (removed < self.anchors[i + 1])] - self.anchors[i] for i in
                range(len(self.files))]
            self.anchors -= np.cumsum([0] + [np.size(r) for r in self.removed_idxs])
        self.transform = transform
        self.cache_fid = None
        self.cache_npy = None

    # https://stackoverflow.com/questions/68224572/how-to-determine-the-shape-size-of-npz-file
    @staticmethod
    def npz_len(npz):
        """
        Takes a path to an .npz file, which is a Zip archive of .npy files and returns the batch size of stored data,
        i.e. of the first .npy found
        """
        with zipfile.ZipFile(npz) as archive:
            for name in archive.namelist():
                if not name.endswith('.npy'):
                    continue
                npy = archive.open(name)
                version = np.lib.format.read_magic(npy)
                shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                return shape[0]

    def load_npy(self, fid):
        if not fid == self.cache_fid:
            self.cache_fid = fid
            self.cache_npy = np.load(str(self.files[fid]))['data']
            self.cache_npy = np.delete(self.cache_npy, self.removed_idxs[fid], axis=0)
        return self.cache_npy

    def __len__(self):
        # return sum(self.batch_lens)
        return self.anchors[-1]

    def __getitem__(self, idx):
        fid = np.argmax(idx < self.anchors) - 1
        idx = idx - self.anchors[fid]
        numpy_array = self.load_npy(fid)[idx]
        if self.transform is not None:
            torch_array = self.transform(numpy_array)
        return torch_array


def load_data(dataspec, cfg):
    """
    Load datasets, with finite eval set and infinitely looping training set
    """
    if not dataspec in DATASET_PATH.keys():
        raise ValueError('Unknown dataset. Add dataspec to load_data() or use one of \n%s' % list(DATASET_PATH.keys()))

    if dataspec in ['ImageNet64']:
        train_data, eval_data = [NPZLoader(DATASET_PATH[dataspec], train=mode, transform=ToIntTensor()) for mode in
                                 [True, False]]
    # elif:   # Add more datasets here

    train_iter, eval_iter = [DataLoader(d, batch_size=cfg.batch_size, shuffle=cfg.get('shuffle', False),
                                        pin_memory=cfg.get('pin_memory', True), num_workers=cfg.get('num_workers', 1))
                             for d in [train_data, eval_data]]
    train_iter = cycle(train_iter)

    return train_iter, eval_iter


def load_checkpoint(path):
    """
    Load model from checkpoint.

    Input:
    ------
    path: path to a folder containing hyperparameters as config.json and parameters as checkpoint.pt
    """
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config = ConfigDict(json.load(f))

    model = UQDM(config).to(device)
    cp_path = config.get('restore_ckpt', None)
    if cp_path is not None:
        model.load(os.path.join(path, cp_path))

    return model


"""
UQDM: Diffusion model, Distributions, Entropy Coding, UQDM
"""

@contextmanager
def local_seed(seed, i=0):
    # Allow for local randomness, use hashing to get unique local seeds for subsequent draws
    if seed is None:
        yield
    else:
        with torch.random.fork_rng():
            local_seed = hash((seed, i)) % (2 ** 32)
            torch.manual_seed(local_seed)
            yield


class LogisticDistribution(TransformedDistribution):
    """
    Creates a logistic distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the affine transform of a standard logistic distribution.
    Patterned after https://github.com/pytorch/pytorch/blob/main/torch/distributions/logistic_normal.py

    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution

    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        base_dist = Uniform(torch.tensor(0, dtype=loc.dtype, device=loc.device),
                            torch.tensor(1, dtype=loc.dtype, device=loc.device))
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        transforms = [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
        super().__init__(
            base_dist, transforms, validate_args=validate_args
        )

    @property
    def mean(self):
        return self.loc

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticDistribution, _instance)
        return super().expand(batch_shape, _instance=new)

    def cdf(self, x):
        # Should be numerically more stable than the default.
        return torch.sigmoid((x - self.loc) / self.scale)

    @staticmethod
    def log_sigmoid(x):
        # A numerically more stable implementation of torch.log(torch.sigmoid(x)).
        # c.f. https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.log_sigmoid.html#jax.nn.log_sigmoid
        return -torch.nn.functional.softplus(-x)

    def log_cdf(self, x):
        standardized = (x - self.loc) / self.scale
        return self.log_sigmoid(standardized)

    def log_survival_function(self, x):
        standardized = (x - self.loc) / self.scale
        return self.log_sigmoid(- standardized)


class NormalDistribution(torch.distributions.Normal):
    """
    Overrides the Normal distribution to add a numerically more stable log_cdf
    """

    def log_cdf(self, x):
        x = (x - self.loc) / self.scale
        # more stable, for float32 ported from JAX, using log(1-x) ~= -x, x >> 1
        # for small x
        x_l = torch.clip(x, max=-10)
        log_scale = -0.5 * x_l ** 2 - torch.log(-x_l) - 0.5 * np.log(2. * np.pi)
        # asymptotic series
        even_sum = torch.zeros_like(x)
        odd_sum = torch.zeros_like(x)
        x_2n = x_l ** 2
        for n in range(1, 3 + 1):
            y = np.prod(np.arange(2 * n - 1, 1, -2)) / x_2n
            if n % 2:
                odd_sum += y
            else:
                even_sum += y
            x_2n *= x_l ** 2
        x_lower = log_scale + torch.log(1 + even_sum - odd_sum)
        return torch.where(
            x > 5, -torch.special.ndtr(-x),
            torch.where(x > -10, torch.special.ndtr(torch.clip(x, min=-10)).log(), x_lower))

    def log_survival_function(self, x):
        raise NotImplementedError


class UniformNoisyDistribution(torch.distributions.Distribution):
    """
    Add uniform noise U[-delta/2, +delta/2] to a distribution.
    Adapted from https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/distributions/uniform_noise.py
    Also see https://pytorch.org/docs/stable/_modules/torch/distributions/distribution.html
    """

    arg_constraints = {}
    # arg_constraints = {"delta": torch.distributions.constraints.nonnegative}

    def __init__(self, base_dist, delta):
        super().__init__()
        self.base_dist = base_dist
        self.delta = delta  # delta is the noise width.
        self.half = delta / 2.
        self.log_delta = torch.log(delta)

    def sample(self, sample_shape=torch.Size([])):
        x = self.base_dist.sample(sample_shape)
        x += self.delta * torch.rand(x.shape, dtype=x.dtype, device=x.device) - self.half
        return x

    @property
    def mean(self):
        return self.base_dist.mean

    def discretize(self, u, tail_mass=2 ** -8):
        """
        Turn the continuous distribution into a discrete one by discretizing to the grid u + k * delta.
        Returns the pmf of k = round((x -  p_mean) / delta + u) as this is used for UQ, ignoring outlier values in the tails.
        """
        # For quantiles: Because p(x) = (G(x+d/2) - G(x-d/2))/d,
        # P(X <= x) = 1/d int_{x-d/2}^{x+d/2} G(u) du <= G(x+d/2) or >= G(x-d/2) which might be tighter for small d
        # P(X <= G^-1(a) - d/2) <= a, P(K <= (G^-1(a) - p_mean)/d - 1/2 - p_mean/d + u) <= a
        L = torch.floor((self.base_dist.icdf(tail_mass / 2) - self.base_dist.mean).min() / self.delta - 0.5)
        R = torch.ceil((self.base_dist.icdf(1 - tail_mass / 2) - self.base_dist.mean).max() / self.delta + 0.5)
        x = (torch.arange(L, R + 1, device=u.device).reshape(-1, *4*[1]) - u) * self.delta + self.base_dist.mean
        # Assume pdf is locally linear then ln(p(x+-d/2)) = ln(p(x)*d) = ln(p(x)) + ln(d)
        logits = self.log_prob(x) + torch.log(self.delta)
        return OverflowCategorical(logits=logits, L=L, R=R)

    def log_prob(self, y):
        # return torch.log(self.base_dist.cdf(y + self.half) - self.base_dist.cdf(y - self.half)) - self.log_delta
        if not hasattr(self.base_dist, "log_cdf"):
            raise NotImplementedError(
                "`log_prob()` is not implemented unless the base distribution implements `log_cdf()`.")
        try:
            return self._log_prob_with_logsf_and_logcdf(y)
        except NotImplementedError:
            return self._log_prob_with_logcdf(y)

    @staticmethod
    def _logsum_expbig_minus_expsmall(big, small):
        # Numerically stable evaluation of log(exp(big) - exp(small)).
        # https://github.com/tensorflow/compression/blob/a41fc70fc092bc6b72d5075deec34cbb47ef9077/tensorflow_compression/python/distributions/uniform_noise.py#L33
        return torch.where(
            torch.isinf(big), big, torch.log1p(-torch.exp(small - big)) + big
        )

    def _log_prob_with_logcdf(self, y):
        return self._logsum_expbig_minus_expsmall(
            self.base_dist.log_cdf(y + self.half), self.base_dist.log_cdf(y - self.half)) - self.log_delta

    def _log_prob_with_logsf_and_logcdf(self, y):
        """Compute log_prob(y) using log survival_function and cdf together."""
        # There are two options that would be equal if we had infinite precision:
        # Log[ sf(y - .5) - sf(y + .5) ]
        #   = Log[ exp{logsf(y - .5)} - exp{logsf(y + .5)} ]
        # Log[ cdf(y + .5) - cdf(y - .5) ]
        #   = Log[ exp{logcdf(y + .5)} - exp{logcdf(y - .5)} ]
        h = self.half
        base = self.base_dist
        logsf_y_plus = base.log_survival_function(y + h)
        logsf_y_minus = base.log_survival_function(y - h)
        logcdf_y_plus = base.log_cdf(y + h)
        logcdf_y_minus = base.log_cdf(y - h)

        # Important:  Here we use select in a way such that no input is inf, this
        # prevents the troublesome case where the output of select can be finite,
        # but the output of grad(select) will be NaN.

        # In either case, we are doing Log[ exp{big} - exp{small} ]
        # We want to use the sf items precisely when we are on the right side of the
        # median, which occurs when logsf_y < logcdf_y.
        condition = logsf_y_plus < logcdf_y_plus
        big = torch.where(condition, logsf_y_minus, logcdf_y_plus)
        small = torch.where(condition, logsf_y_plus, logcdf_y_minus)
        return self._logsum_expbig_minus_expsmall(big, small) - self.log_delta


class OverflowCategorical(torch.distributions.Categorical):
    """
    Discrete distribution over [L, L+1, ..., R-1, R] with LaPlace-based tail_masses for values <L and >R.
    """

    def __init__(self, logits, L, R):
        self.L = L
        self.R = R
        # stable version of log(1 - sum_i exp(logp_i))
        self.overflow = torch.log(torch.clip(- torch.expm1(torch.logsumexp(logits, dim=0)), min=0))
        super().__init__(logits=torch.movedim(torch.cat([logits, self.overflow[None]], dim=0), 0, -1))


class EntropyModel:
    """
    Entropy codec for discrete data based on Arithmetic Coding / Range Coding.
    Adapted from https://github.com/tensorflow/compression.
    For learned backward variances every symbol has a unique coding prior that requires a unique cdf table,
    which is computed in parallel here.
    """

    def __init__(self, prior, range_coder_precision=16):
        """

        Inputs:
        -------
        prior     - [Categorical or OverflowCategorical] prior model over integers (optionally with allocated tail mass
                    which will be encoded via Elias gamma code embedded into the range coder).
        range_coder_precision - precision passed to the range coding op, how accurately prior is quantized.
        """
        super().__init__()
        self.prior = prior
        self.prior_shape = self.prior.probs.shape[:-1]
        self.precision = range_coder_precision

        # Build quantization tables
        total = 2 ** self.precision
        probs = self.prior.probs.reshape(-1, self.prior.probs.shape[-1])
        quantized_pdf = torch.round(probs * total).to(torch.int32)
        quantized_pdf = torch.clip(quantized_pdf, min=1)

        # Normalize pdf so that sum pmf_i = 2 ** precision
        while True:
            mask = quantized_pdf.sum(dim=-1) > total
            if not mask.any():
                break
            # m * (log2(v) - log2(v-1))
            penalty = probs[mask] * (torch.log2(1 + 1 / (quantized_pdf[mask] - 1)))
            # inf if v = 1 as intended but handle nan if also pmf = 0
            idx = penalty.nan_to_num(torch.inf).argmin(dim=-1)
            quantized_pdf[mask, idx] -= 1
        while True:
            mask = quantized_pdf.sum(axis=-1) < total
            if not mask.any():
                break
            # m * (log2(v+1) - log2(v))
            penalty = probs[mask] * (torch.log2(1 + 1 / quantized_pdf[mask]))
            idx = penalty.argmax(dim=-1)
            quantized_pdf[mask, idx] += 1

        quantized_cdf = torch.cumsum(quantized_pdf, dim=-1)
        self.quantized_cdf = torch.cat([
            - self.precision * torch.ones((quantized_pdf.shape[0], 1), device=device),
            torch.zeros((quantized_pdf.shape[0], 1), device=device),
            quantized_cdf
        ], dim=-1).reshape(-1)
        self.indexes = torch.arange(quantized_pdf.shape[0], dtype=torch.int32)
        self.offsets = self.prior.L if type(self.prior) is OverflowCategorical else 0

    def compress(self, x):
        """
        Compresses a floating-point tensor to a bit string with the discretized prior.
        """
        x = (x - self.offsets).to(torch.int32).reshape(-1).cpu()
        codec = gen_ops.create_range_encoder([], self.quantized_cdf.cpu())
        codec = gen_ops.entropy_encode_index(codec, self.indexes.cpu(), x)
        bits = gen_ops.entropy_encode_finalize(codec).numpy()
        return bits

    def decompress(self, bits):
        """
        Decompresses a tensor from bit strings. This requires knowledge of the image shape,
        which for arbitrary images sizes needs to be sent as side-information.
        """
        bits = tf.convert_to_tensor(bits, dtype=tf.string)
        codec = gen_ops.create_range_decoder(bits, self.quantized_cdf.cpu())
        codec, x = gen_ops.entropy_decode_index(codec, self.indexes.cpu(), self.indexes.shape, tf.int32)
        # sanity = gen_ops.entropy_decode_finalize(codec)
        x = torch.from_numpy(x.numpy()).reshape(self.prior_shape).to(device).to(torch.float32) + self.offsets
        return x


class Diffusion(torch.nn.Module):
    """
    Progressive Compression with Gaussian Diffusion as in [Ho et al., 2020; Theis et al., 2022].
    """

    def __init__(self, config):
        """
        Hyperparamters are set via a config dict.

        config.model
            .n_timesteps           - number of diffusion steps, should be the same for training and inference, default:4
            .prior_type            - type of base distribution g_t, 'logistic' or 'normal'
            .base_prior_scale      - variance of g_t, 'forward_kernel' or 'default'
            .learned_prior_scale   - if to learn the variance of g_t, default: true
            .noise_schedule        - 'fixed_linear' or 'learned_linear'
            .fix_gamma_max         - set if using 'learned_linear' to only learn gamma_min
            .gamma_min             - initial start value at t=0
            .gamma_max             - initial end value at t=T
            .ema_rate              - default: 0.9999
            # network hyperparameters (c.f. VDM_Net.__init__)
            .attention_everywhere  -
            .use_fourier_features  -
            .n_attention_heads     -
            .n_channels            - default: 3
            .vocab_size            - default: 256
            .embedding_dim         -
            .n_blocks              -
            .norm_groups           -
            .dropout_prob          -
        config.data (c.f. torch DataLoader)
            .shuffle     - false is recommended for faster loading with naive data loading,
            .pin_memory  -
            .batch_size  -
            .num_workers -
            .data_spec   - "imagenet", add more in data_load
        config.training
            .n_steps                 - total steps on the training set, if continuing from a checkpoint
                                       this should be set to desired fine-tuning steps + all previous steps
            .log_metrics_every_steps - default: 1000
            .checkpoint_every_steps  - default: 10000
            .eval_every_steps        - default: 10000
            .eval_steps_to_run       - how many steps to evaluate on, set to None for the full eval set
        config.optim (c.f. torch Adam)
            .weight_decay   -
            .beta1          -
            .eps            -
            .lr             -
            .warmup         - linear learning rate warm-up, default: 1000
            .grad_clip_norm - maximal gradient norm per step , default: 1.0
        """
        super().__init__()
        self.config = config
        self.score_net = VDM_Net(config)
        self.gamma = self.get_noise_schedule(config)
        self.ema = ExponentialMovingAverage(self.score_net.parameters(), decay=config.model.ema_rate)

        # Init optimizer now to allow loading/saving optimizer state from checkpoints
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                                          eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        self.step = 0
        self.denoised = None
        self.compress_bits = []

    def sigma2(self, t):
        return torch.sigmoid(self.gamma(t))

    def sigma(self, t):
        return torch.sqrt(self.sigma2(t))

    def alpha(self, t):
        return torch.sqrt(torch.sigmoid(-self.gamma(t)))

    def q_t(self, x, t=1):
        # q(z_t | x) = N(alpha_t x, sigma^2_t).
        return Normal(loc=self.alpha(t) * x, scale=self.sigma(t))

    def p_1(self):
        # p(z_1) = N(0, 1)
        return Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

    def p_s_t(self, p_loc, p_scale, t, s):
        # p(z_s | z_t) = N(p_loc, p_scale^2)
        if self.config.model.prior_type == 'logistic':
            base_dist = LogisticDistribution(loc=p_loc, scale=p_scale * np.sqrt(3. / np.pi ** 2))
        elif self.config.model.prior_type in ('gaussian', 'normal'):
            base_dist = NormalDistribution(loc=p_loc, scale=p_scale)
        else:
            try:
                base_dist = getattr(torch.distributions, self.config.model.prior_type)
            except AttributeError:
                raise ValueError(f"Unknown prior type {self.config.model.prior_type}")
        return base_dist

    def q_s_t(self, q_loc, q_scale):
        # q(z_s | z_t, x) = N(q_loc, q_scale^2)
        return NormalDistribution(loc=q_loc, scale=q_scale)

    def relative_entropy_coding(self, q, p, compress_mode=None):
        # Exponential runtime with naive REC algorithms
        raise NotImplementedError

    def get_s_t_params(self, z_t, t, s, x=None, clip_denoised=True, cache_denoised=False, deterministic=False):
        """
        Compute the (location, scale) parameters of either q(z_s | z_t, x)
        or the reverse process distribution p(z_s | z_t) = q(z_s | z_t, x=x_hat) for the given z_t and times t, s.

        Inputs:
        -------
        x              - if not None compute the parameters of q(z_t | z, x) instead p(z_s | z_t)
        clip_denoised  - if True, will clip the denoised prediction x_hat(z_t) to [-1, 1];
                         this might be used to draw better samples.
        cache_denoised - keep the denoised prediction in memory for later use
        deterministic  - if True, compute the mean needed for flow-based sampling instead, removing less noise overall
        """
        gamma_t, gamma_s = self.gamma(t), self.gamma(s)
        alpha_t, alpha_s = self.alpha(t), self.alpha(s)
        sigma_t, sigma_s = self.sigma(t), self.sigma(s)
        # expm1 = 1 - alpha_t^2 / alpha_s^2 * sigma_s^2 / sigma_t^2 = sigma_t|s^2 / sigma_t^2
        expm1_term = - torch.special.expm1(gamma_s - gamma_t)

        # Parameters of q(z_s | z_t, x)
        # q_var = sigma_s^2 * sigma^2_t|s / sigma^2_t, c.f. VDM eq (25)
        #       = sigma_s^2 * expm1_term, c.f. VDM eq (33)
        # q_loc = alpha_t / alpha_s * sigma_s^2 / sigma_t^2 * z_t + alpha_s * sigma_t|s^2 / sigma_t^2 * x, c.f. VDM eq (26)
        #       = alpha_s * ((1 - expm1_term) / alpha_t * z_t + expm1_term * x)
        #       = alpha_s / alpha_t * (z_t - sigma_t|s^2 / sigma_t * eps), c.f. VDM eq (29)
        #       = alpha_s / alpha_t * (z_t - sigma_t * expm1_term * eps), c.f. VDM eq (32)

        #       = alpha_s / alpha_t * (z_t - sigma_t * eps) + c * eps,
        #         with c = alpha_s / alpha_t * sigma_t * (1 - expm1_term) = alpha_t / alpha_s * sigma_s^2 / sigma_t
        #       = alpha_s / alpha_t * x + c * eps,
        # for flow-based set var = 0 and c = sigma_s, c.f. DDIM eq (12)
        # -> loc = alpha_s / alpha_t * z_t + (sigma_s - alpha_s / alpha_t * sigma_t) * eps
        #        = alpha_s / alpha_t * z_t + (sigma_s - alpha_s / alpha_t * sigma_t) * (z_t - alpha_t * x) / sigma_t
        #        = alpha_s / alpha_t * z_t + (sigma_s / sigma_t - alpha_s / alpha_t) * (z_t - alpha_t * x)
        #        = (alpha_s / alpha_t + sigma_s / sigma_t - alpha_s / alpha_t) * z_t - alpha_t * (sigma_s / sigma_t - alpha_s / alpha_t) * x
        #        = sigma_s / sigma_t * z_t - (alpha_t * sigma_s / sigma_t - alpha_s) * x

        # Set x = x_hat or eps = eps_hat for p(z_s | z_t)
        if x is None:
            if self.config.model.get('learned_prior_scale'):
                eps_hat, pred_scale_factors = self.score_net(z_t, gamma_t)
            else:
                eps_hat = self.score_net(z_t, gamma_t)
            # Compute denoised prediction only if necessary
            if clip_denoised or cache_denoised:
                x = (z_t - sigma_t * eps_hat) / alpha_t  # c.f. VDM eq (30)
            if clip_denoised:
                x.clamp_(-1.0, 1.0)
            if cache_denoised:
                self.denoised = x

            # Variance of q(z_s | z_t, x)
            scale = sigma_s * torch.sqrt(expm1_term)
            # Additional modifications for p(z_s | z_t)
            if self.config.model.get('base_prior_scale', 'forward_kernel') == 'forward_kernel':
                # use sigma_t|s^2, the variance of q(z_t | z_s) instead
                scale = sigma_t * torch.sqrt(expm1_term)
            if self.config.model.get('learned_prior_scale'):
                scale = scale * pred_scale_factors
        else:
            scale = sigma_s * torch.sqrt(expm1_term)

        # Mean of q(z_s | z_t, x)
        if x is not None:
            if deterministic:
                loc = sigma_s / sigma_t * z_t - (alpha_t * sigma_s / sigma_t - alpha_s) * x
            else:
                loc = alpha_s * ((1 - expm1_term) / alpha_t * z_t + expm1_term * x)
        else:
            if deterministic:
                loc = alpha_s / alpha_t * z_t + (sigma_s - alpha_s / alpha_t * sigma_t) * eps_hat
            else:
                loc = alpha_s / alpha_t * (z_t - sigma_t * expm1_term * eps_hat)

        return loc, scale

    def transmit_q_s_t(self, x, z_t, t, s, compress_mode=None, cache_denoised=False):
        """
        Perform a single transmission step of drawing a sample of z_t given z_s from q(z_t | z_s, x),
        under the conditional prior p(z_t | z_s).
        This will be approximated by REC/channel simulation at test time for actual compression.

        Inputs:
        -------
        x             - the continuous data; belongs to the diffusion space (usually scaled to [-1, 1])
        z_t           - the previously communicated latent state
        t, s          - the previous and current time steps, in [0, 1]; s < t.
        compress_mode - if to compress to bits in inference mode (which is slower), one of [None, 'encode', 'decode']

        Returns:
        --------
        z_s  - the new latent state
        rate - (estimate of) the KL divergence between q(z_s | z_t, x) and p(z_s | z_t)
        """
        # Compute parameters of q(z_s | z_t, x) and the prior p(z_s | z_t)
        p_loc, p_scale = self.get_s_t_params(z_t, t, s, cache_denoised=cache_denoised)
        q_loc, q_scale = self.get_s_t_params(z_t, t, s, x=x)
        p_s_t = self.p_s_t(p_loc, p_scale, t, s)
        q_s_t = self.q_s_t(q_loc, q_scale)
        z_s, rate = self.relative_entropy_coding(q_s_t, p_s_t, compress_mode=compress_mode)
        return z_s, rate

    def transmit_image(self, z_0, x_raw, compress_mode=None):
        if compress_mode in ['encode', 'decode']:
            p = torch.distributions.Categorical(logits=self.log_probs_x_z0(z_0=z_0))
        if compress_mode == 'decode':
            # consume bits
            x_raw = self.entropy_decode(self.compress_bits.pop(0), p)
        elif compress_mode == 'encode':
            # accumulate bits
            self.compress_bits += [self.entropy_encode(x_raw, p)]
        return x_raw

    def forward(self, x_raw, z_1=None, recon_method=None, compress_mode=None, seed=None):
        """
        Run a given data batch through the encoding/decoding path and compute the loss and other metrics.

        Inputs:
        -------
        x             - batch of shape [B, C, H, W]
        z_1           - if provided, will use this as the topmost latent state instead of sampling from q(z_1 | x).
        recon_method  - (optional) one of ['ancestral', 'denoise', 'flow-based']; determines how a progressive
                        reconstruction will be computed based on an intermediate latent state.
        compress_mode - if to compress to bits in inference mode (which is slower), one of [None, 'encode', 'decode']
        seed          - allow for common randomness
        """
        rescale_to_bpd = 1. / (np.prod(x_raw.shape[1:]) * np.log(2.))

        # Transform from uint8 in [0, 255] to float in [-1, 1]; the first r.v. of the diffusion process.
        x = 2 * ((x_raw.float() + .5) / self.config.model.vocab_size) - 1

        # 1. PRIOR/LATENT LOSS
        # KL z1 with N(0,1) prior; should be close to 0.
        if z_1 is None and not torch.is_inference_mode_enabled():
            # During training me might want to optimize the noise schedule so use the full NELBO
            q_1 = self.q_t(x)
            p_1 = self.p_1()
            with local_seed(seed, i=0):
                z_1 = q_1.sample()
            loss_prior = kl_divergence(q_1, p_1).sum(dim=[1, 2, 3])
        else:
            # In actual compression, we can't do REC for the Gaussian q(z_1|x) under p(z_1), so
            # instead both encoder/decoder will draw from p(z_1).
            if z_1 is None:
                p_1 = self.p_1()
                with local_seed(seed, i=0):
                    z_1 = p_1.sample(x.shape)
            loss_prior = torch.zeros(x.shape[0], device=device)

        # 2. DIFFUSION LOSS
        # Sample through the hierarchy and sum together KL[q(z_s | z_t, x)||p(z_s | z_t)) for the diffusion loss.
        z_s = z_1
        rate_s = loss_prior
        loss_diff = 0.
        times = torch.linspace(1, 0, self.config.model.n_timesteps + 1, device=device)
        assert len(times) >= 2, "Need at least one diffusion step."
        metrics = []
        for i in range(len(times) - 1):
            z_t = z_s
            rate_t = rate_s
            t, s = times[i], times[i + 1]
            with local_seed(seed, i=i + 1):
                z_s, rate_s = self.transmit_q_s_t(x, z_t, t, s, compress_mode=compress_mode,
                                                  cache_denoised=recon_method == 'denoise')
            loss_diff += rate_s

            if recon_method is not None:
                x_hat_t = self.denoise_z_t(z_t, recon_method, times=times[i:])
                metrics += [{
                    'prog_bpds': rate_t.cpu() * rescale_to_bpd,
                    'prog_x_hats': x_hat_t.detach().cpu(),
                    'prog_mses': torch.mean((x_hat_t - x_raw).float() ** 2, dim=[1, 2, 3]).cpu(),
                }]

        z_0 = z_s
        if recon_method is not None:
            if recon_method == 'ancestral':
                x_hat_t = self.decode_p_x_z_0(z_0=z_0, method='sample')
            else:
                x_hat_t = self.decode_p_x_z_0(z_0=z_0, method='argmax')
            metrics += [{
                'prog_bpds': rate_s.cpu() * rescale_to_bpd,
                'prog_x_hats': x_hat_t.detach().cpu(),
                'prog_mses': torch.mean((x_hat_t - x_raw).float() ** 2, dim=[1, 2, 3]).cpu(),
            }]

        # 3. RECONSTRUCTION LOSS.
        # Using the same likelihood model as in VDM.
        log_probs = self.log_probs_x_z0(z_0=z_0, x_raw=x_raw)
        loss_recon = -log_probs.sum(dim=[1, 2, 3])
        x_raw = self.transmit_image(z_0, x_raw, compress_mode=compress_mode)
        if recon_method is not None:
            metrics += [{
                'prog_bpds': loss_recon.cpu() * rescale_to_bpd,
                'prog_x_hats': x_raw.cpu(),
                'prog_mses': torch.zeros(x.shape[:1]),
            }]
            metrics = default_collate(metrics)
        else:
            metrics = {}

        bpd_latent = torch.mean(loss_prior) * rescale_to_bpd
        bpd_recon = torch.mean(loss_recon) * rescale_to_bpd
        bpd_diff = torch.mean(loss_diff) * rescale_to_bpd
        loss = bpd_recon + bpd_latent + bpd_diff
        metrics.update({
            "bpd": loss,
            "bpd_latent": bpd_latent,
            "bpd_recon": bpd_recon,
            "bpd_diff": bpd_diff,
        })

        return loss, metrics

    @torch.no_grad()
    def sample(self, init_z=None, shape=None, times=None, deterministic=False,
               clip_samples=False, decode_method='argmax', return_hist=False):
        """
        Perform ancestral / flow-based sampling.

        Inputs:
        -------
        init_z        - latent state [B, C, H, W]
        shape         - if no init_z is given specify the shape of z instead
        times         - (optional) provide a custom (e.g. partial) sequence of steps
        deterministic - use flow-based sampling instead of ancestral sampling
        clip_samples  - clip latents to [-1, 1]
        decode_method - 'argmax' or 'sample'
        return_hist   - if set return full history of latent states
        """
        if init_z is None:
            assert shape is not None
            p_1 = self.p_1()
            z = p_1.sample(shape)
        else:
            z = init_z
        if return_hist:
            samples = [z]
        if times is None:
            times = torch.linspace(1.0, 0.0, self.config.model.n_timesteps + 1, device=device)

        # for i in trange(len(times) - 1, desc="sampling"):
        for i in range(len(times) - 1):
            t, s = times[i], times[i + 1]
            p_loc, p_scale = self.get_s_t_params(z, t, s, clip_denoised=clip_samples, deterministic=deterministic)
            if deterministic:
                z = p_loc
            else:
                z = self.p_s_t(p_loc, p_scale, t, s).sample()
            if return_hist:
                samples.append(z)
        x_raw = self.decode_p_x_z_0(z_0=z, method=decode_method)

        if return_hist:
            return x_raw, samples + [x_raw]
        else:
            return x_raw

    def entropy_encode(self, k, p):
        """
        Encode integer array k to bits using a prior / coding distribution p.
        We might want to quantize scale for determinism and added stability across multiple machines.
        """
        # When using a scalar prior it would be better to quantize u as in tfc.UniversalBatchedEntropyModel
        assert self.config.model.learned_prior_scale
        em = EntropyModel(p)
        bitstring = em.compress(k)
        return bitstring

    def entropy_decode(self, bits, p):
        """
        Decode integer array from bits using the prior p.
        """
        assert self.config.model.learned_prior_scale
        em = EntropyModel(p)
        k = em.decompress(bits)
        return k

    @torch.inference_mode()
    def compress(self, image):
        # return the bits for each step
        self.compress_bits = []
        # accumulate bits
        self.forward(image.to(device), compress_mode='encode', seed=0)
        return self.compress_bits

    @torch.inference_mode()
    def decompress(self, bits, image_shape, recon_method='denoise'):
        # consume the bits for each step, return the intermediate reconstructions for each step
        self.compress_bits = bits.copy()
        # consume the bits for each step
        _, metrics = self.forward(torch.zeros(image_shape, device=device), compress_mode='decode',
                                  recon_method=recon_method, seed=0)
        return metrics['prog_x_hats']

    def log_probs_x_z0(self, z_0, x_raw=None):
        """
        Computes log p(x_raw | z_0), under the Gaussian approximation of q(z_0|x) introduced in VDM, section 3.3.
        If `x_raw` is not provided, this method computes the log probs of every
        possible value of x_raw under a factorized categorical distribution; otherwise,
        it will evaluate the log probs of the given `x_raw`.

        Internally we compute p(x_i | z_0i), with i = pixel index, for all possible values
        of x_i in the vocabulary. We approximate this with q(z_0i | x_i).
        Un-normalized logits are: -1/2 SNR_0 (z_0 / alpha_0 - k)^2
        where k takes all possible x_i values. Logits are then normalized to logprobs.

        If `x_raw` is None, the method returns a tensor of shape (B, C, H, W,
        vocab_size) containing, for each pixel, the log probabilities for all
        `vocab_size` possible values of that pixel. The output sums to 1 over
        the last dimension. Otherwise, we will select the log probs of the given `x_raw`.

        Inputs:
        -------
        z_0   - z_0 to be decoded, shape (B, C, H, W).
        x_raw - Input uint8 image, shape (B, C, H, W).

        Returns:
        --------
        log_probs - Log probabilities [B, C, H, W, vocab_size] if `x_raw` is None else [B, C, H, W]
        """
        gamma_0 = self.gamma(torch.tensor([0.0], device=device))
        z_0_rescaled = z_0 / torch.sqrt(torch.sigmoid(-gamma_0))
        # Compute a tensor of log p(x | z) for all possible values of x.
        # Logits are exact if there are no dependencies between dimensions of x
        x_vals = torch.arange(self.config.model.vocab_size, device=z_0_rescaled.device)
        x_vals = 2 * ((x_vals + .5) / self.config.model.vocab_size) - 1
        x_vals = torch.reshape(x_vals, [1] * z_0_rescaled.ndim + [-1])
        z = z_0_rescaled.unsqueeze(-1)  # (B, D1, ..., D_n) -> (B, D1, ..., D_n, 1) for broadcasting
        logits = -0.5 * torch.exp(-gamma_0) * (z - x_vals) ** 2  # (B, D1, ..., D_n, V)
        logprobs = torch.log_softmax(logits, dim=-1)  # (B, C, H, W, V)

        if x_raw is None:
            # Has an extra dimension for vocab_size.
            return logprobs
        else:
            # elementwise log prob, same shape as x_raw
            x_one_hot = nn.functional.one_hot(x_raw.long(), num_classes=self.config.model.vocab_size)
            # Select the correct log probabilities.
            log_probs = (x_one_hot * logprobs).sum(-1)  # (B, C, H, W)
            return log_probs

    def decode_p_x_z_0(self, z_0, method='argmax'):
        """
        Decode the given latent state z_0 to the data space,
        using the observation model p(x | z_0).

        Inputs:
        -------
        z_0    - the latent state [B, C, H, W]
        method - 'argmax' or 'sample'

        Returns:
        --------
        x_raw - the decoded x, mapped to data (integer) space
        """
        logprobs = self.log_probs_x_z0(z_0=z_0)  # (B, C, H, W, vocab_size)
        if method == 'argmax':
            x_raw = torch.argmax(logprobs, dim=-1)  # (B, C, H, W)
        elif method == 'sample':
            x_raw = torch.distributions.Categorical(logits=logprobs).sample()
        else:
            raise ValueError(f"Unknown decoding method {method}")
        return x_raw

    def denoise_z_t(self, z_t, recon_method, times=None):
        """
        Make a progressive data reconstruction based on z_t and compute its reconstruction quality.

        Inputs:
        -------
        z_t          - noisy diffusion latent variable
        recon_method - one of 'denoise', 'ancestral', 'flow_based'
        times        - remaining time steps including current t, for ancestral / flow-based sampling
        """
        if recon_method == 'ancestral':
            x_hat_t = self.sample(
                times=times, init_z=z_t,
                clip_samples=True, decode_method='argmax', return_hist=False
            )
        elif recon_method == 'flow_based':
            x_hat_t = self.sample(
                times=times, init_z=z_t, deterministic=True,
                clip_samples=False, decode_method='argmax', return_hist=False
            )
        elif recon_method == 'denoise':
            # Load from cache
            assert self.denoised is not None
            # Map to data space
            x_hat_t = self.decode_p_x_z_0(z_0=self.denoised, method='argmax')
            self.denoised = None
        else:
            raise ValueError(f"Unknown progressive reconstruction method {recon_method}")

        return x_hat_t

    @staticmethod
    def get_noise_schedule(config):
        # gamma is the negative log-snr as in VDM eq (3)
        gamma_min, gamma_max, schedule = [getattr(config.model, k) for k in
                                          ['gamma_min', 'gamma_max', 'noise_schedule']]
        assert gamma_max > gamma_min, "SNR should be decreasing in time"
        if schedule == "fixed_linear":
            gamma = Diffusion.FixedLinearSchedule(gamma_min, gamma_max)
        elif schedule == "learned_linear":
            gamma = Diffusion.LearnedLinearSchedule(gamma_min, gamma_max, config.model.get('fix_gamma_max'))
        # elif:    # add different noise schedules here
        else:
            raise ValueError('Unknown noise schedule %s' % schedule)
        return gamma

    class FixedLinearSchedule(torch.nn.Module):
        def __init__(self, gamma_min, gamma_max):
            super().__init__()
            self.gamma_min = gamma_min
            self.gamma_max = gamma_max

        def forward(self, t):
            return self.gamma_min + (self.gamma_max - self.gamma_min) * t

    class LearnedLinearSchedule(torch.nn.Module):
        def __init__(self, gamma_min, gamma_max, fix_gamma_max=False):
            super().__init__()
            self.fix_gamma_max = fix_gamma_max
            if fix_gamma_max:
                self.gamma_max = torch.tensor(gamma_max)
            else:
                self.b = torch.nn.Parameter(torch.tensor(gamma_min))
            self.w = torch.nn.Parameter(torch.tensor(gamma_max - gamma_min))

        def forward(self, t):
            w = self.w.abs()
            if self.fix_gamma_max:
                return w * (t - 1.) + self.gamma_max
            else:
                return self.b + w * t

    def save(self):
        torch.save({
            'model': self.score_net.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step
        }, self.self.config.checkpoint_path)

    def load(self, path):
        cp = torch.load(path, map_location=device, weights_only=False)
        # score_net + gamma
        self.score_net.load_state_dict(cp['model'])
        self.ema.load_state_dict(cp['ema'])
        self.optimizer.load_state_dict(cp['optimizer'])
        self.step = cp['step']

    def trainer(self, train_iter, eval_iter=None):
        """
        Train UQDM for a specified number of steps on a train set.
        Hyperparameters are set via self.config.training, self.config.eval, and self.config.optim.
        """

        if self.step >= self.config.training.n_steps:
            print('Skipping training, increase training.n_steps if more steps are desired.')

        while self.step < self.config.training.n_steps:
            # Parameter update step
            batch = next(train_iter).to(device)
            self.optimizer.zero_grad()
            model.train()
            loss, metrics = self(batch)
            loss.backward()
            if self.config.optim.warmup > 0:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.config.optim.lr * np.minimum(self.step / self.config.optim.warmup, 1.0)
            if self.config.optim.grad_clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.optim.grad_clip_norm)
            self.optimizer.step()
            self.step += 1
            self.ema.update(model.parameters())

            last = self.step == self.config.training.n_steps
            # Save model checkpoint
            if self.step % self.config.training.log_metrics_every_steps == 0 or last:
                self.save()
            # Print train metrics
            if self.step % self.config.training.log_metrics_every_steps == 0 or last:
                print(metrics)
            # Compute and print validation metrics
            if eval_iter is not None and (self.step % self.config.training.eval_every_steps == 0 or last):
                n_batches = self.config.training.eval_steps_to_run
                res = []
                for batch in tqdm(islice(eval_iter, n_batches), total=n_batches or len(eval_iter),
                                  desc='Evaluating on test set'):
                    batch = batch.to(device)
                    with torch.inference_mode():
                        self.ema.store(model.parameters())
                        self.ema.copy_to(model.parameters())
                        model.eval()
                        _, ths_metrics = self(batch)
                        self.ema.restore(model.parameters())
                    res += [ths_metrics]
                res = default_collate(res)
                print({k: v.mean().item() for k, v in res.items()})

    @staticmethod
    def mse_to_psnr(mse, max_val):
        with np.errstate(divide='ignore'):
            return -10 * (np.log10(mse) - 2 * np.log10(max_val))

    @torch.inference_mode()
    def evaluate(self, eval_iter, n_batches=None, seed=None):
        """
        Evaluate rate-distortion on the test set.

        Inputs:
        -------
        n_batches - (optionally) give a number of batches to evaluate
        """

        res = []
        for X in tqdm(islice(eval_iter, n_batches), total=n_batches or len(eval_iter), desc='Evaluating UQDM'):
            X = X.to(device)
            ths_res = {}
            for recon_method in ('denoise', 'ancestral', 'flow_based'):
                # If evaluating bpds as file sizes:
                # self.compress_bits = []
                # loss, metrics = self(X, recon_method=recon_method, seed=seed, compress_mode='encode')
                # bpds = np.cumsum([len(b) * 8 for b in self.compress_bits]) / np.prod(X.shape)
                loss, metrics = self(X, recon_method=recon_method, seed=seed)
                bpds = np.cumsum(metrics['prog_bpds'].mean(dim=1))
                psnrs = self.mse_to_psnr(metrics['prog_mses'].mean(dim=1), max_val=255.)
                ths_res[recon_method] = dict(bpds=bpds, psnrs=psnrs)
            res += [ths_res]
        res = default_collate(res)

        for recon_method in res.keys():
            bpps = np.round(3 * res[recon_method]['bpds'].mean(axis=0).numpy(), 4)
            psnrs = np.round(res[recon_method]['psnrs'].mean(axis=0).numpy(), 4)
            print('Reconstructions via: %s\nbpps:  %s\npsnrs: %s\n' % (recon_method, bpps, psnrs))


class UQDM(Diffusion):
    """
    Making Progressive Compression tractable with Universal Quantization.
    """

    def __init__(self, config):
        """
        See Diffusion.__init__ for hyperparameters.
        """
        super().__init__(config)
        self.compress_bits = None

    def p_s_t(self, p_loc, p_scale, t, s):
        # p(z_s | z_t) is a convolution of g_t and U(+- d_t), d_t = sqrt(12) * sigma_s * sqrt(exmp1term)
        delta_t = self.sigma(s) * torch.sqrt(- 12 * torch.special.expm1(self.gamma(s) - self.gamma(t)))
        base_dist = super().p_s_t(p_loc, p_scale, t, s)
        return UniformNoisyDistribution(base_dist, delta_t)

    def q_s_t(self, q_loc, q_scale):
        # q(z_s | z_t, x) = U(q_loc +- sqrt(3) * q_scale)
        return Uniform(low=q_loc - np.sqrt(3) * q_scale, high=q_loc + np.sqrt(3) * q_scale)

    def relative_entropy_coding(self, q, p, compress_mode=None):
        # Transmit sample z_s ~ q(z_s | z_t, x)
        if not torch.is_inference_mode_enabled():
            z_s = q.sample()
        else:
            # Apply universal quantization
            # shared U(-0.5, 0.5), seeds have already been set in self.forward
            u = torch.rand(q.mean.shape, device=q.mean.device) - 0.5

            # very slow, ~ 25 symbols/s
            # cp = tfc.NoisyLogistic(loc=0.0, scale=(p.base_dist.scale / p.delta).cpu().numpy())
            # em2 = tfc.UniversalBatchedEntropyModel(cp, coding_rank=4, compression=True, num_noise_levels=30)
            # k = (q.mean - p.mean) / p.delta
            # bitstring = em2.compress(k.cpu())
            # k_hat = em2.decompress(bitstring, [])

            if compress_mode in ['encode', 'decode']:
                p_discrete = p.discretize(u)
            if compress_mode == 'decode':
                # consume bits
                quantized = self.entropy_decode(self.compress_bits.pop(0), p_discrete)
            else:
                # Add dither U(-delta/2, delta/2)
                # Transmit residual q - p for greater numerical stability
                quantized = torch.round((q.mean - p.mean + p.delta * u) / p.delta)
                if compress_mode == 'encode':
                    # accumulate bits
                    self.compress_bits += [self.entropy_encode(quantized, p_discrete)]
            # Subtract the same (pseudo-random) dither using shared randomness
            z_s = quantized * p.delta + p.mean - p.delta * u

        # Evaluate z_s under log (posterior/prior) to get MC estimate of KL.
        rate = - p.log_prob(z_s) - torch.log(p.delta)
        rate = torch.sum(rate, dim=[1, 2, 3])
        return z_s, rate


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # model = load_checkpoint('checkpoints/uqdm-tiny')
    # model = load_checkpoint('checkpoints/uqdm-small')
    model = load_checkpoint('checkpoints/uqdm-medium')
    # model = load_checkpoint('checkpoints/uqdm-big')
    train_iter, eval_iter = load_data('ImageNet64', model.config.data)

    # model.trainer(train_iter, eval_iter)
    model.evaluate(eval_iter, n_batches=10, seed=seed)

    # Compress one image
    image = next(iter(eval_iter))
    compressed = model.compress(image)
    bits = [len(b) * 8 for b in compressed]
    reconstructions = model.decompress(compressed, image.shape, recon_method='denoise')
    assert (reconstructions[-1] == image).all()
    print('Reconstructions via: denoise, compression to bits\nbpps:  %s'
          % np.round(np.cumsum(bits) / np.prod(image.shape) * 3, 4))
