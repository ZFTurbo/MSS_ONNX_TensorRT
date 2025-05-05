import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from demucs.spec import spectro, ispectro
from demucs.hdemucs import pad1d
from einops import rearrange, pack, unpack, reduce, repeat
from einops.layers.torch import Rearrange
from beartype.typing import Tuple, Optional, List, Callable
from librosa import filters

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class STFT:
    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = config.dim_f

    def stft(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape([*batch_dims, c * 2, -1, x.shape[-1]])
        return x[..., :self.dim_f, :]

    def istft(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims, 2, -1])
        return x

    

class HTDemucs_processor:
    def __init__(self, config):
        self.use_train_segment = False
        self.hop_length = config.audio.hop_length
        self.wiener_iters=0
        self.wiener_residual = False
        self.cac = True
        self.S = 2
        self.z = None
        self.nfft = 4096
        self.length = None
        self.length_pre_pad = None
        self.B = None
        self.C = None
        self.Fq = None
        self.T = None
        self.mean = None
        self.std = None
        self.meant = None
        self.stdt = None
        self.num_subbands = 1
        self.sources = config.training.instruments
        
    def cac2cws(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, k, f // k, t)
        x = x.reshape(b, c * k, f // k, t)
        return x

    def cws2cac(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c // k, k, f, t)
        x = x.reshape(b, c // k, f * k, t)
        return x
    
    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x 
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2: 2 + le]
        return z
    
    def _magnitude(self, z):
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m
    
    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4 ** scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad: pad + length]
        return x
    
    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual
        

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame],
                    mix_stft[sample, frame],
                    niters,
                    residual=residual,
                )
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)
    
    def _mask(self, z, m):
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)
    
    def stft(self, mix):
        self.meant = mix.mean(dim=(1, 2), keepdim=True)
        self.stdt = mix.std(dim=(1, 2), keepdim=True)
        self.length = mix.shape[-1]
        self.length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                # print('Training length: {} Segment: {} Sample rate: {}'.format(training_length, self.segment, self.samplerate))
                if mix.shape[-1] < training_length:
                    self.length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - self.length_pre_pad))
                # print("Mix: {}".format(mix.shape))
        # print("Length: {}".format(length))
        self.z = self._spec(mix)
        # print("Z: {} Type: {}".format(z.shape, z.dtype))
        
        mag = self._magnitude(self.z)
        x = mag
        if self.num_subbands > 1:
            x = self.cac2cws(x)

        self.B, self.C, self.Fq, self.T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        self.mean = x.mean(dim=(1, 2, 3), keepdim=True)
        self.std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - self.mean) / (1e-5 + self.std)
        # x will be the freq. branch input.
        return x
    
    def istft(self, x, xt):
        self.S = len(self.sources)

        if self.num_subbands > 1:
            x = x.view(self.B, -1, self.Fq, self.T)
            x = self.cws2cac(x)
        
        x = x.view(self.B, self.S, -1, self.Fq * self.num_subbands, self.T)
        
        x = x * self.std[:, None] + self.mean[:, None]
        zout = self._mask(self.z, x) 
        
        x = self._ispec(zout, self.length)

        xt = xt.view(self.B, self.S, -1, self.length)
            
        xt = xt * self.stdt[:, None] + self.meant[:, None]
        x = xt + x
        
        if self.length_pre_pad:
            x = x[..., :self.length_pre_pad]
        
        return x
    

DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)
    
class BS_roformer_processor:
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
    ):
        self.x_is_mps = False
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )
        self.num_stems = num_stems
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )
        self.stft_repr = None
        
    
    def stft(self, raw_audio, target=None, return_loss_breakdown=False):
        self.raw_audio = raw_audio
        device = self.raw_audio.device
        self.x_is_mps = True if device.type == "mps" else False
        
        if self.raw_audio.ndim == 2:
            self.raw_audio = rearrange(self.raw_audio, 'b t -> b 1 t')
        channels = self.raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        self.raw_audio, batch_audio_channel_packed_shape = pack_one(self.raw_audio, '* t')

        self.stft_window = self.stft_window_fn(device=device)
        
        try:
            stft_repr = torch.stft(self.raw_audio, **self.stft_kwargs, window=self.stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(self.raw_audio.cpu() if self.x_is_mps else self.raw_audio, **self.stft_kwargs, window=self.stft_window.cpu() if self.x_is_mps else self.stft_window, return_complex=True).to(device)

        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        self.stft_repr = rearrange(stft_repr,
                              'b s f t c -> b (f s) t c')  # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting

        x = rearrange(self.stft_repr, 'b f t c -> b t (f c)')
        return x
    
    def istft(self, mask):
        stft_repr = rearrange(self.stft_repr, 'b f t c -> b 1 f t c')

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        # same as torch.stft() fix for MacOS MPS above
        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=self.stft_window, return_complex=False, length=self.raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if self.x_is_mps else stft_repr, **self.stft_kwargs, window=self.stft_window.cpu() if self.x_is_mps else self.stft_window, return_complex=False, length=self.raw_audio.shape[-1]).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=self.num_stems)

        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio
    
    
class Mel_band_roformer_processor:
    def __init__(
        self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            num_bands=60,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            dim_freqs_in=1025,
            sample_rate=44100,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            match_input_audio_length=False,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
    ):
        self.stft_repr = None
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        
        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
        
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )
        
        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]
        
        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        mel_filter_bank[0][0] = 1.

        mel_filter_bank[-1, -1] = 1.

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'
        
        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')
        self.freq_indices = freq_indices
        self.freqs_per_band  = freqs_per_band

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')
        self.num_freqs_per_band = num_freqs_per_band
        self.num_bands_per_freq  = num_bands_per_freq
        
        self.match_input_audio_length = match_input_audio_length
        self.batch = None
        self.channels = None
    
    
    def stft(self, raw_audio, target=None, return_loss_breakdown=False):
        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        self.batch, self.channels, raw_audio_length = raw_audio.shape

        self.istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and self.channels == 1) or (
                    self.stereo and self.channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
        self.stft_window = self.stft_window_fn(device=device)
        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=self.stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        self.stft_repr = rearrange(stft_repr,
                              'b s f t c -> b (f s) t c')

        batch_arange = torch.arange(self.batch, device=device)[..., None]

        x = self.stft_repr[batch_arange, self.freq_indices]
        
        return x
    
    def istft(self, masks):
        stft_repr = rearrange(self.stft_repr, 'b f t c -> b 1 f t c')

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=self.batch, n=self.num_stems, t=stft_repr.shape[-1])

        stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=self.num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=self.channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_repr = stft_repr * masks_averaged

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=self.stft_window, return_complex=False, length=self.istft_length)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=self.batch, s=self.audio_channels, n=self.num_stems)

        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')
        
        return recon_audio

class SCNet_processor:
    def __init__(self, config):
        self.sources = config.model.sources
        self.audio_channels = config.model.audio_channels
        self.dims = config.model.dims
        self.hop_length = config.model.hop_size
        self.conv_config = {
            'compress': config.model.compress,
            'kernel': config.model.conv_kernel,
        }

        self.stft_config = {
            'n_fft': config.model.nfft,
            'hop_length': self.hop_length,
            'win_length': config.model.win_size,
            'center': True,
            'normalized': config.model.normalized
        }
        self.B = None
        self.padding = None
        self.L = None
        self.C = None
        self.Fr = None
        self.T = None

    
    def stft(self, x):
        self.B = x.shape[0]
        self.padding = self.hop_length - x.shape[-1] % self.hop_length
        if (x.shape[-1] + self.padding) // self.hop_length % 2 == 0:
            self.padding += self.hop_length
        x = F.pad(x, (0, self.padding))
        self.L = x.shape[-1]
        x = x.reshape(-1, self.L)
        x = torch.stft(x, **self.stft_config, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        self.B, self.C, self.Fr, self.T = x.shape
        return x
    
    def istft(self, x):
        n = self.dims[0]
        x = x.view(self.B, n, -1, self.Fr, self.T)
        x = x.reshape(-1, 2, self.Fr, self.T).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        x = torch.istft(x, **self.stft_config)
        x = x.reshape(self.B, len(self.sources), self.audio_channels, -1)
        x = x[:, :, :, :-self.padding]
        return x
    

        