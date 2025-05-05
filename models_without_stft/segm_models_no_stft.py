import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from utils import prefer_target_instrument


def get_act(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type[:3] == 'elu':
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha)
    else:
        raise Exception


def get_decoder(config, c):
    decoder = None
    decoder_options = dict()
    if config.model.decoder_type == 'unet':
        try:
            decoder_options = dict(config.decoder_unet)
        except:
            pass
        decoder = smp.Unet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'fpn':
        try:
            decoder_options = dict(config.decoder_fpn)
        except:
            pass
        decoder = smp.FPN(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'unet++':
        try:
            decoder_options = dict(config.decoder_unet_plus_plus)
        except:
            pass
        decoder = smp.UnetPlusPlus(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'manet':
        try:
            decoder_options = dict(config.decoder_manet)
        except:
            pass
        decoder = smp.MAnet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'linknet':
        try:
            decoder_options = dict(config.decoder_linknet)
        except:
            pass
        decoder = smp.Linknet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'pspnet':
        try:
            decoder_options = dict(config.decoder_pspnet)
        except:
            pass
        decoder = smp.PSPNet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'pspnet':
        try:
            decoder_options = dict(config.decoder_pspnet)
        except:
            pass
        decoder = smp.PSPNet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'pan':
        try:
            decoder_options = dict(config.decoder_pan)
        except:
            pass
        decoder = smp.PAN(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'deeplabv3':
        try:
            decoder_options = dict(config.decoder_deeplabv3)
        except:
            pass
        decoder = smp.DeepLabV3(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'deeplabv3plus':
        try:
            decoder_options = dict(config.decoder_deeplabv3plus)
        except:
            pass
        decoder = smp.DeepLabV3Plus(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    return decoder


class Segm_Models_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        act = get_act(act_type=config.model.act)

        self.num_target_instruments = len(prefer_target_instrument(config))
        self.num_subbands = config.model.num_subbands

        dim_c = self.num_subbands * config.audio.num_channels * 2
        c = config.model.num_channels
        f = config.audio.dim_f // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.unet_model = get_decoder(config, c)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

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

    def forward(self, x):

        mix = x = self.cac2cws(x)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        x = self.unet_model(x)

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = self.cws2cac(x)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)
        return x
