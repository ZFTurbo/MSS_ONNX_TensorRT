# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from typing import Dict, Union

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import demix, get_model_from_config, normalize_audio, denormalize_audio, draw_spectrogram
from utils import prefer_target_instrument, apply_tta, load_start_checkpoint, load_onnx_model, load_engine_model

import warnings

warnings.filterwarnings("ignore")


def run_folder(model, args, config, device, verbose: bool = False):
    """
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module or onnxruntime.InferenceSession
        Pre-trained model for source separation or ONNX inference session.
    args : Namespace
        Arguments containing input folder, output folder, and processing options.
    config : Dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True
    start_time = time.time()

    for path in mixture_paths:
        print(f"Processing track: {path}")
        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f'Cannot read track: {format(path)}')
            print(f'Error message: {str(e)}')
            continue
        

        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    print(f'Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)

        waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar, use_onnx=args.use_onnx, use_tensorrt=args.use_tensorrt, use_compile=args.use_compile)

        if args.use_tta and not args.use_onnx:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        file_name = os.path.splitext(os.path.basename(path))[0]
        file_name = '_'.join(file_name.split('_')[:-1])

        output_dir = os.path.join(args.store_dir, file_name)

        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            subtype = 'PCM_16' if args.flac_file and args.pcm_type == 'PCM_16' else 'FLOAT'

            output_path = f"{output_dir}_{instr}.{codec}"
            sf.write(output_path, estimates.T, sr, subtype=subtype)
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{instr}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, default="", help="path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Code will generate spectrograms for resulted stems."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental if provided")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24'], default='PCM_24',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--use_onnx", action='store_true', help="Use ONNX model for inference")
    parser.add_argument("--onnx_model_path", type=str, default='', help="Path to ONNX model file")
    parser.add_argument("--use_tensorrt", action='store_true', help="Use TensorRT model for inference")
    parser.add_argument("--tensorrt_model_path", type=str, default="", help="Path to TensorRT model for inference")
    parser.add_argument("--use_compile", action='store_true', help="Use torch.compile for models")
    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def proc_folder(dict_args):
    args = parse_args(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True
    if args.use_compile:
        model, config = get_model_from_config(f"my_{args.model_type}", args.config_path)
    else:
        model, config = get_model_from_config(args.model_type, args.config_path)
    
    if args.start_check_point != '':
        load_start_checkpoint(args, model, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    if args.use_onnx:
        if not args.onnx_model_path:
            raise ValueError("ONNX model path must be provided when using ONNX inference")
        session = load_onnx_model(args.onnx_model_path, device)
    elif args.use_tensorrt:
        if not args.tensorrt_model_path:
            raise ValueError("TensorRT model path must be provided when using ONNX inference")
        engine = load_engine_model(args.tensorrt_model_path)
    else:
        if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        model = model.to(device)
    
    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))
    if args.use_onnx:
        run_folder(session, args, config, device, verbose=True)
    elif args.use_tensorrt:
        run_folder(engine, args, config, device, verbose=True)
    elif args.use_compile:
        model = torch.compile(model)
        run_folder(model, args, config, device, verbose=True)
    else:
        run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
