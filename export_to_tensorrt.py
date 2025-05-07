import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import os
import argparse
from typing import Dict, Any, Union, List
import onnx
import torch
import onnxruntime as ort
from models.preprocess import STFT, HTDemucs_processor, BS_roformer_processor, Mel_band_roformer_processor

class TensorRTExporter:
    def __init__(self, onnx_path: str, model_type: str, config: Dict[str, Any]):
        """
        Initialize TensorRT exporter.
        
        Args:
            onnx_path: Path to ONNX model
            model_type: Type of model ('htdemucs', 'bs_roformer', etc.)
            config: Model configuration dictionary
        """
        self.onnx_path = onnx_path
        self.model_type = model_type
        self.cfg = config
        self.onnx_path = onnx_path
        self.logger = None
        self.builder = None
        self.network = None
        self.config = None
        self.parser = None
        
        if model_type == 'htdemucs':
            self.preprocessor = HTDemucs_processor(config)
        elif model_type == 'bs_roformer':
            self.preprocessor = BS_roformer_processor(**dict(config.model))
        elif model_type == 'mel_band_roformer':
            self.preprocessor = Mel_band_roformer_processor(**dict(config.model))
        else:
            self.preprocessor = STFT(config.audio)

    def _set_optimization_profile(self, input_shapes: Dict[str, List[int]]):
        """
        Set optimization profile for dynamic shapes.
        
        Args:
            input_shapes: Dictionary of input names and their shapes
        """
        for name, shape in input_shapes.items():
            self.profile.set_shape(
                name,
                tuple([1] + shape[1:]), 
                tuple([self.cfg.inference.batch_size] + shape[1:]),
                tuple([self.cfg.inference.batch_size] + shape[1:])
            )
        self.config.add_optimization_profile(self.profile)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input tensor according to model type.
        """
        return self.preprocessor.stft(x)

    def _get_input_shapes(self) -> Dict[str, List[int]]:
        """
        Get input shapes based on model type.
        """
        if self.model_type == 'htdemucs':
            chunk_size = self.cfg.training.samplerate * self.cfg.training.segment
            return {
                'stft_input': list(self._preprocess_input(torch.randn((1, 2, chunk_size), device='cuda')).shape),
                'raw_audio': [1, 2, chunk_size]
            }
        else:
            chunk_size = self.cfg.audio.chunk_size
            print((1, 2, chunk_size))
            return {
                'input': list(self._preprocess_input(torch.randn((1, 2, chunk_size), device='cuda')).shape)
            }

    def export(self, output_path: str, fp16: bool = True) -> None:
        """
        Export ONNX model to TensorRT format.
        
        Args:
            output_path: Path to save TensorRT engine
            fp16: Whether to use FP16 precision
        """
        input_shapes = self._get_input_shapes()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.config = self.builder.create_builder_config()
        if self.cfg.inference.batch_size > 1:
            
            self.profile = self.builder.create_optimization_profile()
            self._set_optimization_profile(input_shapes)

        if fp16 and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
        
        self.parser = trt.OnnxParser(self.network, self.logger)
        
        with open(self.onnx_path, 'rb') as model:
            if not self.parser.parse(model.read()):
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        print("Building TensorRT engine...")
        engine = self.builder.build_engine_with_config(self.network, self.config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {output_path}")

def export_to_tensorrt(
    onnx_path: str,
    model_type: str,
    config: Dict[str, Any],
    output_path: str,
    fp16: bool = True
) -> None:
    """
    Helper function to export ONNX model to TensorRT format.
    
    Args:
        onnx_path: Path to ONNX model
        model_type: Type of model ('htdemucs', 'bs_roformer', etc.)
        config: Model configuration dictionary
        output_path: Path to save TensorRT engine
        fp16: Whether to use FP16 precision
    """
    exporter = TensorRTExporter(onnx_path, model_type, config)
    exporter.export(output_path, fp16)

def parse_args():
    parser = argparse.ArgumentParser(description='Export ONNX model to TensorRT format')
    parser.add_argument('--onnx_path', type=str, required=True,
                      help='Path to ONNX model')
    parser.add_argument('--model_type', type=str, required=True,
                      help='Type of model (htdemucs, bs_roformer, etc.)')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to model configuration file')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save TensorRT engine')
    parser.add_argument('--fp16', action='store_true',
                      help='Use FP16 precision')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.model_type == 'htdemucs':
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config_path)
    else:
        import yaml
        from ml_collections import ConfigDict
        with open(args.config_path, 'r') as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    export_to_tensorrt(
        onnx_path=args.onnx_path,
        model_type=args.model_type,
        config=config,
        output_path=args.output_path,
        fp16=args.fp16
    )

if __name__ == "__main__":
    main() 