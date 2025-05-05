import torch
import torch.nn as nn
from typing import Dict, Any, Union
import onnxruntime
import numpy as np
import argparse
import os
from utils import get_model_from_config, load_start_checkpoint
from models.preprocess import STFT, HTDemucs_processor, BS_roformer_processor, Mel_band_roformer_processor

class ModelExporter:
    def __init__(self, config: Dict[str, Any], model: nn.Module, model_type: str):
        """
        Initialize the model exporter.
        
        Args:
            config: Model configuration dictionary
            model: PyTorch model to export
            model_type: Type of model ('htdemucs', 'bs_roformer', 'mel_band_roformer', etc.)
        """
        self.config = config
        self.model = model
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        if model_type == 'htdemucs':
            self.preprocessor = HTDemucs_processor(config)
        elif model_type == 'bs_roformer':
            self.preprocessor = BS_roformer_processor(**dict(config.model))
        elif model_type == 'mel_band_roformer':
            self.preprocessor = Mel_band_roformer_processor(**dict(config.model))
        else:
            self.preprocessor = STFT(config.audio)

    def _get_input_shape(self) -> tuple:
        """
        Determine input shape based on model type and configuration.
        """
        if self.model_type == 'htdemucs':
            chunk_size = self.config.training.samplerate * self.config.training.segment
            return (1, 2, chunk_size)
        else:
            chunk_size = self.config.audio.chunk_size
            return (1, 2, chunk_size)

    def _create_dummy_input(self) -> torch.Tensor:
        """
        Create dummy input tensor for ONNX export.
        """
        input_shape = self._get_input_shape()
        return torch.randn(input_shape, device=self.device)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input tensor according to model type.
        """
        if self.model_type == 'htdemucs':
            # For HTDemucs, we need both raw audio and STFT representation
            stft_repr = self.preprocessor.stft(x)
            return stft_repr, x
        else:
            # For other models, just return STFT representation
            return self.preprocessor.stft(x)

    def export(self, output_path: str, opset_version: int = 17) -> None:
        """
        Export the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            opset_version: ONNX opset version to use
        """
        dummy_input = self._create_dummy_input()
        
        if self.model_type == 'htdemucs':
            if self.config.inference > 1:
                stft_repr, raw_audio = self._preprocess_input(dummy_input)
                torch.onnx.export(
                    self.model,
                    (stft_repr, raw_audio),
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['stft_input', 'raw_audio'],
                    output_names=['output_x', 'output_xt'],
                    dynamic_axes={
                        'stft_input': {0: 'batch_size'},
                        'raw_audio': {0: 'batch_size'},
                        'output_x': {0: 'batch_size'},
                        'output_xt': {0: 'batch_size'}
                    }
                )
            else:
                stft_repr, raw_audio = self._preprocess_input(dummy_input)
                torch.onnx.export(
                    self.model,
                    (stft_repr, raw_audio),
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['stft_input', 'raw_audio'],
                    output_names=['output_x', 'output_xt']
                )
        else:
            if self.config.inference > 1:
                stft_repr = self._preprocess_input(dummy_input)
                torch.onnx.export(
                    self.model,
                    stft_repr,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            else:
                stft_repr = self._preprocess_input(dummy_input)
                torch.onnx.export(
                    self.model,
                    stft_repr,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        ort_session = onnxruntime.InferenceSession(output_path)
        
        if self.model_type == 'htdemucs':
            ort_inputs = {
                'stft_input': stft_repr.cpu().numpy(),
                'raw_audio': raw_audio.cpu().numpy()
            }
        else:
            ort_inputs = {ort_session.get_inputs()[0].name: stft_repr.cpu().numpy()}
            
        ort_outputs = ort_session.run(None, ort_inputs)
        
        with torch.no_grad():
            if self.model_type == 'htdemucs':
                torch_output = self.model(stft_repr, raw_audio)
            else:
                torch_output = self.model(stft_repr)
        
        np.testing.assert_allclose(
            torch_output.cpu().numpy(),
            ort_outputs[0],
            rtol=1e-3,
            atol=1e-5
        )
        
        print(f"Model successfully exported to {output_path}")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {torch_output.shape}")

def export_model_to_onnx(
    config: Dict[str, Any],
    model: nn.Module,
    model_type: str,
    output_path: str,
    opset_version: int = 12
) -> None:
    """
    Helper function to export a model to ONNX format.
    
    Args:
        config: Model configuration dictionary
        model: PyTorch model to export
        model_type: Type of model ('htdemucs', 'bs_roformer', 'mel_band_roformer', etc.)
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version to use
    """
    exporter = ModelExporter(config, model, model_type)
    exporter.export(output_path, opset_version)

def parse_args():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX format')
    parser.add_argument('--model_type', type=str, required=True,
                      help='Type of model (htdemucs, bs_roformer, mel_band_roformer, etc.)')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to model configuration file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save the ONNX model')
    parser.add_argument('--opset_version', type=int, default=17,
                      help='ONNX opset version to use')
    parser.add_argument('--force_cpu', action='store_true',
                      help='Force CPU usage even if CUDA is available')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, config = get_model_from_config(f"my_{args.model_type}", args.config_path)
    
    args.start_check_point = args.checkpoint_path
    load_start_checkpoint(args, model, type_='inference')
    
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    export_model_to_onnx(
        config=config,
        model=model,
        model_type=args.model_type,
        output_path=args.output_path,
        opset_version=args.opset_version
    )

if __name__ == "__main__":
    main()
