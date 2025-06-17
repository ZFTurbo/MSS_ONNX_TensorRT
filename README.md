# Running Models in ONNX/TensorRT

### Running with ONNX

To run a model in ONNX format, use the following parameters:

```bash
python inference.py \
    --model_type htdemucs \
    --config_path path/to/config.yaml \
    --input_folder path/to/input \
    --store_dir path/to/output \
    --use_onnx \
    --onnx_model_path path/to/model.onnx
```

Key parameters:
- `--use_onnx`: Enable the use of an ONNX model
- `--onnx_model_path`: Path to the ONNX model
- `--model_type`: Model type (htdemucs, bs_roformer, mel_band_roformer, etc.)
- `--config_path`: Path to the configuration file
- `--input_folder`: Folder with input audio files
- `--store_dir`: Folder to save the results

### Running with TensorRT

To run a model in TensorRT format, use the following parameters:

```bash
python inference.py \
    --model_type htdemucs \
    --config_path path/to/config.yaml \
    --input_folder path/to/input \
    --store_dir path/to/output \
    --use_tensorrt \
    --tensorrt_model_path path/to/model.engine
```

Key parameters:
- `--use_tensorrt`: Enable the use of a TensorRT model
- `--tensorrt_model_path`: Path to the TensorRT engine file
- `--model_type`: Model type (htdemucs, bs_roformer, mel_band_roformer, etc.)
- `--config_path`: Path to the configuration file
- `--input_folder`: Folder with input audio files
- `--store_dir`: Folder to save the results

# Exporting to ONNX

A module for exporting audio source separation models from PyTorch to ONNX format.

## Description

The `export_to_onnx` module provides functionality for converting audio source separation models from PyTorch format to ONNX. It supports various model types, including:
- HTDemucs
- BS Roformer
- Mel Band Roformer
- mdx23c
- segm

## Usage

### Exporting to ONNX

```python
from export_to_onnx import export_model_to_onnx

export_model_to_onnx(
    config=your_config,
    model=your_model,
    model_type='htdemucs',
    output_path='path/to/output/model.onnx'
)
```

### As a standalone script

```bash
python export_to_onnx.py \
    --model_type htdemucs \
    --config_path path/to/config.yaml \
    --checkpoint_path path/to/checkpoint.pth \
    --output_path path/to/output/model.onnx \
    --opset_version 17 \
    --force_cpu
```

### Command-line parameters

- `--model_type`: Model type (htdemucs, bs_roformer, mel_band_roformer, etc.)
- `--config_path`: Path to the model's configuration file
- `--checkpoint_path`: Path to the model checkpoint
- `--output_path`: Path to save the ONNX model
- `--opset_version`: ONNX opset version (default is 17)
- `--force_cpu`: Force CPU usage even if CUDA is available

# Exporting to TensorRT

## Description

The `export_to_tensorrt` module provides functionality for converting audio source separation models from ONNX format to a TensorRT Engine. It supports various model types, including:
- HTDemucs
- BS Roformer
- Mel Band Roformer
- mdx23c
- segm

## Usage

### Exporting to TensorRT

```python
from export_to_tensorrt import export_to_tensorrt

export_to_tensorrt(
    onnx_path='path/to/model.onnx',
    model_type='htdemucs',
    config=your_config,
    output_path='path/to/output/model.engine',
    fp16=True
)
```

### As a standalone script

```bash
python export_to_tensorrt.py \
    --onnx_path path/to/model.onnx \
    --model_type htdemucs \
    --config_path path/to/config.yaml \
    --output_path path/to/output/model.engine \
    --fp16
```

### Command-line parameters

- `--onnx_path`: Path to the ONNX model
- `--model_type`: Model type (htdemucs, bs_roformer, mel_band_roformer, etc.)
- `--config_path`: Path to the model's configuration file
- `--output_path`: Path to save the TensorRT engine
- `--fp16`/`--fp8`: Use FP16 precision (optional)
