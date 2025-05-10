# Запуск моделей в ONNX/TensorRT

### Запуск с ONNX

Для запуска модели в формате ONNX используйте следующие параметры:

```bash
python inference.py \
    --model_type htdemucs \
    --config_path path/to/config.yaml \
    --input_folder path/to/input \
    --store_dir path/to/output \
    --use_onnx \
    --onnx_model_path path/to/model.onnx
```

Основные параметры:
- `--use_onnx`: Включить использование ONNX модели
- `--onnx_model_path`: Путь к ONNX модели
- `--model_type`: Тип модели (htdemucs, bs_roformer, mel_band_roformer и т.д.)
- `--config_path`: Путь к конфигурационному файлу
- `--input_folder`: Папка с входными аудио файлами
- `--store_dir`: Папка для сохранения результатов

### Запуск с TensorRT

Для запуска модели в формате TensorRT используйте следующие параметры:

```bash
python inference.py \
    --model_type htdemucs \
    --config_path path/to/config.yaml \
    --input_folder path/to/input \
    --store_dir path/to/output \
    --use_tensorrt \
    --tensorrt_model_path path/to/model.engine
```

Основные параметры:
- `--use_tensorrt`: Включить использование TensorRT модели
- `--tensorrt_model_path`: Путь к TensorRT engine файлу
- `--model_type`: Тип модели (htdemucs, bs_roformer, mel_band_roformer и т.д.)
- `--config_path`: Путь к конфигурационному файлу
- `--input_folder`: Папка с входными аудио файлами
- `--store_dir`: Папка для сохранения результатов

# Экспорт в ONNX

Модуль для экспорта моделей разделения источников звука из PyTorch в формат ONNX.

## Описание

Модуль `export_to_onnx` предоставляет функциональность для конвертации моделей разделения источников звука из формата PyTorch в ONNX. Поддерживает различные типы моделей, включая:
- HTDemucs
- BS Roformer
- Mel Band Roformer
- mdx23c
- segm

## Использование

### Экспорт в ONNX

```python
from export_to_onnx import export_model_to_onnx

export_model_to_onnx(
    config=your_config,
    model=your_model,
    model_type='htdemucs',
    output_path='path/to/output/model.onnx'
)
```

### Как отдельный скрипт

```bash
python export_to_onnx.py \
    --model_type htdemucs \
    --config_path path/to/config.yaml \
    --checkpoint_path path/to/checkpoint.pth \
    --output_path path/to/output/model.onnx \
    --opset_version 17 \
    --force_cpu
```

### Параметры командной строки

- `--model_type`: Тип модели (htdemucs, bs_roformer, mel_band_roformer и т.д.)
- `--config_path`: Путь к файлу конфигурации модели
- `--checkpoint_path`: Путь к чекпоинту модели
- `--output_path`: Путь для сохранения ONNX модели
- `--opset_version`: Версия ONNX opset (по умолчанию 17)
- `--force_cpu`: Принудительное использование CPU даже при наличии CUDA

# Экспорт в TensorRT

## Описание

Модуль `export_to_tensorrt` предоставляет функциональность для конвертации моделей разделения источников звука из формата ONNX в TensorRT Engine. Поддерживает различные типы моделей, включая:
- HTDemucs
- BS Roformer
- Mel Band Roformer
- mdx23c
- segm

## Использование

### Экспорт в TensorRT

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

### Как отдельный скрипт

```bash
python export_to_tensorrt.py \
    --onnx_path path/to/model.onnx \
    --model_type htdemucs \
    --config_path path/to/config.yaml \
    --output_path path/to/output/model.engine \
    --fp16
```

### Параметры командной строки

- `--onnx_path`: Путь к ONNX модели
- `--model_type`: Тип модели (htdemucs, bs_roformer, mel_band_roformer и т.д.)
- `--config_path`: Путь к файлу конфигурации модели
- `--output_path`: Путь для сохранения TensorRT engine
- `--fp16`/`--fp8`: Использовать FP16 точность (опционально)
