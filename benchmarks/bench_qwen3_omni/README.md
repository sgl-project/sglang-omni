# Benchmarking Qwen3-Omni with Hugging Face and SGLang-Omni

## Hugging Face Benchmark

Please make sure you have installed `torchcodec` for video processing.

```shell
uv pip install torchcodec==0.9
```

```bash
python bench_huggingface.py --image-path ../../tests/data/cars.jpg
python bench_huggingface.py --video-path ../../tests/data/draw.mp4
python bench_huggingface.py --audio-path ../../tests/data/cough.wav
python bench_huggingface.py --image-path ../../tests/data/cars.jpg --video-path ../../tests/data/draw.mp4
python bench_huggingface.py --image-path ../../tests/data/cars.jpg --audio-path ../../tests/data/cough.wav
python bench_huggingface.py --video-path ../../tests/data/draw.mp4 --audio-path ../../tests/data/cough.wav
python bench_huggingface.py --image-path ../../tests/data/cars.jpg --video-path ../../tests/data/draw.mp4 --audio-path ../../tests/data/cough.wav
```

## SGLang-Omni Benchmark

```shell
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg
python bench_sglang_omni.py --video-path ../../tests/data/draw.mp4
python bench_sglang_omni.py --audio-path ../../tests/data/cough.wav
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg --video-path ../../tests/data/draw.mp4
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg --audio-path ../../tests/data/cough.wav
python bench_sglang_omni.py --video-path ../../tests/data/draw.mp4 --audio-path ../../tests/data/cough.wav
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg --video-path ../../tests/data/draw.mp4 --audio-path ../../tests/data/cough.wav
```
