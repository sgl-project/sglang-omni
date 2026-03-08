
# run huggingface benchmark
python bench_huggingface.py --enable-image > bench_huggingface_image.log 2>&1
python bench_huggingface.py --enable-video > bench_huggingface_video.log 2>&1
python bench_huggingface.py --enable-audio > bench_huggingface_audio.log 2>&1
python bench_huggingface.py --enable-image --enable-video > bench_huggingface_image_video.log 2>&1
python bench_huggingface.py --enable-image --enable-audio > bench_huggingface_image_audio.log 2>&1
python bench_huggingface.py --enable-video --enable-audio > bench_huggingface_video_audio.log 2>&1
python bench_huggingface.py --enable-image --enable-video --enable-audio > bench_huggingface_image_video_audio.log 2>&1

# run sglang-omni benchmark
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg > bench_sglang_omni_image.log 2>&1
python bench_sglang_omni.py --video-path ../../tests/data/draw.mp4 > bench_sglang_omni_video.log 2>&1
python bench_sglang_omni.py --audio-path ../../tests/data/cough.wav > bench_sglang_omni_audio.log 2>&1
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg --video-path ../../tests/data/draw.mp4 > bench_sglang_omni_image_video.log 2>&1
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg --audio-path ../../tests/data/cough.wav > bench_sglang_omni_image_audio.log 2>&1
python bench_sglang_omni.py --video-path ../../tests/data/draw.mp4 --audio-path ../../tests/data/cough.wav > bench_sglang_omni_video_audio.log 2>&1
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg --video-path ../../tests/data/draw.mp4 --audio-path ../../tests/data/cough.wav > bench_sglang_omni_image_video_audio.log 2>&1
