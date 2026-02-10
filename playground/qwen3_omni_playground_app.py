import io
import os

import torch

os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from argparse import ArgumentParser

import gradio as gr
import gradio.processing_utils as processing_utils
import modelscope_studio.components.antd as antd
import modelscope_studio.components.base as ms
import numpy as np
import soundfile as sf
from gradio_client import utils as client_utils
from qwen_omni_utils import process_mm_info

import base64
import numpy as np
from scipy.io import wavfile  # 使用 scipy 保存 wav 文件，更简单支持 int16

import soundfile as sf
from openai import OpenAI

import base64

import os
import oss2
import json
import time
import subprocess
import numpy as np

OSS_RETRY = 10
OSS_RETRY_DELAY = 3
WAV_BIT_RATE = 16
WAV_SAMPLE_RATE = os.environ.get("WAV_SAMPLE_RATE", 16000)

# OSS_CONFIG_PATH = "/mnt/workspace/feizi.wx/.oss_config.json"

endpoint = os.getenv("OSS_ENDPOINT")
region = os.getenv("OSS_REGION")
bucket_name = os.getenv("OSS_BUCKET_NAME")
API_KEY = os.environ['API_KEY']
OSS_ACCESS_KEY_ID = os.environ['OSS_ACCESS_KEY_ID']
OSS_ACCESS_KEY_SECRET = os.environ['OSS_ACCESS_KEY_SECRET']
OSS_CONFIG_PATH = {}


class OSSReader:

    def __init__(self):
        # 初始化OSS配置
        self.bucket2object = {
            bucket_name:
            oss2.Bucket(oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET),
                        endpoint, bucket_name),
        }
        print(
            f"Loaded OSS config from: {OSS_CONFIG_PATH}\nSupported buckets: {list(self.bucket2object.keys())}"
        )

    def _parse_oss_path(self, oss_path):
        """解析oss路径，返回bucket名称和实际路径"""
        assert oss_path.startswith("oss://"), f"Invalid oss path {oss_path}"
        bucket_name, object_key = oss_path.split("oss://")[-1].split("/", 1)
        object_key = f"studio-temp/Qwen3-Omni-Demo/{object_key}"
        return bucket_name, object_key

    def _retry_operation(self,
                         func,
                         *args,
                         retries=OSS_RETRY,
                         delay=OSS_RETRY,
                         **kwargs):
        """通用的重试机制"""
        for _ in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Retry: {_} Error: {str(e)}")
                if _ == retries - 1:
                    raise e
                time.sleep(delay)

    def get_public_url(self, oss_path):
        bucket_name, object_key = self._parse_oss_path(oss_path)
        url = self._retry_operation(self.bucket2object[bucket_name].sign_url,
                                    'GET',
                                    object_key,
                                    600,
                                    slash_safe=True).replace(
                                        'http://', 'https://')
        return url.replace("-internal", '')

    def file_exists(self, oss_path):
        """判断文件是否存在"""
        bucket_name, object_key = self._parse_oss_path(oss_path)
        return self._retry_operation(
            self.bucket2object[bucket_name].object_exists, object_key)

    def download_file(self, oss_path, local_path):
        """下载OSS上的文件到本地"""
        bucket_name, object_key = self._parse_oss_path(oss_path)
        self._retry_operation(
            self.bucket2object[bucket_name].get_object_to_file, object_key,
            local_path)

    def upload_file(self, local_path, oss_path, overwrite=True):
        """上传本地文件到OSS"""
        bucket_name, object_key = self._parse_oss_path(oss_path)
        # 检查文件是否存在
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file {local_path} does not exist")
        # 检查目标文件是否存在（当overwrite=False时）
        if not overwrite and self.file_exists(oss_path):
            print(f"File {oss_path} already exists, skip upload")
            return False
        # 执行上传操作
        try:
            self._retry_operation(
                self.bucket2object[bucket_name].put_object_from_file,
                object_key, local_path)
            return True
        except Exception as e:
            print(f"Upload failed: {str(e)}")
            return False

    def upload_audio_from_array(self,
                                data,
                                sample_rate,
                                oss_path,
                                overwrite=True):
        """将音频数据保存为WAV格式并上传到OSS"""
        bucket_name, object_key = self._parse_oss_path(oss_path)

        # 检查目标文件是否存在（当overwrite=False时）
        if not overwrite and self.file_exists(oss_path):
            print(f"File {oss_path} already exists, skip upload")
            return False

        try:
            # 使用 BytesIO 在内存中生成 WAV 格式数据
            import wave
            from io import BytesIO

            byte_io = BytesIO()
            with wave.open(byte_io, 'wb') as wf:
                wf.setnchannels(1)  # 单声道
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(sample_rate)  # 设置采样率
                # 将 float32 数据转换为 int16 并写入 WAV
                data_int16 = np.clip(data, -1, 1) * 32767
                data_int16 = data_int16.astype(np.int16)
                wf.writeframes(data_int16.tobytes())

            # 上传到 OSS
            self._retry_operation(self.bucket2object[bucket_name].put_object,
                                  object_key, byte_io.getvalue())
            return True
        except Exception as e:
            print(f"Upload failed: {str(e)}")
            return False

    def get_object(self, oss_path):
        """读取OSS上的音频文件，返回音频数据和采样率"""
        bucket_name, object_key = self._parse_oss_path(oss_path)
        return self._retry_operation(
            self.bucket2object[bucket_name].get_object, object_key)

    def read_text_file(self, oss_path):
        """读取OSS上的文本文件"""
        bucket_name, object_key = self._parse_oss_path(oss_path)
        result = self._retry_operation(
            self.bucket2object[bucket_name].get_object, object_key)
        return result.read().decode('utf-8')

    def read_audio_file(self, oss_path):
        """读取OSS上的音频文件，返回音频数据和采样率"""
        bucket_name, object_key = self._parse_oss_path(oss_path)
        result = self._retry_operation(
            self.bucket2object[bucket_name].get_object, object_key)
        # ffmpeg 命令：从标准输入读取音频并输出PCM浮点数据
        command = [
            'ffmpeg',
            '-i',
            '-',  # 输入来自管道
            '-ar',
            str(WAV_SAMPLE_RATE),  # 输出采样率
            '-ac',
            '1',  # 单声道
            '-f',
            'f32le',  # 指定输出格式
            '-'  # 输出到管道
        ]
        # 启动ffmpeg子进程
        process = subprocess.Popen(command,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        # 写入音频字节并获取输出
        stdout_data, stderr_data = process.communicate(input=result.read())
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr_data.decode('utf-8')}")
        # 将PCM数据转换为numpy数组
        wav_data = np.frombuffer(stdout_data, dtype=np.float32)
        return wav_data, WAV_SAMPLE_RATE

    def get_wav_duration_from_bin(self, oss_path):
        oss_bin_path = oss_path + ".ar16k.bin"
        bucket_name, object_key = self._parse_oss_path(oss_bin_path)
        metadata = self._retry_operation(
            self.bucket2object[bucket_name].get_object_meta, object_key)
        duration = float(metadata.headers['Content-Length']) / (16000 * 2)
        return duration

    def read_wavdata_from_oss(self,
                              oss_path,
                              start=None,
                              end=None,
                              force_bin=False):
        bucket_name, object_key = self._parse_oss_path(oss_path)
        oss_bin_key = object_key + ".ar16k.bin"
        if start is None or end is None:
            if self.bucket2object[bucket_name].object_exists(oss_bin_key):
                wav_data = self._retry_operation(
                    self.bucket2object[bucket_name].get_object,
                    oss_bin_key).read()
            elif not force_bin:
                wav_data, _ = self.read_audio_file(oss_path)
            else:
                raise ValueError(f"Cannot find bin file for {oss_path}")
        else:
            bytes_per_second = WAV_SAMPLE_RATE * (WAV_BIT_RATE // 8)
            # 计算字节偏移量
            start_offset = round(start * bytes_per_second)
            end_offset = round(end * bytes_per_second)
            if not (end_offset - start_offset) % 2:
                end_offset -= 1
            # 使用范围请求只获取指定字节范围的数据
            wav_data = self._retry_operation(
                self.bucket2object[bucket_name].get_object,
                oss_bin_key,
                byte_range=(start_offset, end_offset),
                headers={
                    'x-oss-range-behavior': 'standard'
                }).read()
        if not isinstance(wav_data, np.ndarray):
            wav_data = np.frombuffer(wav_data, np.int16).flatten() / 32768.0
        return wav_data.astype(np.float32)

    def _list_files_by_suffix(self, oss_dir, suffix):
        """递归搜索以某个后缀结尾的所有文件，返回所有文件的OSS路径列表"""
        bucket_name, dir_key = self._parse_oss_path(oss_dir)
        file_list = []

        def _recursive_list(prefix):
            for obj in oss2.ObjectIterator(self.bucket2object[bucket_name],
                                           prefix=prefix,
                                           delimiter='/'):
                if obj.is_prefix():  # 如果是目录，递归搜索
                    _recursive_list(obj.key)
                elif obj.key.endswith(suffix):
                    file_list.append(f"oss://{bucket_name}/{obj.key}")

        _recursive_list(dir_key)
        return file_list

    def list_files_by_suffix(self, oss_dir, suffix):
        return self._retry_operation(self._list_files_by_suffix, oss_dir,
                                     suffix)

    def _list_files_by_prefix(self, oss_dir, file_prefix):
        """递归搜索以某个后缀结尾的所有文件，返回所有文件的OSS路径列表"""
        bucket_name, dir_key = self._parse_oss_path(oss_dir)
        file_list = []

        def _recursive_list(prefix):
            for obj in oss2.ObjectIterator(self.bucket2object[bucket_name],
                                           prefix=prefix,
                                           delimiter='/'):
                if obj.is_prefix():  # 如果是目录，递归搜索
                    _recursive_list(obj.key)
                elif os.path.basename(obj.key).startswith(file_prefix):
                    file_list.append(f"oss://{bucket_name}/{obj.key}")

        _recursive_list(dir_key)
        return file_list

    def list_files_by_prefix(self, oss_dir, file_prefix):
        return self._retry_operation(self._list_files_by_prefix, oss_dir,
                                     file_prefix)


def encode_base64(base64_path):
    with open(base64_path, "rb") as base64_file:
        return base64.b64encode(base64_file.read()).decode("utf-8")


def _load_model_processor(args):
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    model = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    return model, None


oss_reader = OSSReader()


def _launch_demo(args, model, processor):
    # Voice settings
    VOICE_OPTIONS = {
        "Cherry / 芊悦": "Cherry",
        "Serena / 苏瑶": "Serena",
        "Ethan / 晨煦": "Ethan",
        "Chelsie / 千雪": "Chelsie",
        "Momo / 茉兔": "Momo",
        "Vivian / 十三": "Vivian",
        "Moon / 月白": "Moon",
        "Maia / 四月": "Maia",
        "Kai / 凯": "Kai",
        "Nofish / 不吃鱼": "Nofish",
        "Bella / 萌宝": "Bella",
        "Jennifer / 詹妮弗": "Jennifer",
        "Ryan / 甜茶": "Ryan",
        "Katerina / 卡捷琳娜": "Katerina",
        "Aiden / 艾登": "Aiden",
        "Bodega / 西班牙语-博德加": "Bodega",
        "Alek / 俄语-阿列克": "Alek",
        "Dolce / 意大利语-多尔切": "Dolce",
        "Sohee / 韩语-素熙": "Sohee",
        "Ono Anna / 日语-小野杏": "Ono Anna",
        "Lenn / 德语-莱恩": "Lenn",
        "Sonrisa / 西班牙语拉美-索尼莎": "Sonrisa",
        "Emilien / 法语-埃米尔安": "Emilien",
        "Andre / 葡萄牙语欧-安德雷": "Andre",
        "Radio Gol / 葡萄牙语巴-拉迪奥·戈尔": "Radio Gol",
        "Eldric Sage / 精品百人-沧明子": "Eldric Sage",
        "Mia / 精品百人-乖小妹": "Mia",
        "Mochi / 精品百人-沙小弥": "Mochi",
        "Bellona / 精品百人-燕铮莺": "Bellona",
        "Vincent / 精品百人-田叔": "Vincent",
        "Bunny / 精品百人-萌小姬": "Bunny",
        "Neil / 精品百人-阿闻": "Neil",
        "Elias / 墨讲师": "Elias",
        "Arthur / 精品百人-徐大爷": "Arthur",
        "Nini / 精品百人-邻家妹妹": "Nini",
        "Ebona / 精品百人-诡婆婆": "Ebona",
        "Seren / 精品百人-小婉": "Seren",
        "Pip / 精品百人-调皮小新": "Pip",
        "Stella / 精品百人-美少女阿月": "Stella",
        "Li / 南京-老李": "Li",
        "Marcus / 陕西-秦川": "Marcus",
        "Roy / 闽南-阿杰": "Roy",
        "Peter / 天津-李彼得": "Peter",
        "Eric / 四川-程川": "Eric",
        "Rocky / 粤语-阿强": "Rocky",
        "Kiki / 粤语-阿清": "Kiki",
        "Sunny / 四川-晴儿": "Sunny",
        "Jada / 上海-阿珍": "Jada",
        "Dylan / 北京-晓东": "Dylan",
    }
    DEFAULT_VOICE = "Cherry / 芊悦"

    default_system_prompt = ''

    language = args.ui_language

    def get_text(text: str, cn_text: str):
        if language == 'en':
            return text
        if language == 'zh':
            return cn_text
        return text

    def to_mp4(path):
        import subprocess
        if path and path.endswith(".webm"):
            mp4_path = path.replace(".webm", ".mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    path,
                    "-c:v",
                    "libx264",  # 使用 H.264
                    "-preset",
                    "ultrafast",  # 最快速度！
                    "-tune",
                    "fastdecode",  # 优化快速解码（利于后续处理）
                    "-pix_fmt",
                    "yuv420p",  # 兼容性像素格式
                    "-c:a",
                    "aac",  # 音频编码
                    "-b:a",
                    "128k",  # 可选：限制音频比特率加速
                    "-threads",
                    "0",  # 使用所有线程
                    "-f",
                    "mp4",
                    mp4_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            return mp4_path
        return path  # 已经是 mp4 或 None

    def format_history(history: list, system_prompt: str):
        print(history)
        messages = []
        if system_prompt != "":
            messages.append({
                "role":
                "system",
                "content": [{
                    "type": "text",
                    "text": system_prompt
                }]
            })

        current_user_content = []

        for item in history:
            role = item['role']
            content = item['content']

            if role != "user":
                if current_user_content:
                    messages.append({
                        "role": "user",
                        "content": current_user_content
                    })
                    current_user_content = []

                if isinstance(content, str):
                    messages.append({
                        "role":
                        role,
                        "content": [{
                            "type": "text",
                            "text": content
                        }]
                    })
                else:
                    pass
                continue

            if isinstance(content, str):
                current_user_content.append({"type": "text", "text": content})
            elif isinstance(content, (list, tuple)):
                for file_path in content:
                    mime_type = client_utils.get_mimetype(file_path)
                    media_type = None

                    if mime_type.startswith("image"):
                        media_type = "image_url"
                    elif mime_type.startswith("video"):
                        media_type = "video_url"
                        file_path = to_mp4(file_path)
                    elif mime_type.startswith("audio"):
                        media_type = "input_audio"

                    if media_type:
                        # base64_media = encode_base64(file_path)
                        import uuid
                        request_id = str(uuid.uuid4())
                        oss_path = f"oss://{bucket_name}//studio-temp/Qwen3-Omni-Demo/" + request_id
                        oss_reader.upload_file(file_path, oss_path)
                        media_url = oss_reader.get_public_url(oss_path)
                        if media_type == "input_audio":
                            current_user_content.append({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": media_url,
                                    "format": "wav",
                                },
                            })
                        if media_type == "image_url":
                            current_user_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": media_url
                                },
                            })
                        if media_type == "video_url":
                            current_user_content.append({
                                "type": "video_url",
                                "video_url": {
                                    "url": media_url
                                },
                            })
                    else:
                        current_user_content.append({
                            "type": "text",
                            "text": file_path
                        })

        if current_user_content:
            media_items = []
            text_items = []

            for item in current_user_content:
                if item["type"] == "text":
                    text_items.append(item)
                else:
                    media_items.append(item)

            messages.append({
                "role": "user",
                "content": media_items + text_items
            })

        return messages

    def predict(messages,
                voice_choice=DEFAULT_VOICE,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                return_audio=False,
                enable_thinking=False):
        # print('predict history: ', messages)
        if enable_thinking:
            return_audio = False
        if return_audio:
            completion = model.chat.completions.create(
                model="qwen3-omni-flash-2025-12-01",
                messages=messages,
                modalities=["text", "audio"],
                audio={
                    "voice": VOICE_OPTIONS[voice_choice],
                    "format": "wav"
                },
                extra_body={
                    'enable_thinking': False,
                    "top_k": top_k
                },
                stream_options={"include_usage": True},
                stream=True,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            completion = model.chat.completions.create(
                model="qwen3-omni-flash-2025-12-01",
                messages=messages,
                modalities=["text"],
                extra_body={
                    'enable_thinking': enable_thinking,
                    "top_k": top_k
                },
                stream_options={"include_usage": True},
                stream=True,
                temperature=temperature,
                top_p=top_p,
            )
        audio_string = ""
        output_text = ""
        reasoning_content = "<think>\n\n"  # 完整思考过程
        answer_content = ""  # 完整回复
        is_answering = False  # 是否进入回复阶段
        print(return_audio, enable_thinking)
        for chunk in completion:
            if chunk.choices:
                if hasattr(chunk.choices[0].delta, "audio"):
                    try:
                        audio_string += chunk.choices[0].delta.audio["data"]
                    except Exception as e:
                        output_text += chunk.choices[0].delta.audio[
                            "transcript"]
                        yield {"type": "text", "data": output_text}
                else:
                    delta = chunk.choices[0].delta
                    if enable_thinking:
                        if hasattr(delta, "reasoning_content"
                                   ) and delta.reasoning_content is not None:
                            if not is_answering:
                                print(delta.reasoning_content,
                                      end="",
                                      flush=True)
                            reasoning_content += delta.reasoning_content
                            yield {"type": "text", "data": reasoning_content}
                        if hasattr(delta, "content") and delta.content:
                            if not is_answering:
                                reasoning_content += "\n\n</think>\n\n"
                                is_answering = True
                            answer_content += delta.content
                            yield {
                                "type": "text",
                                "data": reasoning_content + answer_content
                            }
                    else:
                        if hasattr(delta, "content") and delta.content:
                            output_text += chunk.choices[0].delta.content
                            yield {"type": "text", "data": output_text}
            else:
                print(chunk.usage)

        wav_bytes = base64.b64decode(audio_string)
        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)

        if audio_string != "":
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_np, samplerate=24000, format="WAV")
            wav_io.seek(0)
            wav_bytes = wav_io.getvalue()
            audio_path = processing_utils.save_bytes_to_cache(
                wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE)
            yield {"type": "audio", "data": audio_path}

    def media_predict(audio,
                      video,
                      history,
                      system_prompt,
                      voice_choice,
                      temperature,
                      top_p,
                      top_k,
                      return_audio=False,
                      enable_thinking=False):
        # First yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        files = [audio, video]

        for f in files:
            if f:
                history.append({"role": "user", "content": (f, )})

        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

        formatted_history = format_history(
            history=history,
            system_prompt=system_prompt,
        )

        history.append({"role": "assistant", "content": ""})

        for chunk in predict(formatted_history, voice_choice, temperature,
                             top_p, top_k, return_audio, enable_thinking):
            print('chunk', chunk)
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # microphone
                    None,  # webcam
                    history,  # media_chatbot
                    gr.update(visible=False),  # submit_btn
                    gr.update(visible=True),  # stop_btn
                )
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })

        # Final yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    def chat_predict(text,
                     audio,
                     image,
                     video,
                     history,
                     system_prompt,
                     voice_choice,
                     temperature,
                     top_p,
                     top_k,
                     return_audio=False,
                     enable_thinking=False):

        # Process audio input
        if audio:
            history.append({"role": "user", "content": (audio, )})

        # Process text input
        if text:
            history.append({"role": "user", "content": text})

        # Process image input
        if image:
            history.append({"role": "user", "content": (image, )})

        # Process video input
        if video:
            history.append({"role": "user", "content": (video, )})

        formatted_history = format_history(history=history,
                                           system_prompt=system_prompt)

        yield None, None, None, None, history

        history.append({"role": "assistant", "content": ""})
        for chunk in predict(formatted_history, voice_choice, temperature,
                             top_p, top_k, return_audio, enable_thinking):
            print('chat_predict chunk', chunk)

            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history

    # --- CORRECTED UI LAYOUT ---
    with gr.Blocks(
            theme=gr.themes.Soft(font=[
                gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"
            ]),
            css=".gradio-container {max-width: none !important;}") as demo:
        gr.Markdown("# Qwen3-Omni Demo")
        gr.Markdown(
            "**Instructions**: Interact with the model through text, audio, images, or video. Use the tabs to switch between Online and Offline chat modes."
        )
        gr.Markdown(
            "**使用说明**：1️⃣ 点击音频录制按钮，或摄像头-录制按钮 2️⃣ 输入音频或者视频 3️⃣ 点击提交并等待模型的回答")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Parameters (参数)")
                system_prompt_textbox = gr.Textbox(label="System Prompt",
                                                   value=default_system_prompt,
                                                   lines=4,
                                                   max_lines=8)
                voice_choice = gr.Dropdown(label="Voice Choice",
                                           choices=VOICE_OPTIONS,
                                           value=DEFAULT_VOICE,
                                           visible=True)
                return_audio = gr.Checkbox(label="Return Audio （返回语音）",
                                           value=True,
                                           interactive=True,
                                           elem_classes="checkbox-large")
                enable_thinking = gr.Checkbox(label="Enable Thinking （启用思维链）",
                                              value=False,
                                              interactive=True,
                                              elem_classes="checkbox-large")
                temperature = gr.Slider(label="Temperature",
                                        minimum=0.1,
                                        maximum=2.0,
                                        value=0.6,
                                        step=0.1)
                top_p = gr.Slider(label="Top P",
                                  minimum=0.05,
                                  maximum=1.0,
                                  value=0.95,
                                  step=0.05)
                top_k = gr.Slider(label="Top K",
                                  minimum=1,
                                  maximum=100,
                                  value=20,
                                  step=1)

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Online"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Audio-Video Input (音视频输入)")
                                microphone = gr.Audio(
                                    sources=['microphone'],
                                    type="filepath",
                                    label="Record Audio (录制音频)")
                                webcam = gr.Video(
                                    sources=['webcam', "upload"],
                                    label="Record/Upload Video (录制/上传视频)",
                                    elem_classes="media-upload")
                                with gr.Row():
                                    submit_btn_online = gr.Button(
                                        "Submit (提交)",
                                        variant="primary",
                                        scale=2)
                                    stop_btn_online = gr.Button("Stop (停止)",
                                                                visible=False,
                                                                scale=1)
                                clear_btn_online = gr.Button(
                                    "Clear History (清除历史)")
                            with gr.Column(scale=2):
                                # FIX: Re-added type="messages"
                                media_chatbot = gr.Chatbot(
                                    label="Chat History (对话历史)",
                                    type="messages",
                                    height=650,
                                    layout="panel",
                                    bubble_full_width=False,
                                    allow_tags=["think"],
                                    render=False)
                                media_chatbot.render()

                        def clear_history_online():
                            return [], None, None

                        submit_event_online = submit_btn_online.click(
                            fn=media_predict,
                            inputs=[
                                microphone, webcam, media_chatbot,
                                system_prompt_textbox, voice_choice,
                                temperature, top_p, top_k, return_audio,
                                enable_thinking
                            ],
                            outputs=[
                                microphone, webcam, media_chatbot,
                                submit_btn_online, stop_btn_online
                            ])
                        stop_btn_online.click(
                            fn=lambda: (gr.update(visible=True),
                                        gr.update(visible=False)),
                            outputs=[submit_btn_online, stop_btn_online],
                            cancels=[submit_event_online],
                            queue=False)
                        clear_btn_online.click(
                            fn=clear_history_online,
                            outputs=[media_chatbot, microphone, webcam])

                    with gr.TabItem("Offline"):
                        # FIX: Re-added type="messages"
                        chatbot = gr.Chatbot(label="Chat History (对话历史)",
                                             type="messages",
                                             height=550,
                                             layout="panel",
                                             bubble_full_width=False,
                                             allow_tags=["think"],
                                             render=False)
                        chatbot.render()

                        with gr.Accordion(
                                "📎 Click to upload multimodal files (点击上传多模态文件)",
                                open=False):
                            with gr.Row():
                                audio_input = gr.Audio(
                                    sources=["upload", 'microphone'],
                                    type="filepath",
                                    label="Audio",
                                    elem_classes="media-upload")
                                image_input = gr.Image(
                                    sources=["upload", 'webcam'],
                                    type="filepath",
                                    label="Image",
                                    elem_classes="media-upload")
                                video_input = gr.Video(
                                    sources=["upload", 'webcam'],
                                    label="Video",
                                    elem_classes="media-upload")

                        with gr.Row():
                            text_input = gr.Textbox(
                                show_label=False,
                                placeholder=
                                "Enter text or upload files and press Submit... (输入文本或者上传文件并点击提交)",
                                scale=7)
                            submit_btn_offline = gr.Button("Submit (提交)",
                                                           variant="primary",
                                                           scale=1)
                            stop_btn_offline = gr.Button("Stop (停止)",
                                                         visible=False,
                                                         scale=1)
                            clear_btn_offline = gr.Button("Clear (清空) ",
                                                          scale=1)

                        def clear_history_offline():
                            return [], None, None, None, None

                        submit_event_offline = gr.on(
                            triggers=[
                                submit_btn_offline.click, text_input.submit
                            ],
                            fn=chat_predict,
                            inputs=[
                                text_input, audio_input, image_input,
                                video_input, chatbot, system_prompt_textbox,
                                voice_choice, temperature, top_p, top_k,
                                return_audio, enable_thinking
                            ],
                            outputs=[
                                text_input, audio_input, image_input,
                                video_input, chatbot
                            ])
                        stop_btn_offline.click(
                            fn=lambda: (gr.update(visible=True),
                                        gr.update(visible=False)),
                            outputs=[submit_btn_offline, stop_btn_offline],
                            cancels=[submit_event_offline],
                            queue=False)
                        clear_btn_offline.click(fn=clear_history_offline,
                                                outputs=[
                                                    chatbot, text_input,
                                                    audio_input, image_input,
                                                    video_input
                                                ])

        gr.HTML("""
            <style>
                .media-upload { min-height: 160px; border: 2px dashed #ccc; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
                .media-upload:hover { border-color: #666; }
            </style>
        """)

    demo.queue(default_concurrency_limit=100, max_size=100).launch(
        max_threads=100,
        ssr_mode=False,
        share=args.share,
        inbrowser=args.inbrowser,
        # ssl_certfile="examples/offline_inference/qwen3_omni_moe/cert.pem",
        # ssl_keyfile="examples/offline_inference/qwen3_omni_moe/key.pem",
        # ssl_verify=False,
        server_port=args.server_port,
        server_name=args.server_name,
    )


DEFAULT_CKPT_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only',
                        action='store_true',
                        help='Run demo with CPU only')

    parser.add_argument(
        '--flash-attn2',
        action='store_true',
        default=False,
        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--use-transformers',
                        action='store_true',
                        default=False,
                        help='Use transformers for inference.')
    parser.add_argument(
        '--share',
        action='store_true',
        default=False,
        help='Create a publicly shareable link for the interface.')
    parser.add_argument(
        '--inbrowser',
        action='store_true',
        default=False,
        help=
        'Automatically launch the interface in a new tab on the default browser.'
    )
    parser.add_argument('--server-port',
                        type=int,
                        default=8905,
                        help='Demo server port.')
    parser.add_argument('--server-name',
                        type=str,
                        default='0.0.0.0',
                        help='Demo server name.')
    parser.add_argument('--ui-language',
                        type=str,
                        choices=['en', 'zh'],
                        default='zh',
                        help='Display language for the UI.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)
