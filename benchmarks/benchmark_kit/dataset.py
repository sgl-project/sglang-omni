from dataclasses import dataclass


@dataclass
class ImageDataset:
    img_path: str
    prompt: str
    target_response: str


@dataclass
class TextDataset:
    prompt: str
    target_response: str


@dataclass
class AudioDataset:
    audio_path: str
    prompt: str
    target_response: str
    target_audio_path: str


@dataclass
class VideoDataset:
    video_path: str
    prompt: str
    target_response: str
    target_video_path: str
