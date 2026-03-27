from dataclasses import dataclass


@dataclass
class Data:
    prompt_text: str | None = None
    prompt_audio_path: str | None = None
    prompt_audio_path: str | None = None
    target_text: str | None = None
    target_audio_path: str | None = None
    target_video_path: str | None = None
    target_img_path: str | None = None
