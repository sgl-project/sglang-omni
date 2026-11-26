# SPDX-License-Identifier: Apache-2.0
"""What the baseline measures: the record types, and how it asks and scores each lane row."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

from tests.test_ci.xpu_model.model_table import XPU_MODELS, XpuModel

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "tests" / "data"
REFERENCE_AUDIO = DATA_DIR / "query_to_cars.wav"
REFERENCE_TEXT = "How many cars are there in the picture?"
SPOKEN_TEXT = "The quick brown fox jumps over the lazy dog."
ARITHMETIC_PROMPT = "Answer with the number only: 17 plus 25?"
ARITHMETIC_ANSWER = "42"
MUSIC_PROMPT = "A short calm piano melody."

DEFAULT_BASELINE = REPO_ROOT / "scripts" / "xpu" / "xpu_model_baseline.json"

ProbeKind = Literal["transcription", "speech", "music", "chat"]


class Reference(TypedDict):
    audio_path: str
    text: str


class RequestFields(TypedDict, total=False):
    voice: str
    ref_audio: str
    ref_text: str
    task_type: str
    language: str
    instructions: str
    references: list[Reference]
    stage_params: dict[str, dict[str, int]]
    max_new_tokens: int


class Sample(TypedDict, total=False):
    exact: dict[str, str | float]
    atmost: dict[str, float]
    close: dict[str, float]
    speed: dict[str, float]
    observed: dict[str, float]
    heard: str


class CardMemory(TypedDict):
    before: dict[str, float]
    after: dict[str, float]


class Record(TypedDict, total=False):
    exact: dict[str, str | float]
    atmost: dict[str, float]
    close: dict[str, float]
    speed: dict[str, float]
    spread: dict[str, float]
    trimmed_spread: dict[str, float]
    observed: dict[str, float]
    observed_span: dict[str, float]
    cards: str
    judge_cards: str
    judge_serve_extra: list[str]
    cards_used_mib: CardMemory
    launches: int
    warmups: int
    within_run_spread: dict[str, float]


class Tolerances(TypedDict):
    close: float
    absolute: float
    speed: float
    speed_headroom: float
    speed_ceiling: float


class Environment(TypedDict, total=False):
    fingerprint: dict[str, str]
    recorded_from_commit: str
    tolerances: Tolerances
    serve_extra: dict[str, list[str]]
    models: dict[str, Record]


class BaselineFile(TypedDict):
    environments: dict[str, Environment]


@dataclass(frozen=True, kw_only=True)
class BaselineModel:
    """One lane row as the baseline measures it: the row carries the recipe, this adds the request and scoring."""

    row: XpuModel
    name: str
    probe: ProbeKind
    payload: RequestFields = field(default_factory=RequestFields)
    # Another model key, served alongside, that transcribes this model's audio back to text.
    judge: str = ""
    # The server refuses a seed, so the generated length is observed rather than gated.
    seedless: bool = False
    # The key the committed baseline records this row under, when it is not the row's own.
    recorded_as: str = ""

    @property
    def key(self) -> str:
        return self.recorded_as or self.row.key

    @property
    def model_path(self) -> str:
        return self.row.checkpoint_spec

    @property
    def cards(self) -> int:
        return self.row.cards

    def serve_args_on(self, family: str) -> tuple[str, ...]:
        return self.row.serve_args_on(family)

    @property
    def env(self) -> dict[str, str]:
        return dict(self.row.env)

    @property
    def startup_seconds(self) -> int:
        return self.row.startup_seconds

    @property
    def companions(self) -> tuple[str, ...]:
        return self.row.companions

    @property
    def unservable(self) -> str:
        return self.row.serving

    @property
    def config_variant(self) -> str:
        return self.row.variant or ""


LANE_ROWS = {row.key: row for row in XPU_MODELS}
CLONED_VOICE = RequestFields(
    voice="default", ref_audio=str(REFERENCE_AUDIO), ref_text=REFERENCE_TEXT
)
REFERENCE_LIST = RequestFields(
    voice="default",
    references=[Reference(audio_path=str(REFERENCE_AUDIO), text=REFERENCE_TEXT)],
)
AUK_REQUEST = RequestFields(
    voice="default",
    instructions="A warm, relaxed voice.",
    stage_params={"auk_engine": {"gen_seconds": 4}},
)


def speech(
    key: str, name: str, payload: RequestFields, **options: bool
) -> BaselineModel:
    """A speech row, judged by Qwen3-ASR transcribing its audio back."""
    return BaselineModel(
        row=LANE_ROWS[key],
        name=name,
        probe="speech",
        payload=payload,
        judge="qwen3-asr",
        **options,
    )


MODELS: tuple[BaselineModel, ...] = (
    BaselineModel(row=LANE_ROWS["qwen3-asr"], name="Qwen3-ASR", probe="transcription"),
    BaselineModel(
        row=LANE_ROWS["whisper"], name="Whisper large-v3", probe="transcription"
    ),
    BaselineModel(row=LANE_ROWS["fun-asr"], name="Fun-ASR-Nano", probe="transcription"),
    speech("qwen3-tts", "Qwen3-TTS Base", CLONED_VOICE),
    speech(
        "qwen3-tts-cv",
        "Qwen3-TTS CustomVoice",
        RequestFields(voice="Ryan", task_type="CustomVoice", language="English"),
    ),
    speech("cosyvoice3", "Fun-CosyVoice3", CLONED_VOICE),
    speech("zonos2", "ZONOS2", RequestFields(voice="default"), seedless=True),
    speech("dots-tts", "dots.tts SOAR", REFERENCE_LIST),
    BaselineModel(
        row=LANE_ROWS["minimax-music3"],
        name="MiniMax Music 3",
        probe="music",
        # An explicit token cap: the default length makes the run unbounded and not comparable.
        payload=RequestFields(
            instructions=(
                "An energetic arena rock anthem with distorted electric guitars, punchy live "
                "drums and a soaring male vocal at 130 BPM, wide stereo image"
            ),
            max_new_tokens=750,
        ),
    ),
    speech("ming-tts", "Ming-Omni-TTS", RequestFields(voice="default"), seedless=True),
    BaselineModel(
        row=LANE_ROWS["omni-text"],
        name="Qwen3-Omni thinker (text only)",
        probe="chat",
        recorded_as="qwen3-omni-text",
    ),
    BaselineModel(
        row=LANE_ROWS["moss-transcribe-diarize"],
        name="MOSS-Transcribe-Diarize",
        probe="transcription",
    ),
    BaselineModel(
        row=LANE_ROWS["nemotron35-asr"], name="Nemotron 3.5 ASR", probe="transcription"
    ),
    BaselineModel(row=LANE_ROWS["arkasr"], name="ARK-ASR", probe="transcription"),
    speech("higgs", "Higgs Audio v3 TTS", RequestFields(voice="default")),
    speech("auk", "AuK", AUK_REQUEST),
    speech("auk-flash", "AuK-Flash", AUK_REQUEST),
    speech(
        "qwen3-tts-vd",
        "Qwen3-TTS VoiceDesign",
        RequestFields(
            voice="default",
            task_type="VoiceDesign",
            instructions="A warm, natural young adult voice.",
        ),
    ),
    speech("dots-tts-base", "dots.tts Base", REFERENCE_LIST),
    speech("dots-tts-mf", "dots.tts MF", REFERENCE_LIST),
    BaselineModel(
        row=LANE_ROWS["ming-flash-omni"], name="Ming-flash-omni-2.0", probe="chat"
    ),
    BaselineModel(row=LANE_ROWS["minicpm-o"], name="MiniCPM-o 4.5", probe="chat"),
    BaselineModel(row=LANE_ROWS["personaplex"], name="PersonaPlex", probe="speech"),
    BaselineModel(
        row=LANE_ROWS["nemotron-voicechat"], name="Nemotron VoiceChat", probe="speech"
    ),
    BaselineModel(
        row=LANE_ROWS["audar-tts"], name="Audar-TTS V1 Turbo", probe="speech"
    ),
)
MODEL_BY_KEY = {model.key: model for model in MODELS}
