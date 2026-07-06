from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class AudioSegmentInput:
    audio_id: str
    file_path: Path
    session_path: Path
    start_utc_sec: float | None = None
    end_utc_sec: float | None = None
    duration_sec: float | None = None
    context_text: str | None = None
    context_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["file_path"] = str(self.file_path)
        data["session_path"] = str(self.session_path)
        return data


@dataclass(frozen=True)
class AudioSummary:
    audio_id: str
    provider: str
    model: str
    source_path: str
    text: str
    environment_summary: str
    speech_summary: str
    confidence: float | None = None
    understanding: dict[str, Any] = field(default_factory=dict)
    raw_cache_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpeechModel(Protocol):
    provider_name: str
    model_name: str

    def summarize_audio(self, audio: AudioSegmentInput) -> AudioSummary:
        ...


class TextLLM(Protocol):
    provider_name: str
    model_name: str

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(frozen=True)
class VideoClipInput:
    clip_id: str
    file_path: Path
    start_utc_sec: float | None = None
    end_utc_sec: float | None = None
    duration_sec: float | None = None
    fps: float | None = None
    context_text: str | None = None
    context_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["file_path"] = str(self.file_path)
        return data


class VideoModel(Protocol):
    provider_name: str
    model_name: str

    def understand_video(self, video: VideoClipInput, prompt: str) -> dict[str, Any]:
        ...
