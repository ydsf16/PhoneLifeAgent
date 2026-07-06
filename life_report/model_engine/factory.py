from __future__ import annotations

from pathlib import Path

from .aliyun import AliyunSpeechModel, AliyunTextLLM, AliyunVideoModel
from .base import SpeechModel, TextLLM, VideoModel
from .mock import MockSpeechModel, MockTextLLM, MockVideoModel


def create_speech_model(
    provider: str,
    model: str | None = None,
    cache_dir: Path | None = None,
    max_audio_seconds: int | None = None,
) -> SpeechModel:
    if provider == "mock":
        return MockSpeechModel(model_name=model or "mock-audio")
    if provider == "aliyun":
        return AliyunSpeechModel(
            model_name=model or "qwen3.5-omni-plus",
            cache_dir=cache_dir,
            max_audio_seconds=max_audio_seconds,
        )
    raise ValueError(f"Unsupported speech model provider: {provider}")


def create_text_llm(provider: str, model: str | None = None, enable_thinking: bool = False) -> TextLLM:
    if provider in {"mock", "none"}:
        return MockTextLLM(model_name=model or "mock-text")
    if provider == "aliyun":
        return AliyunTextLLM(model_name=model or "qwen3.7-plus", enable_thinking=enable_thinking)
    raise ValueError(f"Unsupported text model provider: {provider}")


def create_video_model(provider: str, model: str | None = None) -> VideoModel:
    if provider in {"mock", "none"}:
        return MockVideoModel(model_name=model or "mock-video")
    if provider == "aliyun":
        return AliyunVideoModel(model_name=model or "qwen3.5-omni-plus")
    raise ValueError(f"Unsupported video model provider: {provider}")
