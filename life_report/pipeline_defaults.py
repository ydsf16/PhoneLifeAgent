from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PROVIDER = "aliyun"
DEFAULT_MOCK_PROVIDER = "mock"
DEFAULT_SUMMARY_PROVIDER = "aliyun"
DEFAULT_FINAL_PROVIDER = "aliyun"

DEFAULT_AUDIO_MODEL = "qwen3.5-omni-plus"
DEFAULT_VIDEO_MODEL = "qwen3.5-omni-plus"
DEFAULT_SUMMARY_TEXT_MODEL = "qwen3.7-plus"
DEFAULT_STORY_TEXT_MODEL = "qwen3.7-plus"
DEFAULT_COMIC_TEXT_MODEL = "qwen3.7-plus"
DEFAULT_HIGHLIGHT_TEXT_MODEL = "qwen3.7-plus"
DEFAULT_COMIC_IMAGE_MODEL = "doubao-seedream-4-5-251128"

DEFAULT_AUDIO_CONCURRENCY = 3
DEFAULT_VIDEO_CONCURRENCY = 3
DEFAULT_MAX_STORY_KEYFRAMES = 16
DEFAULT_COMIC_MAX_PANELS = 9
DEFAULT_COMIC_MAX_REFERENCES_PER_PANEL = 1
DEFAULT_HIGHLIGHT_TARGET_SECONDS = 45
DEFAULT_HIGHLIGHT_MAX_SEGMENTS = 8

DEFAULT_SUMMARY_THINKING = False
DEFAULT_STORY_THINKING = True
DEFAULT_COMIC_THINKING = True
DEFAULT_HIGHLIGHT_THINKING = True


@dataclass(frozen=True)
class GenerationPolicy:
    provider: str
    text_model: str
    enable_thinking: bool
    target_seconds: int | None = None
    max_segments: int | None = None
    max_panels: int | None = None
    max_references_per_panel: int | None = None
    max_story_keyframes: int | None = None
    audio_concurrency: int | None = None
    video_concurrency: int | None = None


def summary_generation_policy(
    provider: str = DEFAULT_SUMMARY_PROVIDER,
    text_model: str = DEFAULT_SUMMARY_TEXT_MODEL,
    enable_thinking: bool = DEFAULT_SUMMARY_THINKING,
) -> GenerationPolicy:
    return GenerationPolicy(
        provider=provider,
        text_model=text_model,
        enable_thinking=enable_thinking,
        max_story_keyframes=DEFAULT_MAX_STORY_KEYFRAMES,
        audio_concurrency=DEFAULT_AUDIO_CONCURRENCY,
        video_concurrency=DEFAULT_VIDEO_CONCURRENCY,
    )


def story_generation_policy(
    provider: str = DEFAULT_FINAL_PROVIDER,
    text_model: str = DEFAULT_STORY_TEXT_MODEL,
    max_story_keyframes: int = DEFAULT_MAX_STORY_KEYFRAMES,
    enable_thinking: bool = DEFAULT_STORY_THINKING,
) -> GenerationPolicy:
    return GenerationPolicy(
        provider=provider,
        text_model=text_model,
        enable_thinking=enable_thinking,
        max_story_keyframes=max_story_keyframes,
    )


def comic_generation_policy(
    provider: str = DEFAULT_FINAL_PROVIDER,
    text_model: str = DEFAULT_COMIC_TEXT_MODEL,
    max_panels: int = DEFAULT_COMIC_MAX_PANELS,
    max_references_per_panel: int = DEFAULT_COMIC_MAX_REFERENCES_PER_PANEL,
    enable_thinking: bool = DEFAULT_COMIC_THINKING,
) -> GenerationPolicy:
    return GenerationPolicy(
        provider=provider,
        text_model=text_model,
        enable_thinking=enable_thinking,
        max_panels=max_panels,
        max_references_per_panel=max_references_per_panel,
    )


def highlight_generation_policy(
    provider: str = DEFAULT_FINAL_PROVIDER,
    text_model: str = DEFAULT_HIGHLIGHT_TEXT_MODEL,
    target_seconds: int = DEFAULT_HIGHLIGHT_TARGET_SECONDS,
    max_segments: int = DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
    enable_thinking: bool = DEFAULT_HIGHLIGHT_THINKING,
) -> GenerationPolicy:
    return GenerationPolicy(
        provider=provider,
        text_model=text_model,
        enable_thinking=enable_thinking,
        target_seconds=target_seconds,
        max_segments=max_segments,
    )
