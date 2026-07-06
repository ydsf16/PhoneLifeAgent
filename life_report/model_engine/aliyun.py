from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .base import AudioSegmentInput, AudioSummary, VideoClipInput


DEFAULT_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OMNI_MODEL = "qwen3.5-omni-plus"
DEFAULT_TEXT_MODEL = "qwen3.7-plus"


def _openai_client(api_key: str, base_url: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Missing dependency: install with pip install -e ".[aliyun]"') from exc
    timeout = float(os.environ.get("DASHSCOPE_TIMEOUT_SEC", "300"))
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


class AliyunSpeechModel:
    provider_name = "aliyun_omni"

    def __init__(
        self,
        model_name: str = DEFAULT_OMNI_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: Path | None = None,
        max_audio_seconds: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("DASHSCOPE_OPENAI_BASE_URL")
            or os.environ.get("ALIYUN_OPENAI_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        )
        self.cache_dir = cache_dir
        self.max_audio_seconds = max_audio_seconds

    def summarize_audio(self, audio: AudioSegmentInput) -> AudioSummary:
        if not self.api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY. Export it locally before using --provider aliyun.")
        if not audio.file_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio.file_path}")

        request_audio_path, request_audio_format = self._request_audio_path(audio)
        audio_data = base64.b64encode(request_audio_path.read_bytes()).decode("utf-8")
        client = _openai_client(self.api_key, self.base_url)

        prompt = (
            "你是 PhoneLifeAgent 的音频理解模块。"
            "输入是一段 iPhone LifeLogger 连续录制的 5 分钟左右音频。"
            "你的目标不是简单转写，而是把这段音频转成后续 Life Story、Event Timeline、TODO 和 Memory 可用的结构化证据。"
            "请关注：人声中的关键信息、活动变化、地点/场景线索、明确或隐含的待办、值得长期记住的偏好/关系/事实。"
            "请只输出 JSON，不要输出 markdown。字段必须包括："
            "transcript、scene_summary、important_moments、people_or_speakers、environment、"
            "event_candidates、todo_candidates、memory_candidates、emotion_or_tone、"
            "life_story_hint、confidence、evidence_notes。"
            "event_candidates/todo_candidates/memory_candidates 使用数组；如果没有就输出空数组。"
            "life_story_hint 用第一人称，适合作为我这一天故事的一段素材。"
            "如果提供了 Location/Motion Context，请把它作为辅助证据，用于判断地点、移动状态和音频可靠性；如果和音频内容冲突，请明确写不确定。"
        )
        if audio.context_text:
            prompt += "\n\n" + audio.context_text
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/{request_audio_format};base64,{audio_data}",
                                "format": request_audio_format,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            modalities=["text"],
        )
        raw = response.model_dump(mode="json")
        raw_cache_path = self._write_raw_cache(audio.audio_id, raw)
        text = response.choices[0].message.content or ""
        parsed = _parse_json_text(text)

        return AudioSummary(
            audio_id=audio.audio_id,
            provider=self.provider_name,
            model=self.model_name,
            source_path=str(audio.file_path),
            text=text,
            environment_summary=str(parsed.get("scene_summary") or parsed.get("environment_summary") or text),
            speech_summary=str(parsed.get("speech_summary") or parsed.get("transcript") or ""),
            confidence=_float_or_none(parsed.get("confidence")),
            understanding=parsed,
            raw_cache_path=str(raw_cache_path) if raw_cache_path else None,
            metadata={
                "transcript": parsed.get("transcript"),
                "scene_summary": parsed.get("scene_summary"),
                "important_moments": parsed.get("important_moments"),
                "people_or_speakers": parsed.get("people_or_speakers"),
                "activity_guess": parsed.get("activity_guess"),
                "environment": parsed.get("environment"),
                "event_candidates": parsed.get("event_candidates"),
                "todo_candidates": parsed.get("todo_candidates"),
                "memory_candidates": parsed.get("memory_candidates"),
                "emotion_or_tone": parsed.get("emotion_or_tone"),
                "life_story_hint": parsed.get("life_story_hint"),
                "evidence_notes": parsed.get("evidence_notes"),
                "start_utc_sec": audio.start_utc_sec,
                "end_utc_sec": audio.end_utc_sec,
                "duration_sec": audio.duration_sec,
                "original_audio_path": str(audio.file_path),
                "request_audio_path": str(request_audio_path),
                "request_audio_format": request_audio_format,
                "max_audio_seconds": self.max_audio_seconds,
                "request_base_url": self.base_url,
                "context": audio.context_metadata,
                "context_text": audio.context_text,
            },
        )

    def _request_audio_path(self, audio: AudioSegmentInput) -> tuple[Path, str]:
        if self.cache_dir is None:
            return audio.file_path, _audio_format(audio.file_path)
        converted_dir = self.cache_dir / "converted_audio"
        converted_dir.mkdir(parents=True, exist_ok=True)
        seconds_label = self.max_audio_seconds if self.max_audio_seconds else "full"
        suffix = f"{seconds_label}s" if isinstance(seconds_label, int) else seconds_label
        converted_path = converted_dir / f"audio_{audio.audio_id}_{suffix}.mp3"
        if converted_path.exists():
            return converted_path, "mp3"

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to prepare local m4a files for Qwen-Omni probing.")

        command = [ffmpeg, "-y", "-i", str(audio.file_path)]
        if self.max_audio_seconds:
            command.extend(["-t", str(self.max_audio_seconds)])
        command.extend(["-vn", "-codec:a", "libmp3lame", "-b:a", "64k", str(converted_path)])
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr = " ".join((result.stderr or "").split())
            raise RuntimeError(f"ffmpeg audio conversion failed with exit {result.returncode}: {' '.join(command)} | {stderr[:1000]}")
        return converted_path, "mp3"

    def _write_raw_cache(self, audio_id: str, raw: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"audio_{audio_id}_raw_response.json"
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"mp3", "wav"}:
        return suffix
    if suffix in {"m4a", "aac"}:
        return "aac"
    return suffix or "mp3"


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class AliyunTextLLM:
    provider_name = "aliyun_openai_compatible"

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.enable_thinking = enable_thinking
        self.base_url = (
            base_url
            or os.environ.get("DASHSCOPE_OPENAI_BASE_URL")
            or os.environ.get("ALIYUN_OPENAI_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        )

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY for Aliyun text model generation.")
        client = _openai_client(self.api_key, self.base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"enable_thinking": self.enable_thinking},
        )
        return (response.choices[0].message.content or "").strip() + "\n"


class AliyunVideoModel:
    provider_name = "aliyun_omni"

    def __init__(
        self,
        model_name: str = DEFAULT_OMNI_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("DASHSCOPE_OPENAI_BASE_URL")
            or os.environ.get("ALIYUN_OPENAI_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        )

    def understand_video(self, video: VideoClipInput, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY for Aliyun video model generation.")
        if not video.file_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video.file_path}")
        video_data = base64.b64encode(video.file_path.read_bytes()).decode("utf-8")
        client = _openai_client(self.api_key, self.base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_data}",
                                "fps": video.fps or 2,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            modalities=["text"],
        )
        text = response.choices[0].message.content or ""
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "clip_id": video.clip_id,
            "source_path": str(video.file_path),
            "text": text,
            "understanding": _parse_json_text(text),
            "metadata": {
                "start_utc_sec": video.start_utc_sec,
                "end_utc_sec": video.end_utc_sec,
                "duration_sec": video.duration_sec,
                "fps": video.fps,
            },
        }
