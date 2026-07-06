from __future__ import annotations

from .base import AudioSegmentInput, AudioSummary


class MockSpeechModel:
    provider_name = "mock"

    def __init__(self, model_name: str = "mock-audio") -> None:
        self.model_name = model_name

    def summarize_audio(self, audio: AudioSegmentInput) -> AudioSummary:
        return AudioSummary(
            audio_id=audio.audio_id,
            provider=self.provider_name,
            model=self.model_name,
            source_path=str(audio.file_path),
            text="",
            environment_summary="模拟音频摘要：这一段可能包含环境声和少量人声。",
            speech_summary="模拟转写：当前未调用真实语音模型。",
            confidence=None,
            understanding={
                "transcript": "",
                "scene_summary": "模拟音频理解：这一段可能包含环境声和少量人声。",
                "important_moments": [],
                "people_or_speakers": [],
                "environment": [],
                "event_candidates": [],
                "todo_candidates": [],
                "memory_candidates": [],
                "emotion_or_tone": "",
                "life_story_hint": "我经历了一段普通的日常声音片段。",
                "evidence_notes": [],
            },
            metadata={
                "start_utc_sec": audio.start_utc_sec,
                "end_utc_sec": audio.end_utc_sec,
                "duration_sec": audio.duration_sec,
                "context": audio.context_metadata,
                "context_text": audio.context_text,
            },
        )


class MockTextLLM:
    provider_name = "mock"

    def __init__(self, model_name: str = "mock-text") -> None:
        self.model_name = model_name

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return f"Model: {self.model_name}\nSource: local fallback text model\n\n{user_prompt}"


class MockVideoModel:
    provider_name = "mock"

    def __init__(self, model_name: str = "mock-video") -> None:
        self.model_name = model_name

    def understand_video(self, video, prompt: str) -> dict:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "clip_id": video.clip_id,
            "source_path": str(video.file_path),
            "understanding": {
                "clip_summary": "模拟视频理解结果。",
                "scene": {"place_type": "unknown"},
                "objects": [],
                "highlight_moments": [{"relative_time_sec": 1.0, "caption": "模拟高光", "importance": 0.5}],
                "keyframe_candidates": [{"relative_time_sec": 1.0, "purpose": "mock", "caption": "模拟关键帧"}],
                "context_used": bool(getattr(video, "context_text", None)),
            },
            "metadata": {"context": getattr(video, "context_metadata", {}), "context_text": getattr(video, "context_text", None)},
        }
