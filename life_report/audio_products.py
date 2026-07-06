from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .context_injection import read_context_text_files
from .model_engine import create_text_llm
from .pipeline_defaults import DEFAULT_SUMMARY_TEXT_MODEL, DEFAULT_SUMMARY_THINKING, summary_generation_policy

DEFAULT_TEXT_MODEL = DEFAULT_SUMMARY_TEXT_MODEL


def build_audio_products(
    understandings_path: Path,
    output_dir: Path,
    story_model: str = DEFAULT_TEXT_MODEL,
    provider: str = "aliyun",
    summary_thinking: bool | None = None,
    location_compact_path: Path | None = None,
    motion_compact_path: Path | None = None,
) -> dict[str, Any]:
    understandings = json.loads(understandings_path.expanduser().resolve().read_text(encoding="utf-8"))
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    timeline = build_audio_timeline(understandings)
    timeline_path = output / "audio_timeline.json"
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    compact_raw = build_audio_compact_raw(timeline)
    compact_raw_path = output / "audio_compact_raw.txt"
    compact_raw_path.write_text(compact_raw, encoding="utf-8")

    context_text = read_context_text_files([location_compact_path, motion_compact_path])
    story_input = build_audio_story_input(
        compact_raw,
        story_model=story_model,
        provider=provider,
        summary_thinking=summary_thinking,
        context_text=context_text,
    )
    story_input_path = output / "audio_story_input.txt"
    story_input_path.write_text(story_input, encoding="utf-8")

    return {
        "audio_timeline_path": str(timeline_path),
        "audio_compact_raw_path": str(compact_raw_path),
        "audio_story_input_path": str(story_input_path),
        "audio_event_count": len(timeline["events"]),
        "audio_moment_count": len(timeline["moments"]),
        "story_model": story_model,
        "context_injected": bool(context_text),
    }


def build_audio_timeline(understandings: list[dict[str, Any]]) -> dict[str, Any]:
    segments = []
    events = []
    moments = []
    todos = []
    memories = []

    for item in understandings:
        audio_id = str(item.get("audio_id", ""))
        metadata = item.get("metadata", {})
        understanding = item.get("understanding", {})
        segment_start = _float_or_none(metadata.get("start_utc_sec"))
        segment_end = _float_or_none(metadata.get("end_utc_sec"))
        source_path = item.get("source_path") or metadata.get("original_audio_path")

        segments.append(
            {
                "audio_id": audio_id,
                "source_path": source_path,
                "start_utc_sec": segment_start,
                "end_utc_sec": segment_end,
                "start_local_time": _format_local_time(segment_start),
                "end_local_time": _format_local_time(segment_end),
                "duration_sec": metadata.get("duration_sec"),
                "scene_summary": understanding.get("scene_summary"),
                "life_story_hint": understanding.get("life_story_hint"),
                "confidence": understanding.get("confidence") or item.get("confidence"),
                "processing_duration_sec": item.get("processing_duration_sec"),
            }
        )

        for index, event in enumerate(_as_list(understanding.get("event_candidates")), start=1):
            event = _as_dict(event)
            rel_start, rel_end = _extract_relative_range(event)
            events.append(
                {
                    "event_id": f"audio_event_{audio_id}_{index:02d}",
                    "audio_id": audio_id,
                    "time_granularity": "event",
                    "relative_start_sec": rel_start,
                    "relative_end_sec": rel_end,
                    "absolute_start_utc_sec": _absolute_time(segment_start, rel_start),
                    "absolute_end_utc_sec": _absolute_time(segment_start, rel_end),
                    "local_time_range": _format_local_range(
                        _absolute_time(segment_start, rel_start),
                        _absolute_time(segment_start, rel_end),
                    ),
                    "event_type": event.get("event_type") or event.get("label") or event.get("type"),
                    "summary": event.get("description") or event.get("summary") or event.get("evidence"),
                    "confidence": event.get("confidence"),
                    "evidence": event.get("evidence"),
                    "evidence_refs": [
                        _evidence_ref(source_path, audio_id, rel_start, rel_end, segment_start)
                    ],
                }
            )

        for index, moment in enumerate(_as_list(understanding.get("important_moments")), start=1):
            moment = _as_dict(moment)
            rel_start, rel_end = _extract_relative_range(moment)
            moments.append(
                {
                    "moment_id": f"audio_moment_{audio_id}_{index:02d}",
                    "audio_id": audio_id,
                    "time_granularity": "moment",
                    "relative_start_sec": rel_start,
                    "relative_end_sec": rel_end,
                    "absolute_start_utc_sec": _absolute_time(segment_start, rel_start),
                    "absolute_end_utc_sec": _absolute_time(segment_start, rel_end),
                    "local_time_range": _format_local_range(
                        _absolute_time(segment_start, rel_start),
                        _absolute_time(segment_start, rel_end),
                    ),
                    "label": moment.get("label"),
                    "description": moment.get("description") or moment.get("text"),
                    "evidence_refs": [
                        _evidence_ref(source_path, audio_id, rel_start, rel_end, segment_start)
                    ],
                }
            )

        for transcript in _as_list(understanding.get("transcript")):
            if isinstance(transcript, dict):
                rel_start, rel_end = _extract_relative_range(transcript)
                moments.append(
                    {
                        "moment_id": f"audio_transcript_{audio_id}_{len(moments) + 1:02d}",
                        "audio_id": audio_id,
                        "time_granularity": "moment",
                        "relative_start_sec": rel_start,
                        "relative_end_sec": rel_end,
                        "absolute_start_utc_sec": _absolute_time(segment_start, rel_start),
                        "absolute_end_utc_sec": _absolute_time(segment_start, rel_end),
                        "local_time_range": _format_local_range(
                            _absolute_time(segment_start, rel_start),
                            _absolute_time(segment_start, rel_end),
                        ),
                        "transcript": transcript.get("text"),
                        "evidence_refs": [
                            _evidence_ref(source_path, audio_id, rel_start, rel_end, segment_start)
                        ],
                    }
                )

        for todo in _as_list(understanding.get("todo_candidates")):
            todos.append({"audio_id": audio_id, "candidate": todo, "source_path": source_path})
        for memory in _as_list(understanding.get("memory_candidates")):
            memories.append({"audio_id": audio_id, "candidate": memory, "source_path": source_path})

    segments.sort(key=lambda item: item.get("start_utc_sec") or 0)
    events.sort(key=lambda item: item.get("absolute_start_utc_sec") or 0)
    moments.sort(key=lambda item: item.get("absolute_start_utc_sec") or 0)
    start = segments[0]["start_utc_sec"] if segments else None
    end = segments[-1]["end_utc_sec"] if segments else None

    return {
        "schema_version": "audio_timeline.v1",
        "time_range": {
            "start_utc_sec": start,
            "end_utc_sec": end,
            "start_local_time": _format_local_time(start),
            "end_local_time": _format_local_time(end),
        },
        "segments": segments,
        "events": events,
        "moments": moments,
        "todo_candidates": todos,
        "memory_candidates": memories,
    }


def build_audio_compact_raw(timeline: dict[str, Any]) -> str:
    events_by_audio = _group_by_audio(timeline.get("events", []))
    moments_by_audio = _group_by_audio(timeline.get("moments", []))
    todo_by_audio = _group_by_audio(timeline.get("todo_candidates", []))
    memory_by_audio = _group_by_audio(timeline.get("memory_candidates", []))
    lines = [
        "PhoneLifeAgent Audio Compact Raw",
        f"Time range: {timeline.get('time_range', {}).get('start_local_time')} -> {timeline.get('time_range', {}).get('end_local_time')}",
        "Purpose: compact model input generated from audio_understandings.json. Keep audio_id evidence.",
        "",
    ]
    for segment in timeline.get("segments", []):
        audio_id = segment.get("audio_id")
        lines.extend(
            [
                f"[Audio {audio_id}] {segment.get('start_local_time')} -> {segment.get('end_local_time')}",
                f"Scene: {_one_line(segment.get('scene_summary'))}",
            ]
        )
        events = events_by_audio.get(str(audio_id), [])
        if events:
            lines.append("Events:")
            for event in events:
                lines.append(
                    f"- {event.get('local_time_range')} | {event.get('event_type') or 'event'} | {_one_line(event.get('summary'))} | conf={event.get('confidence')}"
                )
        key_moments = _select_key_moments(moments_by_audio.get(str(audio_id), []))
        if key_moments:
            lines.append("Moments:")
            for moment in key_moments:
                text = moment.get("transcript") or moment.get("description")
                lines.append(f"- {moment.get('local_time_range')} | {_one_line(text)}")
        todos = todo_by_audio.get(str(audio_id), [])
        if todos:
            lines.append("TODO candidates:")
            for todo in todos:
                lines.append(f"- {_one_line(todo.get('candidate'))}")
        memories = memory_by_audio.get(str(audio_id), [])
        if memories:
            lines.append("Memory candidates:")
            for memory in memories:
                lines.append(f"- {_one_line(memory.get('candidate'))}")
        if segment.get("life_story_hint"):
            lines.append(f"Story hint: {_one_line(segment.get('life_story_hint'))}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_audio_story_input(
    compact_raw: str,
    story_model: str,
    provider: str,
    summary_thinking: bool | None = None,
    context_text: str = "",
) -> str:
    resolved_thinking = DEFAULT_SUMMARY_THINKING if summary_thinking is None else summary_thinking
    policy = summary_generation_policy(provider=provider, text_model=story_model, enable_thinking=resolved_thinking)
    text_model = create_text_llm(policy.provider, model=policy.text_model, enable_thinking=policy.enable_thinking)
    user_prompt = compact_raw
    if context_text:
        user_prompt = (
            compact_raw
            + "\n\nGlobal Location/Motion Context For Audio Summary\n"
            + "Use this context to enrich place and movement interpretation. If context conflicts with audio evidence, preserve uncertainty.\n"
            + context_text
        )
    return text_model.generate_text(_story_input_system_prompt(), user_prompt)


def _story_input_system_prompt() -> str:
    return (
        "你是 PhoneLifeAgent 的音频故事素材整理模块。"
        "输入是程序从 audio_understandings.json 抽取出的 compact raw notes。"
        "输入也可能包含全局 Location/Motion Context。"
        "必须使用简体中文输出，输出纯文本，不要 JSON，不要 markdown 表格，不要使用英文小节标题。"
        "目标文件名是 audio_story_input.txt，后续会给多模态融合和最终 Life Story 使用。"
        "要求：按时间合并相邻重复片段；保留 audio_id 作为 evidence；保留关键事件、关键语音、明确 TODO、高置信 Memory、开放问题。"
        "叙事素材必须用第一人称整理。不要写“用户”“父亲”“拍摄者”；可以写“我听到”“我记录到”“家里有人说”“孩子说”。"
        "只有明确是本人说话或行动时才写“我说/我做”；不确定时写“家里有人/旁边有人/声音里听到”。"
        "不要编造地点、人名或结论；不确定就写不确定。"
        "格式请使用这些中文小节：总览、故事线、关键事件、语音亮点、待办候选、记忆候选、开放问题、证据索引。"
    )


def _extract_relative_range(value: dict[str, Any]) -> tuple[float | None, float | None]:
    for start_key, end_key in [("start", "end"), ("start_time", "end_time")]:
        if start_key in value or end_key in value:
            return _parse_time_offset(value.get(start_key)), _parse_time_offset(value.get(end_key))
    for key in ["time_range", "time", "timestamp"]:
        parsed = _parse_time_range(value.get(key))
        if parsed != (None, None):
            return parsed
    return None, None


def _parse_time_range(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    text = str(value)
    if "-" in text:
        start, end = text.split("-", 1)
        return _parse_time_offset(start), _parse_time_offset(end)
    point = _parse_time_offset(text)
    return point, point


def _parse_time_offset(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)", text)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _evidence_ref(
    source_path: str | None,
    audio_id: str,
    rel_start: float | None,
    rel_end: float | None,
    segment_start: float | None,
) -> dict[str, Any]:
    return {
        "type": "audio",
        "audio_id": audio_id,
        "path": source_path,
        "relative_start_sec": rel_start,
        "relative_end_sec": rel_end,
        "absolute_start_utc_sec": _absolute_time(segment_start, rel_start),
        "absolute_end_utc_sec": _absolute_time(segment_start, rel_end),
    }


def _absolute_time(segment_start: float | None, offset: float | None) -> float | None:
    if segment_start is None or offset is None:
        return None
    return segment_start + offset


def _format_local_range(start: float | None, end: float | None) -> str:
    return f"{_format_local_time(start)} - {_format_local_time(end)}"


def _format_local_time(utc_sec: float | None) -> str:
    if utc_sec is None:
        return "-"
    return datetime.fromtimestamp(utc_sec).strftime("%Y-%m-%d %H:%M:%S")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"description": str(value)}


def _group_by_audio(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        audio_id = str(item.get("audio_id", ""))
        grouped.setdefault(audio_id, []).append(item)
    return grouped


def _select_key_moments(moments: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    with_text = [item for item in moments if item.get("transcript") or item.get("description")]
    return with_text[:limit]


def _one_line(value: Any, limit: int = 260) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
