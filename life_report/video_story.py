from __future__ import annotations

import json
import re
from typing import Any


def build_video_timeline(understandings: list[dict[str, Any]]) -> dict[str, Any]:
    clips = []
    events = []
    keyframes = []
    todos = []
    memories = []

    for item in understandings:
        clip_id = str(item.get("clip_id", ""))
        metadata = item.get("metadata", {})
        understanding = item.get("understanding", {})
        start = _float_or_none(metadata.get("start_utc_sec"))
        end = _float_or_none(metadata.get("end_utc_sec"))
        original_video_path = metadata.get("original_video_path")

        clips.append(
            {
                "clip_id": clip_id,
                "source_path": original_video_path,
                "model_video_path": metadata.get("model_video_path"),
                "report_video_path": metadata.get("report_video_path"),
                "orientation_filter": metadata.get("orientation_filter", ""),
                "start_utc_sec": start,
                "end_utc_sec": end,
                "start_local_time": _format_local_time(start),
                "end_local_time": _format_local_time(end),
                "duration_sec": metadata.get("duration_sec"),
                "clip_summary": understanding.get("clip_summary"),
                "scene": understanding.get("scene"),
                "people": understanding.get("people"),
                "objects": understanding.get("objects"),
                "activities": understanding.get("activities"),
                "emotions": understanding.get("emotions"),
                "life_story_hint": understanding.get("life_story_hint"),
                "confidence": understanding.get("confidence"),
                "processing_duration_sec": item.get("processing_duration_sec"),
            }
        )

        for index, event in enumerate(_as_list(understanding.get("event_candidates")), start=1):
            event = _as_dict(event)
            rel_start, rel_end = _extract_relative_range(event)
            events.append(
                {
                    "event_id": f"video_event_{clip_id}_{index:02d}",
                    "clip_id": clip_id,
                    "time_granularity": "event",
                    "relative_start_sec": rel_start,
                    "relative_end_sec": rel_end,
                    "absolute_start_utc_sec": _absolute_time(start, rel_start),
                    "absolute_end_utc_sec": _absolute_time(start, rel_end),
                    "local_time_range": _format_local_range(_absolute_time(start, rel_start), _absolute_time(start, rel_end)),
                    "event_type": event.get("event_type") or event.get("type") or event.get("label"),
                    "summary": event.get("description") or event.get("summary") or event.get("evidence"),
                    "confidence": event.get("confidence"),
                    "evidence_refs": [_video_evidence_ref(original_video_path, clip_id, rel_start, rel_end, start)],
                }
            )

        frames = _metadata_keyframes(metadata) or _dedupe_keyframe_candidates(_normalize_keyframe_candidates(understanding.get("keyframe_candidates")))
        for index, frame in enumerate(frames, start=1):
            rel_time = _float_or_none(_first_present(frame, ["relative_time_sec", "time_sec", "time"]))
            keyframes.append(
                {
                    "keyframe_id": frame.get("keyframe_id") or f"video_keyframe_{clip_id}_{index:02d}",
                    "clip_id": clip_id,
                    "relative_time_sec": rel_time,
                    "absolute_utc_sec": _float_or_none(frame.get("absolute_utc_sec")) or _absolute_time(start, rel_time),
                    "local_time": frame.get("local_time") or _format_local_time(_absolute_time(start, rel_time)),
                    "caption": frame.get("caption"),
                    "purpose": frame.get("purpose"),
                    "reason": frame.get("reason"),
                    "importance": frame.get("importance"),
                    "keyframe_path": frame.get("keyframe_path"),
                    "source_video_path": original_video_path,
                    "quality_score": frame.get("quality_score"),
                    "accepted": frame.get("accepted", True),
                    "reject_reasons": frame.get("reject_reasons", []),
                    "sharpness": frame.get("sharpness"),
                    "brightness": frame.get("brightness"),
                    "contrast": frame.get("contrast"),
                }
            )

        for todo in _as_list(understanding.get("todo_candidates")):
            todos.append({"clip_id": clip_id, "candidate": todo, "source_path": original_video_path})
        for memory in _as_list(understanding.get("memory_candidates")):
            memories.append({"clip_id": clip_id, "candidate": memory, "source_path": original_video_path})

    clips.sort(key=lambda item: item.get("start_utc_sec") or 0)
    events.sort(key=lambda item: item.get("absolute_start_utc_sec") or 0)
    keyframes.sort(key=lambda item: item.get("absolute_utc_sec") or 0)
    start = clips[0]["start_utc_sec"] if clips else None
    end = clips[-1]["end_utc_sec"] if clips else None

    return {
        "schema_version": "video_timeline.v1",
        "time_range": {
            "start_utc_sec": start,
            "end_utc_sec": end,
            "start_local_time": _format_local_time(start),
            "end_local_time": _format_local_time(end),
        },
        "clips": clips,
        "events": events,
        "keyframes": keyframes,
        "todo_candidates": todos,
        "memory_candidates": memories,
    }


def build_video_story_media_manifest(
    timeline: dict[str, Any],
    max_keyframes: int = 16,
    max_keyframes_per_clip: int = 3,
) -> dict[str, Any]:
    clips_by_id = {str(clip.get("clip_id")): clip for clip in timeline.get("clips", [])}
    selected = _select_story_keyframes(
        timeline.get("keyframes", []),
        clips_by_id,
        max_keyframes=max(0, max_keyframes),
        max_keyframes_per_clip=max(1, max_keyframes_per_clip),
    )
    report_videos = []
    for clip in timeline.get("clips", []):
        clip_id = str(clip.get("clip_id"))
        report_video_path = clip.get("report_video_path")
        if report_video_path:
            report_videos.append(
                {
                    "clip_id": clip_id,
                    "local_time_range": f"{clip.get('start_local_time')} - {clip.get('end_local_time')}",
                    "path": report_video_path,
                    "summary": clip.get("clip_summary"),
                    "orientation_filter": clip.get("orientation_filter", ""),
                }
            )

    return {
        "schema_version": "video_story_media_manifest.v1",
        "purpose": "Selected high-resolution visual assets for the final multimodal Life Story model input.",
        "time_range": timeline.get("time_range", {}),
        "selection_policy": {
            "max_keyframes": max_keyframes,
            "max_keyframes_per_clip": max_keyframes_per_clip,
            "rule": "Keep all keyframes in video_timeline.json; send only accepted quality-filtered fixed-interval keyframes to final story fusion.",
        },
        "selected_keyframes": selected,
        "selected_report_videos": report_videos,
        "all_keyframe_count": len(timeline.get("keyframes", [])),
        "usable_keyframe_count": len([frame for frame in timeline.get("keyframes", []) if _is_usable_keyframe(frame)]),
    }


def build_video_compact_raw(timeline: dict[str, Any], media_manifest: dict[str, Any] | None = None) -> str:
    events_by_clip = _group_by_clip(timeline.get("events", []))
    selected_keyframes = (media_manifest or {}).get("selected_keyframes") or timeline.get("keyframes", [])
    keyframes_by_clip = _group_by_clip(selected_keyframes)
    todo_by_clip = _group_by_clip(timeline.get("todo_candidates", []))
    memory_by_clip = _group_by_clip(timeline.get("memory_candidates", []))
    lines = [
        "PhoneLifeAgent Video Compact Raw",
        f"Time range: {timeline.get('time_range', {}).get('start_local_time')} -> {timeline.get('time_range', {}).get('end_local_time')}",
        "Purpose: compact model input generated from video_understandings.json. Keep clip_id evidence and selected report media paths.",
    ]
    if media_manifest:
        lines.append(
            f"Selected story keyframes: {len(media_manifest.get('selected_keyframes', []))} / {media_manifest.get('all_keyframe_count', 0)}"
        )
    lines.append("")
    for clip in timeline.get("clips", []):
        clip_id = str(clip.get("clip_id"))
        lines.extend(
            [
                f"[Clip {clip_id}] {clip.get('start_local_time')} -> {clip.get('end_local_time')}",
                f"Summary: {_one_line(clip.get('clip_summary'))}",
                f"Scene: {_one_line(clip.get('scene'))}",
                f"Objects: {_one_line(clip.get('objects'))}",
                f"Activities: {_one_line(clip.get('activities'))}",
                f"Report video: {clip.get('report_video_path')}",
            ]
        )
        events = events_by_clip.get(clip_id, [])
        if events:
            lines.append("Events:")
            for event in events:
                lines.append(
                    f"- {event.get('local_time_range')} | {event.get('event_type') or 'event'} | {_one_line(event.get('summary'))} | conf={event.get('confidence')}"
                )
        frames = keyframes_by_clip.get(clip_id, [])[:6]
        if frames:
            lines.append("Selected keyframes:")
            for frame in frames:
                lines.append(
                    f"- {frame.get('local_time')} | score={frame.get('quality_score')} | {_one_line(frame.get('caption') or frame.get('reason'))} | {frame.get('keyframe_path')} | why={_one_line(frame.get('selection_reason'))}"
                )
        todos = todo_by_clip.get(clip_id, [])
        if todos:
            lines.append("TODO candidates:")
            for todo in todos:
                lines.append(f"- {_one_line(todo.get('candidate'))}")
        memories = memory_by_clip.get(clip_id, [])
        if memories:
            lines.append("Memory candidates:")
            for memory in memories:
                lines.append(f"- {_one_line(memory.get('candidate'))}")
        if clip.get("life_story_hint"):
            lines.append(f"Story hint: {_one_line(clip.get('life_story_hint'))}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_video_story_input(
    compact_raw: str,
    story_model: str,
    provider: str,
    summary_thinking: bool | None = None,
    context_text: str = "",
) -> str:
    from .model_engine import create_text_llm
    from .pipeline_defaults import DEFAULT_SUMMARY_THINKING, summary_generation_policy

    resolved_thinking = DEFAULT_SUMMARY_THINKING if summary_thinking is None else summary_thinking
    policy = summary_generation_policy(provider=provider, text_model=story_model, enable_thinking=resolved_thinking)
    text_model = create_text_llm(policy.provider, model=policy.text_model, enable_thinking=policy.enable_thinking)
    user_prompt = compact_raw
    if context_text:
        user_prompt = (
            compact_raw
            + "\n\nGlobal Location/Motion Context For Video Summary\n"
            + "Use this context to enrich place, route, motion, and camera reliability interpretation. If context conflicts with visual/audio evidence, preserve uncertainty.\n"
            + context_text
        )
    return text_model.generate_text(_video_story_input_system_prompt(), user_prompt)


def _video_story_input_system_prompt() -> str:
    return (
        "你是 PhoneLifeAgent 的视频故事素材整理模块。"
        "输入是程序从 video_understandings.json 抽取出的 compact raw notes，包含 report_video 和 report_keyframes 路径。"
        "输入也可能包含全局 Location/Motion Context 和整体路线图路径。"
        "必须使用简体中文输出，输出纯文本，不要 JSON，不要 markdown 表格，不要使用英文小节标题。"
        "目标文件名是 video_story_input.txt，后续会给多模态融合和最终 Life Story 使用。"
        "要求：按时间合并相邻重复片段；保留 clip_id 作为 evidence；保留关键场景、人物、物品、对话、情绪、事件、高光关键帧路径。"
        "叙事素材必须以第一人称视角整理。手机镜头看到的画面，默认写成“我看到/我记录到/镜头里出现”，不要写成英文的 father、daughter、the child。"
        "人物关系不能凭画面编造；如果无法确认关系，写“孩子/成人/家人/旁边的人”，不要写“我的女儿/父亲/母亲”。"
        "只有音视频明确支持本人行为时才写“我做了”；否则写“我看到”“我听到”“画面里有人”。"
        "不要编造地点、人名或关系；不确定就写不确定。"
        "格式请使用这些中文小节：总览、故事线、关键视觉事件、物品和地点、语音亮点、待办候选、记忆候选、开放问题、证据索引。"
    )


def _normalize_keyframe_candidates(value: Any) -> list[dict[str, Any]]:
    normalized = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        if isinstance(item, dict):
            t = _first_present(item, ["relative_time_sec", "time_sec", "time"])
            normalized.append(
                {
                    "relative_time_sec": _float_or_none(t),
                    "caption": item.get("caption") or item.get("description") or item.get("reason"),
                    "purpose": item.get("purpose") or item.get("type"),
                    "importance": item.get("importance"),
                    "reason": item.get("reason") or item.get("description"),
                    "keyframe_path": item.get("keyframe_path"),
                }
            )
        else:
            normalized.append({"relative_time_sec": _float_or_none(item), "caption": "", "purpose": "model_candidate"})
    return [item for item in normalized if item.get("relative_time_sec") is not None]


def _dedupe_keyframe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_time: dict[float, dict[str, Any]] = {}
    for candidate in candidates:
        time_value = _float_or_none(candidate.get("relative_time_sec"))
        if time_value is None:
            continue
        key = round(time_value, 1)
        existing = by_time.get(key)
        if existing is None:
            by_time[key] = candidate
            continue
        if not existing.get("caption") and candidate.get("caption"):
            by_time[key] = candidate
    return [by_time[key] for key in sorted(by_time)]


def _first_present(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


normalize_keyframe_candidates = _normalize_keyframe_candidates
dedupe_keyframe_candidates = _dedupe_keyframe_candidates
first_present = _first_present


def _select_story_keyframes(
    keyframes: list[dict[str, Any]],
    clips_by_id: dict[str, dict[str, Any]],
    max_keyframes: int,
    max_keyframes_per_clip: int,
) -> list[dict[str, Any]]:
    if max_keyframes <= 0:
        return []
    eligible = [frame for frame in keyframes if _is_usable_keyframe(frame)]
    grouped = _group_by_clip(eligible)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_clip_counts: dict[str, int] = {}

    def add_frame(frame: dict[str, Any]) -> None:
        if len(selected) >= max_keyframes:
            return
        clip_id = str(frame.get("clip_id"))
        if per_clip_counts.get(clip_id, 0) >= max_keyframes_per_clip:
            return
        frame_id = str(frame.get("keyframe_id") or f"{clip_id}:{frame.get('relative_time_sec')}:{frame.get('keyframe_path')}")
        if frame_id in selected_ids:
            return
        selected_ids.add(frame_id)
        per_clip_counts[clip_id] = per_clip_counts.get(clip_id, 0) + 1
        selected.append(_story_keyframe_record(frame, clips_by_id.get(clip_id), len(selected) + 1))

    for clip_id in _distributed_clip_ids(clips_by_id, max_keyframes):
        frames = sorted(grouped.get(clip_id, []), key=lambda item: (-_keyframe_story_score(item, clips_by_id.get(clip_id)), item.get("absolute_utc_sec") or 0))
        if frames:
            add_frame(frames[0])

    remaining = sorted(
        eligible,
        key=lambda item: (-_keyframe_story_score(item, clips_by_id.get(str(item.get("clip_id")))), item.get("absolute_utc_sec") or 0),
    )
    for frame in remaining:
        add_frame(frame)

    selected.sort(key=lambda item: (item.get("absolute_utc_sec") is None, item.get("absolute_utc_sec") or 0))
    for index, frame in enumerate(selected, start=1):
        frame["selection_rank"] = index
    return selected


def _distributed_clip_ids(clips_by_id: dict[str, dict[str, Any]], limit: int) -> list[str]:
    clip_ids = list(clips_by_id.keys())
    if limit <= 0 or len(clip_ids) <= limit:
        return clip_ids
    if limit == 1:
        return [clip_ids[len(clip_ids) // 2]]
    step = (len(clip_ids) - 1) / (limit - 1)
    indexes = []
    seen = set()
    for i in range(limit):
        index = round(i * step)
        if index not in seen:
            indexes.append(index)
            seen.add(index)
    return [clip_ids[index] for index in indexes]


def _story_keyframe_record(frame: dict[str, Any], clip: dict[str, Any] | None, rank: int) -> dict[str, Any]:
    clip = clip or {}
    return {
        "selection_rank": rank,
        "keyframe_id": frame.get("keyframe_id"),
        "clip_id": frame.get("clip_id"),
        "local_time": frame.get("local_time"),
        "relative_time_sec": frame.get("relative_time_sec"),
        "absolute_utc_sec": frame.get("absolute_utc_sec"),
        "keyframe_path": frame.get("keyframe_path"),
        "caption": frame.get("caption"),
        "reason": frame.get("reason"),
        "purpose": frame.get("purpose"),
        "importance": frame.get("importance"),
        "quality_score": frame.get("quality_score"),
        "accepted": frame.get("accepted", True),
        "reject_reasons": frame.get("reject_reasons", []),
        "sharpness": frame.get("sharpness"),
        "brightness": frame.get("brightness"),
        "contrast": frame.get("contrast"),
        "selection_reason": _keyframe_selection_reason(frame, clip),
        "report_video_path": clip.get("report_video_path"),
        "source_video_path": frame.get("source_video_path") or clip.get("source_path"),
        "clip_summary": clip.get("clip_summary"),
    }


def _keyframe_story_score(frame: dict[str, Any], clip: dict[str, Any] | None) -> float:
    score = 0.0
    score += _quality_score(frame.get("quality_score"))
    score += _importance_score(frame.get("importance"))
    purpose = str(frame.get("purpose") or "").lower()
    reason = _one_line(frame.get("reason") or frame.get("caption"), limit=120).lower()
    if any(word in purpose for word in ["highlight", "story", "event", "moment"]):
        score += 2.0
    if any(word in reason for word in ["高光", "重点", "事件", "情绪", "人物", "物品", "对话", "标识", "交通", "家庭", "会议"]):
        score += 1.0
    if frame.get("caption"):
        score += 0.6
    if frame.get("reason"):
        score += 0.6
    if frame.get("local_time"):
        score += 0.2
    if clip and clip.get("life_story_hint"):
        score += 0.5
    return score


def _importance_score(value: Any) -> float:
    numeric = _float_or_none(value)
    if numeric is not None:
        return max(0.0, min(3.0, numeric * 3 if numeric <= 1 else numeric))
    text = str(value or "").lower()
    if text in {"high", "important", "critical", "高", "重要"}:
        return 3.0
    if text in {"medium", "mid", "中"}:
        return 2.0
    if text in {"low", "低"}:
        return 1.0
    return 0.0


def _keyframe_selection_reason(frame: dict[str, Any], clip: dict[str, Any]) -> str:
    parts = []
    if frame.get("quality_score") is not None:
        parts.append(f"quality={frame.get('quality_score')}")
    if frame.get("importance") is not None:
        parts.append(f"importance={frame.get('importance')}")
    if frame.get("purpose"):
        parts.append(f"purpose={frame.get('purpose')}")
    if frame.get("reason"):
        parts.append(_one_line(frame.get("reason"), limit=80))
    elif frame.get("caption"):
        parts.append(_one_line(frame.get("caption"), limit=80))
    if clip.get("life_story_hint"):
        parts.append("supports clip story hint")
    return "; ".join(parts) or "representative frame for this clip"


def _metadata_keyframes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    frames = metadata.get("report_keyframes")
    return [frame for frame in frames if isinstance(frame, dict)] if isinstance(frames, list) else []


def _is_usable_keyframe(frame: dict[str, Any]) -> bool:
    return bool(frame.get("keyframe_path")) and frame.get("accepted", True) is not False


def _quality_score(value: Any) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        return 0.0
    return max(0.0, min(4.0, numeric * 4.0))


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


def _video_evidence_ref(
    source_path: str | None,
    clip_id: str,
    rel_start: float | None,
    rel_end: float | None,
    clip_start: float | None,
) -> dict[str, Any]:
    return {
        "type": "video",
        "clip_id": clip_id,
        "path": source_path,
        "relative_start_sec": rel_start,
        "relative_end_sec": rel_end,
        "absolute_start_utc_sec": _absolute_time(clip_start, rel_start),
        "absolute_end_utc_sec": _absolute_time(clip_start, rel_end),
    }


def _absolute_time(clip_start: float | None, offset: float | None) -> float | None:
    if clip_start is None or offset is None:
        return None
    return clip_start + offset


def _format_local_range(start: float | None, end: float | None) -> str:
    return f"{_format_local_time(start)} - {_format_local_time(end)}"


def _format_local_time(utc_sec: float | None) -> str:
    if utc_sec is None:
        return "-"
    from datetime import datetime

    return datetime.fromtimestamp(utc_sec).strftime("%Y-%m-%d %H:%M:%S")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"description": str(value)}


def _group_by_clip(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        clip_id = str(item.get("clip_id", ""))
        grouped.setdefault(clip_id, []).append(item)
    return grouped


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


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
