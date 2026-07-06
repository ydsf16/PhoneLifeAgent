from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_clip_contexts(location_context_path: Path | None = None, motion_context_path: Path | None = None) -> dict[str, Any]:
    return {
        "location": _read_json(location_context_path),
        "motion": _read_json(motion_context_path),
    }


def context_for_audio(contexts: dict[str, Any], audio_id: str) -> dict[str, Any]:
    return _context_for_media(contexts, "audio_segments", "audio_id", audio_id)


def context_for_video(contexts: dict[str, Any], clip_id: str) -> dict[str, Any]:
    return _context_for_media(contexts, "video_clips", "clip_id", clip_id)


def format_model_context(context: dict[str, Any]) -> str:
    parts = []
    location = context.get("location")
    motion = context.get("motion")
    if location:
        parts.append(
            "\n".join(
                [
                    "Location Context:",
                    f"- time: {location.get('local_time_range')}",
                    f"- quality: {location.get('location_quality')}",
                    f"- movement: {location.get('movement')}",
                    f"- map_image: {location.get('map_image')}",
                    f"- geo_facts: {json.dumps(location.get('geo_facts', {}), ensure_ascii=False)}",
                    f"- route_context: {json.dumps(location.get('route_context', {}), ensure_ascii=False)}",
                    f"- model_note: {location.get('model_context_text')}",
                ]
            )
        )
    if motion:
        parts.append(
            "\n".join(
                [
                    "Motion Context:",
                    f"- time: {motion.get('local_time_range')}",
                    f"- state: {motion.get('motion_state')}",
                    f"- intensity: {motion.get('intensity')}",
                    f"- stability: {motion.get('stability')}",
                    f"- confidence: {motion.get('confidence')}",
                    f"- features: {json.dumps(motion.get('features', {}), ensure_ascii=False)}",
                    f"- model_note: {motion.get('model_context_text')}",
                ]
            )
        )
    if not parts:
        return ""
    return (
        "External Context For This Segment\n"
        "Use this as supporting evidence. If it conflicts with audio/video evidence, state uncertainty instead of forcing a conclusion.\n"
        + "\n\n".join(parts)
    )


def context_metadata(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "location": _trim_context(context.get("location")),
        "motion": _trim_context(context.get("motion")),
    }


def read_context_text_files(paths: list[Path | None]) -> str:
    chunks = []
    for path in paths:
        if path is None:
            continue
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            continue
        chunks.append(f"[Context file: {resolved}]\n{resolved.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(chunks).strip()


def _context_for_media(contexts: dict[str, Any], collection: str, id_key: str, media_id: str) -> dict[str, Any]:
    result = {}
    for source in ["location", "motion"]:
        data = contexts.get(source) or {}
        result[source] = _find_by_id(data.get(collection, []), id_key, media_id)
    return {key: value for key, value in result.items() if value}


def _find_by_id(items: list[dict[str, Any]], id_key: str, media_id: str) -> dict[str, Any] | None:
    target = str(media_id)
    for item in items:
        if str(item.get(id_key)) == target:
            return item
    return None


def _trim_context(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    keep = [
        "local_time_range",
        "location_quality",
        "movement",
        "geo_facts",
        "route_context",
        "map_image",
        "motion_state",
        "intensity",
        "stability",
        "confidence",
        "model_context_text",
    ]
    return {key: value.get(key) for key in keep if key in value}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))
