from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProductPaths:
    audio_products_dir: Path | None = None
    video_products_dir: Path | None = None
    location_products_dir: Path | None = None
    motion_products_dir: Path | None = None


class ContextStore:
    def __init__(self, paths: ProductPaths) -> None:
        self.paths = paths
        self.audio_timeline = _read_json(_product_file(paths.audio_products_dir, "audio_timeline.json"))
        self.video_timeline = _read_json(_product_file(paths.video_products_dir, "video_timeline.json"))
        self.location_timeline = _read_json(_product_file(paths.location_products_dir, "location_timeline.json"))
        self.motion_timeline = _read_json(_product_file(paths.motion_products_dir, "motion_timeline.json"))
        self.video_media_manifest = _read_json(_product_file(paths.video_products_dir, "video_story_media_manifest.json"))
        self.clip_location_context = _read_json(_product_file(paths.location_products_dir, "clip_location_context.json"))
        self.clip_motion_context = _read_json(_product_file(paths.motion_products_dir, "clip_motion_context.json"))
        self.audio_story_input = _read_text(_product_file(paths.audio_products_dir, "audio_story_input.txt"))
        self.video_story_input = _read_text(_product_file(paths.video_products_dir, "video_story_input.txt"))
        self.location_compact_raw = _read_text(_product_file(paths.location_products_dir, "location_compact_raw.txt"))
        self.motion_compact_raw = _read_text(_product_file(paths.motion_products_dir, "motion_compact_raw.txt"))

    def query_context(self, start_utc_sec: float | None, end_utc_sec: float | None) -> dict[str, Any]:
        return {
            "audio": {
                "segments": _overlap_items(self.audio_timeline.get("segments", []), start_utc_sec, end_utc_sec),
                "events": _overlap_items(self.audio_timeline.get("events", []), start_utc_sec, end_utc_sec),
                "moments": _overlap_items(self.audio_timeline.get("moments", []), start_utc_sec, end_utc_sec),
            },
            "video": {
                "clips": _overlap_items(self.video_timeline.get("clips", []), start_utc_sec, end_utc_sec),
                "events": _overlap_items(self.video_timeline.get("events", []), start_utc_sec, end_utc_sec),
                "keyframes": _point_items(self.video_timeline.get("keyframes", []), start_utc_sec, end_utc_sec),
            },
            "location": self.get_location_context(start_utc_sec, end_utc_sec),
            "motion": self.get_motion_context(start_utc_sec, end_utc_sec),
        }

    def get_clip_evidence(self, clip_id: str) -> dict[str, Any]:
        clip_id = str(clip_id)
        clip = _first_by_id(self.video_timeline.get("clips", []), "clip_id", clip_id)
        return {
            "clip": clip,
            "events": [item for item in self.video_timeline.get("events", []) if str(item.get("clip_id")) == clip_id],
            "keyframes": [item for item in self.video_timeline.get("keyframes", []) if str(item.get("clip_id")) == clip_id],
            "selected_keyframes": [
                item for item in self.video_media_manifest.get("selected_keyframes", []) if str(item.get("clip_id")) == clip_id
            ],
            "location": _first_by_id(self.clip_location_context.get("video_clips", []), "clip_id", clip_id),
            "motion": _first_by_id(self.clip_motion_context.get("video_clips", []), "clip_id", clip_id),
        }

    def get_audio_evidence(self, audio_id: str) -> dict[str, Any]:
        audio_id = str(audio_id)
        segment = _first_by_id(self.audio_timeline.get("segments", []), "audio_id", audio_id)
        return {
            "segment": segment,
            "events": [item for item in self.audio_timeline.get("events", []) if str(item.get("audio_id")) == audio_id],
            "moments": [item for item in self.audio_timeline.get("moments", []) if str(item.get("audio_id")) == audio_id],
            "location": _first_by_id(self.clip_location_context.get("audio_segments", []), "audio_id", audio_id),
            "motion": _first_by_id(self.clip_motion_context.get("audio_segments", []), "audio_id", audio_id),
        }

    def get_location_context(self, start_utc_sec: float | None, end_utc_sec: float | None) -> dict[str, Any]:
        return {
            "segments": _overlap_items(self.location_timeline.get("segments", []), start_utc_sec, end_utc_sec),
            "overall_map_image": self.location_timeline.get("overall_map_image"),
            "summary": self.location_timeline.get("summary", {}),
        }

    def get_motion_context(self, start_utc_sec: float | None, end_utc_sec: float | None) -> dict[str, Any]:
        return {
            "segments": _overlap_items(self.motion_timeline.get("segments", []), start_utc_sec, end_utc_sec),
            "summary": self.motion_timeline.get("summary", {}),
        }

    def get_keyframes(self, clip_id: str | None = None, max_count: int = 16) -> list[dict[str, Any]]:
        frames = self.video_media_manifest.get("selected_keyframes") or self.video_timeline.get("keyframes", [])
        if clip_id is not None:
            frames = [item for item in frames if str(item.get("clip_id")) == str(clip_id)]
        return list(frames)[: max(0, max_count)]

    def search_events(self, keyword: str) -> list[dict[str, Any]]:
        keyword = keyword.lower()
        events = []
        for source, items in [("audio", self.audio_timeline.get("events", [])), ("video", self.video_timeline.get("events", []))]:
            for item in items:
                text = json.dumps(item, ensure_ascii=False).lower()
                if keyword in text:
                    record = dict(item)
                    record["source"] = source
                    events.append(record)
        return events

    def global_time_range(self) -> dict[str, Any]:
        ranges = [
            self.audio_timeline.get("time_range", {}),
            self.video_timeline.get("time_range", {}),
            self.location_timeline.get("time_range", {}),
            self.motion_timeline.get("time_range", {}),
        ]
        starts = [_float_or_none(item.get("start_utc_sec")) for item in ranges]
        ends = [_float_or_none(item.get("end_utc_sec")) for item in ranges]
        starts = [item for item in starts if item is not None]
        ends = [item for item in ends if item is not None]
        return {
            "start_utc_sec": min(starts) if starts else None,
            "end_utc_sec": max(ends) if ends else None,
        }


def _product_file(directory: Path | None, name: str) -> Path | None:
    if directory is None:
        return None
    return directory.expanduser().resolve() / name


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _first_by_id(items: list[dict[str, Any]], key: str, value: str) -> dict[str, Any] | None:
    for item in items:
        if str(item.get(key)) == value:
            return item
    return None


def _overlap_items(items: list[dict[str, Any]], start: float | None, end: float | None) -> list[dict[str, Any]]:
    if start is None or end is None:
        return list(items)
    matched = []
    for item in items:
        item_start, item_end = _item_time_range(item)
        if item_start is None and item_end is None:
            continue
        item_start = item_start if item_start is not None else item_end
        item_end = item_end if item_end is not None else item_start
        if item_start is not None and item_end is not None and item_end >= start and item_start <= end:
            matched.append(item)
    return matched


def _point_items(items: list[dict[str, Any]], start: float | None, end: float | None) -> list[dict[str, Any]]:
    if start is None or end is None:
        return list(items)
    matched = []
    for item in items:
        point = _float_or_none(item.get("absolute_utc_sec"))
        if point is not None and start <= point <= end:
            matched.append(item)
    return matched


def _item_time_range(item: dict[str, Any]) -> tuple[float | None, float | None]:
    start = _float_or_none(item.get("start_utc_sec") or item.get("absolute_start_utc_sec"))
    end = _float_or_none(item.get("end_utc_sec") or item.get("absolute_end_utc_sec"))
    if start is None and end is None:
        point = _float_or_none(item.get("absolute_utc_sec"))
        return point, point
    return start, end


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
