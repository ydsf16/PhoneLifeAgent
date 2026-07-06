from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .session_loader import read_csv_rows


MAX_CONTEXT_NEAREST_SAMPLE_GAP_SEC = 60.0


def build_motion_products(session_path: Path, output_dir: Path, window_sec: float = 10.0) -> dict[str, Any]:
    session = session_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = read_motion_samples(session)
    windows = build_motion_windows(samples, window_sec=window_sec)
    timeline = build_motion_timeline(windows)
    context = build_clip_motion_context(session, samples, timeline)
    compact_raw = build_motion_compact_raw(timeline, context)

    features_path = output / "motion_features.json"
    timeline_path = output / "motion_timeline.json"
    context_path = output / "clip_motion_context.json"
    compact_path = output / "motion_compact_raw.txt"
    features_path.write_text(json.dumps({"schema_version": "motion_features.v1", "source": "motion/device_motion.csv", "samples": samples, "windows": windows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact_path.write_text(compact_raw, encoding="utf-8")
    return {
        "motion_features_path": str(features_path),
        "motion_timeline_path": str(timeline_path),
        "clip_motion_context_path": str(context_path),
        "motion_compact_raw_path": str(compact_path),
        "sample_count": len(samples),
        "window_count": len(windows),
        "segment_count": len(timeline["segments"]),
        "video_context_count": len(context["video_clips"]),
        "audio_context_count": len(context["audio_segments"]),
    }


def read_motion_samples(session_path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(session_path / "motion" / "device_motion.csv")
    samples = []
    for index, row in enumerate(rows, start=1):
        utc = _float_or_none(row.get("utc_sec"))
        if utc is None:
            continue
        ax = _float_or_none(row.get("user_acc_x")) or 0.0
        ay = _float_or_none(row.get("user_acc_y")) or 0.0
        az = _float_or_none(row.get("user_acc_z")) or 0.0
        rx = _float_or_none(row.get("rot_x")) or 0.0
        ry = _float_or_none(row.get("rot_y")) or 0.0
        rz = _float_or_none(row.get("rot_z")) or 0.0
        samples.append(
            {
                "sample_id": f"motion_{index:06d}",
                "utc_sec": utc,
                "local_time": _format_local_time(utc),
                "user_acc": {"x": ax, "y": ay, "z": az},
                "rotation_rate": {"x": rx, "y": ry, "z": rz},
                "attitude": {
                    "roll": _float_or_none(row.get("roll")),
                    "pitch": _float_or_none(row.get("pitch")),
                    "yaw": _float_or_none(row.get("yaw")),
                },
                "accel_mag": math.sqrt(ax * ax + ay * ay + az * az),
                "gyro_mag": math.sqrt(rx * rx + ry * ry + rz * rz),
            }
        )
    return sorted(samples, key=lambda item: item["utc_sec"])


def build_motion_windows(samples: list[dict[str, Any]], window_sec: float = 10.0) -> list[dict[str, Any]]:
    if not samples:
        return []
    windows = []
    start = samples[0]["utc_sec"]
    end = samples[-1]["utc_sec"]
    index = 1
    cursor = start
    while cursor <= end:
        chunk = [sample for sample in samples if cursor <= sample["utc_sec"] < cursor + window_sec]
        if chunk:
            features = _motion_features(chunk)
            state, intensity, stability, confidence = classify_motion(features)
            windows.append(
                {
                    "window_id": f"motion_win_{index:04d}",
                    "start_utc_sec": chunk[0]["utc_sec"],
                    "end_utc_sec": chunk[-1]["utc_sec"],
                    "start_local_time": _format_local_time(chunk[0]["utc_sec"]),
                    "end_local_time": _format_local_time(chunk[-1]["utc_sec"]),
                    "duration_sec": round(max(0.0, chunk[-1]["utc_sec"] - chunk[0]["utc_sec"]), 3),
                    "sample_count": len(chunk),
                    "state": state,
                    "intensity": intensity,
                    "stability": stability,
                    "features": features,
                    "confidence": confidence,
                }
            )
            index += 1
        cursor += window_sec
    return windows


def build_motion_timeline(windows: list[dict[str, Any]]) -> dict[str, Any]:
    segments = []
    if windows:
        current = [windows[0]]
        for window in windows[1:]:
            previous = current[-1]
            gap = window["start_utc_sec"] - previous["end_utc_sec"]
            if gap <= 15 and window["state"] == previous["state"] and window["intensity"] == previous["intensity"]:
                current.append(window)
            else:
                segments.append(_make_motion_segment(len(segments) + 1, current))
                current = [window]
        segments.append(_make_motion_segment(len(segments) + 1, current))
    start = windows[0]["start_utc_sec"] if windows else None
    end = windows[-1]["end_utc_sec"] if windows else None
    return {
        "schema_version": "motion_timeline.v1",
        "time_range": {
            "start_utc_sec": start,
            "end_utc_sec": end,
            "start_local_time": _format_local_time(start),
            "end_local_time": _format_local_time(end),
        },
        "segments": segments,
        "summary": {
            "window_count": len(windows),
            "segment_count": len(segments),
            "states": dict(Counter(window["state"] for window in windows)),
        },
    }


def build_clip_motion_context(session_path: Path, samples: list[dict[str, Any]], timeline: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "clip_motion_context.v1",
        "video_clips": [
            _motion_context_for_media(row, "clip_id", samples, timeline)
            for row in read_csv_rows(session_path / "video" / "clip_index.csv")
        ],
        "audio_segments": [
            _motion_context_for_media(row, "audio_id", samples, timeline)
            for row in read_csv_rows(session_path / "audio" / "audio_index.csv")
        ],
    }


def build_motion_compact_raw(timeline: dict[str, Any], context: dict[str, Any]) -> str:
    lines = [
        "PhoneLifeAgent Motion Compact Raw",
        f"Time range: {timeline.get('time_range', {}).get('start_local_time')} -> {timeline.get('time_range', {}).get('end_local_time')}",
        f"Summary: {json.dumps(timeline.get('summary', {}), ensure_ascii=False)}",
        "",
    ]
    for segment in timeline.get("segments", []):
        lines.extend(
            [
                f"[{segment['segment_id']}] {segment['start_local_time']} -> {segment['end_local_time']}",
                f"State: {segment['state']} | intensity={segment['intensity']} | stability={segment['stability']} | confidence={segment['confidence']}",
                f"Features: {json.dumps(segment['features'], ensure_ascii=False)}",
                "",
            ]
        )
    lines.append("Video Clip Context:")
    for item in context.get("video_clips", []):
        lines.append(f"- clip {item.get('clip_id')} | {item.get('local_time_range')} | {item.get('model_context_text')}")
    lines.append("")
    lines.append("Audio Segment Context:")
    for item in context.get("audio_segments", []):
        lines.append(f"- audio {item.get('audio_id')} | {item.get('local_time_range')} | {item.get('model_context_text')}")
    return "\n".join(lines).strip() + "\n"


def classify_motion(features: dict[str, float]) -> tuple[str, str, str, float]:
    accel = features.get("accel_rms", 0.0)
    gyro = features.get("gyro_rms", 0.0)
    jerk = features.get("jerk_rms", 0.0)
    if accel < 0.06 and gyro < 0.08:
        return "stationary", "low", "stable", 0.85
    if accel < 0.18 and gyro < 0.35:
        return "steady_motion", "low", "stable", 0.72
    if accel < 0.55 and gyro < 1.6:
        return "walking_like", "moderate", "shaky", 0.7
    if accel < 1.1 and jerk < 1.2:
        return "phone_handling", "moderate", "shaky", 0.62
    return "running_or_shaking", "high", "very_shaky", 0.68


def _motion_context_for_media(row: dict[str, str], id_key: str, samples: list[dict[str, Any]], timeline: dict[str, Any]) -> dict[str, Any]:
    start = _float_or_none(row.get("start_utc_sec"))
    end = _float_or_none(row.get("end_utc_sec"))
    media_id = str(row.get(id_key) or "")
    matched_samples = [sample for sample in samples if start is not None and end is not None and start <= sample["utc_sec"] <= end]
    nearest_sample_gap_sec = None
    context_status = "matched"
    if not matched_samples and start is not None and end is not None:
        mid = (start + end) / 2
        nearest = sorted(samples, key=lambda sample: abs(sample["utc_sec"] - mid))[:1]
        if nearest:
            nearest_sample_gap_sec = round(abs(nearest[0]["utc_sec"] - mid), 3)
            if nearest_sample_gap_sec <= MAX_CONTEXT_NEAREST_SAMPLE_GAP_SEC:
                matched_samples = nearest
                context_status = "nearest"
            else:
                context_status = "stale"
    elif not matched_samples:
        context_status = "missing"
    matched_segments = _overlap_segments(timeline, start, end)
    features = _motion_features(matched_samples) if matched_samples else {}
    state, intensity, stability, confidence = classify_motion(features) if features else ("unknown", "unknown", "unknown", 0.0)
    if matched_segments:
        state = _majority([segment["state"] for segment in matched_segments]) or state
        intensity = _majority([segment["intensity"] for segment in matched_segments]) or intensity
        stability = _majority([segment["stability"] for segment in matched_segments]) or stability
        confidence = round(sum(segment["confidence"] for segment in matched_segments) / len(matched_segments), 3)
    return {
        id_key: media_id,
        "start_utc_sec": start,
        "end_utc_sec": end,
        "local_time_range": f"{_format_local_time(start)} - {_format_local_time(end)}",
        "motion_state": state,
        "intensity": intensity,
        "stability": stability,
        "context_status": context_status,
        "nearest_sample_gap_sec": nearest_sample_gap_sec,
        "matched_segments": [segment["segment_id"] for segment in matched_segments],
        "sample_count": len(matched_samples),
        "features": features,
        "confidence": confidence,
        "model_context_text": _motion_model_context_text(state, intensity, stability, confidence, context_status),
    }


def _make_motion_segment(index: int, windows: list[dict[str, Any]]) -> dict[str, Any]:
    features = _aggregate_features([window["features"] for window in windows])
    state = _majority([window["state"] for window in windows]) or "unknown"
    intensity = _majority([window["intensity"] for window in windows]) or "unknown"
    stability = _majority([window["stability"] for window in windows]) or "unknown"
    confidence = round(sum(window["confidence"] for window in windows) / len(windows), 3)
    return {
        "segment_id": f"motion_seg_{index:04d}",
        "start_utc_sec": windows[0]["start_utc_sec"],
        "end_utc_sec": windows[-1]["end_utc_sec"],
        "start_local_time": _format_local_time(windows[0]["start_utc_sec"]),
        "end_local_time": _format_local_time(windows[-1]["end_utc_sec"]),
        "duration_sec": round(max(0.0, windows[-1]["end_utc_sec"] - windows[0]["start_utc_sec"]), 3),
        "state": state,
        "intensity": intensity,
        "stability": stability,
        "window_count": len(windows),
        "features": features,
        "confidence": confidence,
    }


def _motion_features(samples: list[dict[str, Any]]) -> dict[str, float]:
    accel = [sample["accel_mag"] for sample in samples]
    gyro = [sample["gyro_mag"] for sample in samples]
    jerk = [abs(b - a) for a, b in zip(accel[:-1], accel[1:])]
    return {
        "accel_rms": round(_rms(accel), 4),
        "accel_peak": round(max(accel) if accel else 0.0, 4),
        "gyro_rms": round(_rms(gyro), 4),
        "gyro_peak": round(max(gyro) if gyro else 0.0, 4),
        "jerk_rms": round(_rms(jerk), 4),
    }


def _aggregate_features(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: round(sum(item.get(key, 0.0) for item in items) / len(items), 4) for key in keys}


def _motion_model_context_text(state: str, intensity: str, stability: str, confidence: float, context_status: str = "matched") -> str:
    if context_status in {"stale", "missing"}:
        return "运动上下文：没有可靠的同时间段 DeviceMotion 数据，请不要用运动状态辅助判断。"
    if state == "stationary":
        meaning = "手机基本静止，可把音视频理解为固定地点或稳定拍摄。"
    elif state == "steady_motion":
        meaning = "手机有平稳运动，可能是缓慢移动或乘坐交通工具。"
    elif state == "walking_like":
        meaning = "运动模式接近步行，音频中的脚步声或视频晃动可作为行走证据。"
    elif state == "running_or_shaking":
        meaning = "运动较剧烈，视频视觉判断置信度应降低。"
    elif state == "phone_handling":
        meaning = "可能存在手持操作或短暂晃动。"
    else:
        meaning = "运动状态不确定。"
    return f"运动上下文：state={state}，intensity={intensity}，stability={stability}，confidence={confidence}；{meaning}"


def _overlap_segments(timeline: dict[str, Any], start: float | None, end: float | None) -> list[dict[str, Any]]:
    if start is None or end is None:
        return []
    return [segment for segment in timeline.get("segments", []) if segment["end_utc_sec"] >= start and segment["start_utc_sec"] <= end]


def _majority(values: list[Any]) -> Any:
    clean = [value for value in values if value is not None]
    return Counter(clean).most_common(1)[0][0] if clean else None


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _format_local_time(utc_sec: float | None) -> str:
    if utc_sec is None:
        return "-"
    return datetime.fromtimestamp(utc_sec).strftime("%Y-%m-%d %H:%M:%S")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
