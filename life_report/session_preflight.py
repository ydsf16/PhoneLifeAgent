from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .session_loader import read_csv_rows


SCHEMA_VERSION = "session_preflight.v1"
MIN_MEDIA_BYTES = 1024
MAX_MEDIA_TIME_DRIFT_SEC = 10.0
MAX_REASONABLE_MEDIA_DURATION_SEC = 6 * 60 * 60


def build_session_preflight(session_path: Path, output_dir: Path) -> dict[str, Any]:
    session = session_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    audio = _inspect_media_index(session, "audio", "audio/audio_index.csv", "audio_id", stream_type="audio")
    video = _inspect_media_index(session, "video", "video/clip_index.csv", "clip_id", stream_type="video")
    location = _inspect_location(session)
    motion = _inspect_motion(session)

    quarantined = audio["quarantined"] + video["quarantined"]
    warnings = [*audio["warnings"], *video["warnings"], *location["warnings"], *motion["warnings"]]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "session_path": str(session),
        "summary": {
            "media_rows": audio["total"] + video["total"],
            "valid_media_rows": audio["valid"] + video["valid"],
            "quarantined_media_rows": len(quarantined),
            "audio_rows": audio["total"],
            "valid_audio_rows": audio["valid"],
            "video_rows": video["total"],
            "valid_video_rows": video["valid"],
            "location_rows": location["total"],
            "location_abnormal_rows": location["abnormal"],
            "motion_rows": motion["total"],
            "motion_abnormal_rows": motion["abnormal"],
        },
        "quarantined": quarantined,
        "quarantine_keys": {
            "audio_ids": [item["item_id"] for item in quarantined if item.get("stream") == "audio"],
            "audio_paths": [item["relative_path"] for item in quarantined if item.get("stream") == "audio" and item.get("relative_path")],
            "video_clip_ids": [item["item_id"] for item in quarantined if item.get("stream") == "video"],
            "video_paths": [item["relative_path"] for item in quarantined if item.get("stream") == "video" and item.get("relative_path")],
        },
        "quality_reports": {
            "audio": _strip_quarantined(audio),
            "video": _strip_quarantined(video),
            "location": location,
            "motion": motion,
        },
        "warnings": warnings,
    }

    manifest_path = output / "quarantine_manifest.json"
    health_path = output / "session_health.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    health_path.write_text(json.dumps(_health_summary(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "quarantine_manifest_path": str(manifest_path),
        "session_health_path": str(health_path),
        "quarantined_media_rows": len(quarantined),
        "valid_audio_rows": audio["valid"],
        "valid_video_rows": video["valid"],
        "location_abnormal_rows": location["abnormal"],
        "motion_abnormal_rows": motion["abnormal"],
        "warnings": warnings,
    }


def load_quarantine_manifest(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    manifest_path = path.expanduser().resolve()
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def is_quarantined_audio(row: dict[str, Any], manifest: dict[str, Any]) -> bool:
    keys = manifest.get("quarantine_keys") or {}
    audio_ids = set(str(value) for value in keys.get("audio_ids") or [])
    audio_paths = set(str(value) for value in keys.get("audio_paths") or [])
    rel_path = str(row.get("file_path") or "")
    audio_id = str(row.get("audio_id") or Path(rel_path).stem)
    return audio_id in audio_ids or rel_path in audio_paths


def is_quarantined_video(row: dict[str, Any], manifest: dict[str, Any]) -> bool:
    keys = manifest.get("quarantine_keys") or {}
    clip_ids = set(str(value) for value in keys.get("video_clip_ids") or [])
    video_paths = set(str(value) for value in keys.get("video_paths") or [])
    rel_path = str(row.get("file_path") or "")
    clip_id = str(row.get("clip_id") or Path(rel_path).stem)
    return clip_id in clip_ids or rel_path in video_paths


def _inspect_media_index(session: Path, stream: str, index_rel: str, id_key: str, stream_type: str) -> dict[str, Any]:
    rows = read_csv_rows(session / index_rel)
    quarantined = []
    warnings = []
    ffprobe_available = bool(shutil.which("ffprobe"))
    if not ffprobe_available and rows:
        warnings.append(f"ffprobe unavailable; {stream} media readability checks were limited")

    for row_number, row in enumerate(rows, start=2):
        item_id = str(row.get(id_key) or Path(str(row.get("file_path") or "")).stem or f"row_{row_number}")
        rel_path = str(row.get("file_path") or "")
        reasons = _media_row_reasons(session, row, rel_path)
        probe = _ffprobe_media(session / rel_path, stream_type) if ffprobe_available and rel_path else {}
        if probe.get("error"):
            reasons.append(probe["error"])
        elif probe:
            reasons.extend(_media_probe_reasons(row, probe))
        if reasons:
            quarantined.append(
                {
                    "stream": stream,
                    "item_id": item_id,
                    "row_number": row_number,
                    "relative_path": rel_path,
                    "reasons": reasons,
                    "ffprobe": {key: probe.get(key) for key in ("duration_sec", "codec_type", "codec_name", "width", "height") if key in probe},
                }
            )

    return {
        "stream": stream,
        "index": index_rel,
        "total": len(rows),
        "valid": len(rows) - len(quarantined),
        "quarantined": quarantined,
        "warnings": warnings,
        "ffprobe_available": ffprobe_available,
    }


def _media_row_reasons(session: Path, row: dict[str, Any], rel_path: str) -> list[str]:
    reasons = []
    start = _float_or_none(row.get("start_utc_sec"))
    end = _float_or_none(row.get("end_utc_sec"))
    duration = _float_or_none(row.get("duration_sec"))
    if not rel_path:
        reasons.append("missing_file_path")
    else:
        path = session / rel_path
        if not path.exists():
            reasons.append("file_missing")
        elif not path.is_file():
            reasons.append("not_a_file")
        elif path.stat().st_size < MIN_MEDIA_BYTES:
            reasons.append("file_too_small")
    if start is None or end is None:
        reasons.append("missing_utc_range")
    elif end <= start:
        reasons.append("invalid_utc_range")
    if duration is None:
        reasons.append("missing_duration")
    elif duration <= 0:
        reasons.append("invalid_duration")
    elif duration > MAX_REASONABLE_MEDIA_DURATION_SEC:
        reasons.append("duration_too_long")
    return reasons


def _media_probe_reasons(row: dict[str, Any], probe: dict[str, Any]) -> list[str]:
    reasons = []
    index_duration = _float_or_none(row.get("duration_sec"))
    probe_duration = _float_or_none(probe.get("duration_sec"))
    if probe_duration is None or probe_duration <= 0:
        reasons.append("unreadable_media_duration")
    elif index_duration is not None and abs(index_duration - probe_duration) > max(MAX_MEDIA_TIME_DRIFT_SEC, index_duration * 0.35):
        reasons.append("duration_mismatch")
    return reasons


def _ffprobe_media(path: Path, stream_type: str) -> dict[str, Any]:
    selector = "a:0" if stream_type == "audio" else "v:0"
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                selector,
                "-show_entries",
                "stream=codec_type,codec_name,width,height,duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return {"error": "ffprobe_timeout"}
    if result.returncode != 0:
        return {"error": "ffprobe_failed"}
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"error": "ffprobe_invalid_json"}
    streams = data.get("streams") or []
    if not streams:
        return {"error": f"missing_{stream_type}_stream"}
    stream = streams[0]
    duration = _float_or_none(stream.get("duration")) or _float_or_none((data.get("format") or {}).get("duration"))
    return {
        "codec_type": stream.get("codec_type"),
        "codec_name": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration_sec": duration,
    }


def _inspect_location(session: Path) -> dict[str, Any]:
    rows = read_csv_rows(session / "location" / "geo_location.csv")
    abnormal = 0
    reasons: dict[str, int] = {}
    previous_utc = None
    for row in rows:
        row_reasons = []
        utc = _float_or_none(row.get("utc_sec"))
        lat = _float_or_none(row.get("latitude"))
        lng = _float_or_none(row.get("longitude"))
        accuracy = _float_or_none(row.get("horizontal_accuracy"))
        speed = _float_or_none(row.get("speed"))
        if utc is None:
            row_reasons.append("missing_utc_sec")
        elif previous_utc is not None and utc < previous_utc:
            row_reasons.append("time_reversal")
        if lat is None or lng is None or not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            row_reasons.append("invalid_coordinate")
        if accuracy is not None and accuracy < 0:
            row_reasons.append("invalid_accuracy")
        if speed is not None and speed < -1:
            row_reasons.append("invalid_speed")
        if row_reasons:
            abnormal += 1
            for reason in row_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        if utc is not None:
            previous_utc = utc
    return {"stream": "location", "total": len(rows), "abnormal": abnormal, "reason_counts": reasons, "warnings": []}


def _inspect_motion(session: Path) -> dict[str, Any]:
    rows = read_csv_rows(session / "motion" / "device_motion.csv")
    abnormal = 0
    reasons: dict[str, int] = {}
    previous_utc = None
    numeric_fields = ["user_acc_x", "user_acc_y", "user_acc_z", "rot_x", "rot_y", "rot_z", "roll", "pitch", "yaw"]
    for row in rows:
        row_reasons = []
        utc = _float_or_none(row.get("utc_sec"))
        if utc is None:
            row_reasons.append("missing_utc_sec")
        elif previous_utc is not None and utc < previous_utc:
            row_reasons.append("time_reversal")
        for field in numeric_fields:
            value = _float_or_none(row.get(field))
            if value is not None and not math.isfinite(value):
                row_reasons.append("non_finite_motion_value")
                break
        if row_reasons:
            abnormal += 1
            for reason in row_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        if utc is not None:
            previous_utc = utc
    return {"stream": "motion", "total": len(rows), "abnormal": abnormal, "reason_counts": reasons, "warnings": []}


def _strip_quarantined(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "quarantined"}


def _health_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("summary") or {}
    media_rows = summary.get("media_rows") or 0
    quarantined = summary.get("quarantined_media_rows") or 0
    media_quarantine_rate = quarantined / media_rows if media_rows else 0.0
    return {
        "schema_version": "session_health.v1",
        "session_path": manifest.get("session_path"),
        "status": "warning" if quarantined else "ok",
        "media_quarantine_rate": round(media_quarantine_rate, 4),
        "summary": summary,
        "warnings": manifest.get("warnings") or [],
    }


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
