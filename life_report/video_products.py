from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context_injection import context_for_video, context_metadata, format_model_context, load_clip_contexts, read_context_text_files
from .image_quality import analyze_image_quality, image_similarity
from .model_engine import VideoClipInput, create_video_model
from .pipeline_defaults import DEFAULT_MAX_STORY_KEYFRAMES, DEFAULT_SUMMARY_TEXT_MODEL
from .session_preflight import is_quarantined_audio, is_quarantined_video, load_quarantine_manifest
from .session_loader import read_csv_rows
from .video_story import (
    build_video_compact_raw,
    build_video_story_input,
    build_video_story_media_manifest,
    build_video_timeline,
)


KEYFRAME_INTERVAL_SEC = 2.0
SIMILAR_FRAME_THRESHOLD = 0.965


@dataclass(frozen=True)
class VideoProbeResult:
    output_dir: Path
    video_understandings_path: Path
    video_understandings_pretty_path: Path
    video_preparation_errors_path: Path
    video_model_errors_path: Path
    processed_clip_count: int
    understood_clip_count: int
    preparation_error_count: int = 0
    model_error_count: int = 0


def probe_video_model(
    session_path: Path,
    output_dir: Path,
    provider: str = "mock",
    model: str | None = None,
    limit_clips: int = 3,
    understand_clips: int = 1,
    concurrency: int = 1,
    location_context_path: Path | None = None,
    motion_context_path: Path | None = None,
    quarantine_manifest_path: Path | None = None,
) -> VideoProbeResult:
    session = session_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    clips = read_video_clips(session, quarantine_manifest_path=quarantine_manifest_path)
    if limit_clips > 0:
        clips = clips[:limit_clips]
    contexts = load_clip_contexts(location_context_path, motion_context_path)
    clips = [_attach_video_context(clip, contexts) for clip in clips]

    prepared, preparation_errors = _prepare_clips(
        session,
        output,
        clips,
        concurrency=max(1, concurrency),
        quarantine_manifest_path=quarantine_manifest_path,
    )
    model_inputs = prepared[: max(0, understand_clips)]
    video_model = create_video_model(provider, model=model)
    records = []
    model_errors = []
    if concurrency <= 1:
        for item in model_inputs:
            try:
                records.append(_understand_with_timing(video_model, item))
            except Exception as exc:
                model_errors.append(_model_error(item, exc, provider=provider, model=model))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_understand_with_timing, video_model, item): item for item in model_inputs}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    model_errors.append(_model_error(item, exc, provider=provider, model=model))
    records.sort(key=lambda item: item.get("metadata", {}).get("start_utc_sec") or 0)
    model_errors.sort(key=lambda item: item.get("start_utc_sec") or 0)

    jsonl_path = output / "video_understandings.jsonl"
    json_path = output / "video_understandings.json"
    preparation_errors_path = output / "video_preparation_errors.json"
    model_errors_path = output / "video_model_errors.json"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preparation_errors_path.write_text(json.dumps(preparation_errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_errors_path.write_text(json.dumps(model_errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return VideoProbeResult(
        output,
        jsonl_path,
        json_path,
        preparation_errors_path,
        model_errors_path,
        len(clips),
        len(records),
        len(preparation_errors),
        len(model_errors),
    )


def build_video_products(
    understandings_path: Path,
    output_dir: Path,
    story_model: str = DEFAULT_SUMMARY_TEXT_MODEL,
    provider: str = "aliyun",
    summary_thinking: bool | None = None,
    max_story_keyframes: int = DEFAULT_MAX_STORY_KEYFRAMES,
    location_compact_path: Path | None = None,
    motion_compact_path: Path | None = None,
) -> dict[str, Any]:
    understandings = json.loads(understandings_path.expanduser().resolve().read_text(encoding="utf-8"))
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    timeline = build_video_timeline(understandings)
    timeline_path = output / "video_timeline.json"
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    media_manifest = build_video_story_media_manifest(timeline, max_keyframes=max_story_keyframes)
    media_manifest_path = output / "video_story_media_manifest.json"
    media_manifest_path.write_text(json.dumps(media_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    compact_raw = build_video_compact_raw(timeline, media_manifest=media_manifest)
    compact_raw_path = output / "video_compact_raw.txt"
    compact_raw_path.write_text(compact_raw, encoding="utf-8")

    context_text = read_context_text_files([location_compact_path, motion_compact_path])
    story_input = build_video_story_input(
        compact_raw,
        story_model=story_model,
        provider=provider,
        summary_thinking=summary_thinking,
        context_text=context_text,
    )
    story_input_path = output / "video_story_input.txt"
    story_input_path.write_text(story_input, encoding="utf-8")

    return {
        "video_timeline_path": str(timeline_path),
        "video_compact_raw_path": str(compact_raw_path),
        "video_story_input_path": str(story_input_path),
        "video_story_media_manifest_path": str(media_manifest_path),
        "video_event_count": len(timeline["events"]),
        "video_keyframe_count": len(timeline["keyframes"]),
        "selected_story_keyframe_count": len(media_manifest["selected_keyframes"]),
        "story_model": story_model,
        "context_injected": bool(context_text),
    }


def read_video_clips(session_path: Path, quarantine_manifest_path: Path | None = None) -> list[dict[str, Any]]:
    rows = read_csv_rows(session_path / "video" / "clip_index.csv")
    quarantine_manifest = load_quarantine_manifest(quarantine_manifest_path)
    clips = []
    for row in rows:
        if is_quarantined_video(row, quarantine_manifest):
            continue
        rel_path = row.get("file_path")
        if not rel_path:
            continue
        clips.append(
            {
                "clip_id": str(row.get("clip_id") or Path(rel_path).stem),
                "file_path": session_path / rel_path,
                "start_utc_sec": _float_or_none(row.get("start_utc_sec")),
                "end_utc_sec": _float_or_none(row.get("end_utc_sec")),
                "duration_sec": _float_or_none(row.get("duration_sec")),
                "fps": _float_or_none(row.get("fps")),
            }
        )
    return clips


def _prepare_clips(
    session: Path,
    output: Path,
    clips: list[dict[str, Any]],
    concurrency: int,
    quarantine_manifest_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quarantine_manifest = load_quarantine_manifest(quarantine_manifest_path)
    errors = []
    if concurrency <= 1:
        prepared = []
        for clip in clips:
            try:
                prepared.append(_prepare_clip(session, output, clip, quarantine_manifest))
            except Exception as exc:
                errors.append(_preparation_error(clip, exc))
    else:
        prepared = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_prepare_clip, session, output, clip, quarantine_manifest): clip for clip in clips}
            for future in as_completed(futures):
                clip = futures[future]
                try:
                    prepared.append(future.result())
                except Exception as exc:
                    errors.append(_preparation_error(clip, exc))
    prepared.sort(key=lambda item: item.get("start_utc_sec") or 0)
    errors.sort(key=lambda item: item.get("start_utc_sec") or 0)
    return prepared, errors


def _prepare_clip(session: Path, output: Path, clip: dict[str, Any], quarantine_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    aligned_audio = _aligned_audio_path(output, clip["clip_id"])
    model_video = _model_video_path(output, clip["clip_id"])
    report_video = _report_video_path(output, clip["clip_id"])
    report_keyframe_dir = output / "report_keyframes"
    report_keyframe_dir.mkdir(parents=True, exist_ok=True)
    orientation_filter = _clip_orientation_filter(session, clip["file_path"], clip)
    _extract_aligned_audio(session, clip, aligned_audio, quarantine_manifest=quarantine_manifest)
    _build_model_video(clip["file_path"], aligned_audio, model_video)
    _build_report_video(clip["file_path"], aligned_audio, report_video)
    keyframe_records = _extract_interval_keyframes(clip["file_path"], report_keyframe_dir, clip, orientation_filter=orientation_filter)
    return {
        **clip,
        "aligned_audio_path": str(aligned_audio),
        "model_video_path": str(model_video),
        "report_video_path": str(report_video),
        "orientation_filter": orientation_filter,
        "report_keyframes": keyframe_records,
        "default_report_keyframes": [record["keyframe_path"] for record in keyframe_records if record.get("accepted")],
    }


def _extract_aligned_audio(
    session: Path,
    clip: dict[str, Any],
    output_path: Path,
    quarantine_manifest: dict[str, Any] | None = None,
) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_rows = read_csv_rows(session / "audio" / "audio_index.csv")
    clip_start = clip["start_utc_sec"]
    clip_end = clip["end_utc_sec"]
    if clip_start is None or clip_end is None:
        raise RuntimeError(f"Clip has no UTC time range: {clip['clip_id']}")
    for row in audio_rows:
        if is_quarantined_audio(row, quarantine_manifest or {}):
            continue
        audio_start = _float_or_none(row.get("start_utc_sec"))
        audio_end = _float_or_none(row.get("end_utc_sec"))
        rel_path = row.get("file_path")
        if audio_start is None or audio_end is None or not rel_path:
            continue
        overlap_start = max(clip_start, audio_start)
        overlap_end = min(clip_end, audio_end)
        if overlap_end <= overlap_start:
            continue
        offset = overlap_start - audio_start
        duration = overlap_end - overlap_start
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{offset:.3f}",
                "-i",
                str(session / rel_path),
                "-t",
                f"{duration:.3f}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-af",
                "loudnorm=I=-20:LRA=11:TP=-2",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                str(output_path),
            ]
        )
        return
    _run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000", "-t", f"{clip.get('duration_sec') or 1:.3f}", "-c:a", "aac", str(output_path)])


def _build_model_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = "scale='if(gt(iw,ih),854,-2)':'if(gt(iw,ih),-2,854)',fps=2"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            vf,
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(output_path),
        ]
    )


def _build_report_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(output_path),
        ]
    )


def _extract_interval_keyframes(source_video: Path, keyframe_dir: Path, clip: dict[str, Any], orientation_filter: str | None = None) -> list[dict[str, Any]]:
    duration = max(0.1, float(clip.get("duration_sec") or 1.0))
    points = _interval_points(duration, KEYFRAME_INTERVAL_SEC)
    orientation_filter = orientation_filter if orientation_filter is not None else _source_orientation_filter(source_video)
    records = []
    accepted_records: list[dict[str, Any]] = []
    for index, point in enumerate(points, start=1):
        path = keyframe_dir / f"clip_{clip['clip_id']}_frame_{index:03d}_t{point:.1f}.jpg"
        if not path.exists():
            command = ["ffmpeg", "-y", "-ss", f"{point:.3f}", "-noautorotate", "-i", str(source_video)]
            if orientation_filter:
                command.extend(["-vf", orientation_filter])
            command.extend(["-frames:v", "1", "-q:v", "2", str(path)])
            try:
                _run_ffmpeg(command)
            except Exception as exc:
                records.append(
                    {
                        "keyframe_id": f"video_keyframe_{clip['clip_id']}_{index:03d}",
                        "clip_id": str(clip["clip_id"]),
                        "relative_time_sec": point,
                        "absolute_utc_sec": _absolute_time(clip.get("start_utc_sec"), point),
                        "local_time": _format_local_time(_absolute_time(clip.get("start_utc_sec"), point)),
                        "keyframe_path": str(path),
                        "source_video_path": str(source_video),
                        "caption": f"固定抽帧 {point:.1f}s",
                        "purpose": "fixed_interval",
                        "accepted": False,
                        "quality_score": 0.0,
                        "reject_reasons": ["ffmpeg_keyframe_extract_failed"],
                        "error": _one_line(str(exc), 500),
                    }
                )
                continue
        if not path.exists():
            continue
        quality = analyze_image_quality(path)
        record = {
            "keyframe_id": f"video_keyframe_{clip['clip_id']}_{index:03d}",
            "clip_id": str(clip["clip_id"]),
            "relative_time_sec": point,
            "absolute_utc_sec": _absolute_time(clip.get("start_utc_sec"), point),
            "local_time": _format_local_time(_absolute_time(clip.get("start_utc_sec"), point)),
            "keyframe_path": str(path),
            "source_video_path": str(source_video),
            "caption": f"固定抽帧 {point:.1f}s",
            "purpose": "fixed_interval",
            **quality,
        }
        if record["accepted"] and _is_near_duplicate(record, accepted_records):
            record["accepted"] = False
            record["reject_reasons"] = [*record.get("reject_reasons", []), "near_duplicate"]
        if record["accepted"]:
            accepted_records.append(record)
        records.append(record)
    return records


def _interval_points(duration: float, interval_sec: float = KEYFRAME_INTERVAL_SEC) -> list[float]:
    if duration <= 0:
        return [0.0]
    points = []
    point = 0.0
    while point < duration:
        points.append(round(point, 3))
        point += interval_sec
    return points


def _probe_video_stream(path: Path) -> dict[str, Any]:
    if not shutil.which("ffprobe"):
        return {}
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    streams = data.get("streams") or []
    return streams[0] if streams else {}


def _source_orientation_filter(path: Path) -> str:
    stream = _probe_video_stream(path)
    rotation = _rotation_from_stream(stream)
    normalized = rotation % 360
    if normalized == 180:
        return "hflip,vflip"
    return ""


def _clip_orientation_filter(session: Path, path: Path, clip: dict[str, Any]) -> str:
    stream = _probe_video_stream(path)
    rotation = _rotation_from_stream(stream)
    normalized = rotation % 360
    if normalized == 180:
        return "hflip,vflip"
    if normalized not in {90, 270}:
        return ""
    gravity = _clip_average_gravity(session, clip)
    if gravity is None:
        return ""
    gravity_x, gravity_y, _gravity_z = gravity
    # In current iPhone capture data, landscape clips often carry -90 rotation
    # metadata even when the raw pixels are already landscape. Gravity sign is
    # reliable for distinguishing upright vs upside-down landscape.
    if abs(gravity_x) >= abs(gravity_y) * 1.2:
        return "hflip,vflip" if gravity_x > 0.0 else ""
    return ""


def _clip_average_gravity(session: Path, clip: dict[str, Any]) -> tuple[float, float, float] | None:
    start = _float_or_none(clip.get("start_utc_sec"))
    end = _float_or_none(clip.get("end_utc_sec"))
    if start is None or end is None:
        return None
    path = session / "motion" / "device_motion.csv"
    if not path.exists():
        return None
    values = []
    for row in read_csv_rows(path):
        utc = _float_or_none(row.get("utc_sec"))
        if utc is None or utc < start or utc > end:
            continue
        gx = _float_or_none(row.get("gravity_x"))
        gy = _float_or_none(row.get("gravity_y"))
        gz = _float_or_none(row.get("gravity_z"))
        if gx is None or gy is None or gz is None:
            continue
        values.append((gx, gy, gz))
    if not values:
        return None
    return (
        sum(item[0] for item in values) / len(values),
        sum(item[1] for item in values) / len(values),
        sum(item[2] for item in values) / len(values),
    )


def _rotation_from_stream(stream: dict[str, Any]) -> int:
    rotate = (stream.get("tags") or {}).get("rotate")
    if rotate is not None:
        try:
            return int(float(rotate))
        except (TypeError, ValueError):
            return 0
    for item in stream.get("side_data_list") or []:
        if "rotation" in item:
            try:
                return int(float(item["rotation"]))
            except (TypeError, ValueError):
                return 0
    return 0


def _is_near_duplicate(record: dict[str, Any], accepted_records: list[dict[str, Any]]) -> bool:
    if not accepted_records:
        return False
    previous = accepted_records[-1]
    similarity = image_similarity(Path(record["keyframe_path"]), Path(previous["keyframe_path"]))
    if similarity < SIMILAR_FRAME_THRESHOLD:
        return False
    if float(record.get("quality_score") or 0.0) > float(previous.get("quality_score") or 0.0):
        previous["accepted"] = False
        previous["reject_reasons"] = [*previous.get("reject_reasons", []), "near_duplicate_lower_quality"]
        accepted_records[-1] = record
        return False
    return True


def _understand_with_timing(video_model: Any, item: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    video = VideoClipInput(
        clip_id=str(item["clip_id"]),
        file_path=Path(item["model_video_path"]),
        start_utc_sec=item.get("start_utc_sec"),
        end_utc_sec=item.get("end_utc_sec"),
        duration_sec=item.get("duration_sec"),
        fps=2,
        context_text=item.get("context_text"),
        context_metadata=item.get("context_metadata") or {},
    )
    record = video_model.understand_video(video, _video_prompt(video.context_text))
    finished = time.time()
    record.setdefault("metadata", {}).update(
        {
            "original_video_path": str(item["file_path"]),
            "aligned_audio_path": item["aligned_audio_path"],
            "model_video_path": item["model_video_path"],
            "report_video_path": item["report_video_path"],
            "orientation_filter": item.get("orientation_filter", ""),
            "report_keyframes": item.get("report_keyframes", []),
            "keyframe_quality": item.get("report_keyframes", []),
            "default_report_keyframes": item["default_report_keyframes"],
            "start_utc_sec": item.get("start_utc_sec"),
            "end_utc_sec": item.get("end_utc_sec"),
            "duration_sec": item.get("duration_sec"),
            "context": item.get("context_metadata"),
            "context_text": item.get("context_text"),
        }
    )
    record["processing_duration_sec"] = round(finished - started, 3)
    return record


def _attach_video_context(clip: dict[str, Any], contexts: dict[str, Any]) -> dict[str, Any]:
    clip_context = context_for_video(contexts, str(clip.get("clip_id")))
    return {
        **clip,
        "context_text": format_model_context(clip_context),
        "context_metadata": context_metadata(clip_context),
    }


def _video_prompt(context_text: str | None = None) -> str:
    prompt = (
        "你是 PhoneLifeAgent 的视频+音频理解模块。输入是降帧率后的生活记录短视频，已包含对齐音频。"
        "请只输出 JSON，不要 markdown。必须覆盖：clip_summary、scene、people、objects、activities、audio_dialogue、emotions、"
        "event_candidates、todo_candidates、memory_candidates、highlight_moments、life_story_hint、confidence、evidence_notes。"
        "重点识别场景、人物角色、物品、交通工具、标识牌、屏幕文字、对话、情绪、会议/家庭/高兴/难过/风险等生活事件。"
        "objects 只记录和生活事件有关的物品，不要枚举无意义杂物。"
        "highlight_moments 可给 relative_time_sec，但不要输出 keyframe_candidates；关键帧由程序固定抽取和质量过滤。"
        "如果提供了 Location/Motion Context，请把它作为辅助证据，用于判断地点、移动状态和视频可靠性；如果和画面或声音冲突，请明确写不确定。"
        "不确定写 unknown，不要编造身份、地点或关系。life_story_hint 使用第一人称。"
    )
    if context_text:
        prompt += "\n\n" + context_text
    return prompt


def _aligned_audio_path(output: Path, clip_id: str) -> Path:
    return output / "aligned_audio" / f"clip_{clip_id}_audio.m4a"


def _model_video_path(output: Path, clip_id: str) -> Path:
    return output / "model_video" / f"clip_{clip_id}_model.mp4"


def _report_video_path(output: Path, clip_id: str) -> Path:
    return output / "report_video" / f"clip_{clip_id}_report.mp4"


def _run_ffmpeg(command: list[str]) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for video preprocessing.")
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = _one_line(result.stderr, 1000)
        raise RuntimeError(f"ffmpeg failed with exit {result.returncode}: {' '.join(command)} | {stderr}")


def _preparation_error(clip: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "clip_id": str(clip.get("clip_id")),
        "file_path": str(clip.get("file_path")),
        "start_utc_sec": clip.get("start_utc_sec"),
        "end_utc_sec": clip.get("end_utc_sec"),
        "duration_sec": clip.get("duration_sec"),
        "reason": type(exc).__name__,
        "error": _one_line(str(exc), 1000),
    }


def _model_error(item: dict[str, Any], exc: Exception, provider: str, model: str | None) -> dict[str, Any]:
    return {
        "stage": "video_understanding",
        "clip_id": str(item.get("clip_id")),
        "source_path": str(item.get("file_path")),
        "model_video_path": str(item.get("model_video_path")),
        "report_video_path": str(item.get("report_video_path")),
        "start_utc_sec": item.get("start_utc_sec"),
        "end_utc_sec": item.get("end_utc_sec"),
        "duration_sec": item.get("duration_sec"),
        "provider": provider,
        "model": model,
        "reason": type(exc).__name__,
        "error": _one_line(str(exc), 1000),
        "retryable": True,
    }


def _one_line(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _absolute_time(start: Any, offset: Any) -> float | None:
    start_value = _float_or_none(start)
    offset_value = _float_or_none(offset)
    if start_value is None or offset_value is None:
        return None
    return start_value + offset_value


def _format_local_time(utc_sec: Any) -> str:
    value = _float_or_none(utc_sec)
    if value is None:
        return "-"
    from datetime import datetime

    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
