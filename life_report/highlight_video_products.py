from __future__ import annotations

import html
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from .model_engine import create_text_llm
from .pipeline_defaults import (
    DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
    DEFAULT_HIGHLIGHT_TARGET_SECONDS,
    DEFAULT_HIGHLIGHT_TEXT_MODEL,
    DEFAULT_HIGHLIGHT_THINKING,
    highlight_generation_policy,
)
from .settings_store import apply_api_settings, load_api_settings


OUTPUT_SIZE = (1920, 1080)


def build_highlight_video_products(
    run_dir: Path,
    output_dir: Path | None = None,
    provider: str = "aliyun",
    text_model: str = DEFAULT_HIGHLIGHT_TEXT_MODEL,
    highlight_thinking: bool | None = None,
    target_seconds: int = DEFAULT_HIGHLIGHT_TARGET_SECONDS,
    max_segments: int = DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    def log(message: str) -> None:
        if progress:
            progress(message)

    run = run_dir.expanduser().resolve()
    output = (output_dir or run / "highlight_video").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if provider == "aliyun":
        settings = load_api_settings(Path.cwd())
        if settings.dashscope_api_key or settings.dashscope_openai_base_url:
            apply_api_settings(settings)

    log("Reading Story and video media manifest...")
    story_markdown = _read_text(run / "story" / "life_story.md")
    story_json = _read_json(run / "story" / "life_story.json")
    evidence_pack = _read_json(run / "story" / "story_evidence_pack.json")
    media_manifest = _read_json(run / "video" / "products" / "video_story_media_manifest.json")
    report_videos = _load_report_videos(media_manifest)
    if not report_videos:
        raise RuntimeError("No selected_report_videos found in video_story_media_manifest.json.")

    log("Calling text model for Highlight Video storyline...")
    storyline = build_highlight_storyline(
        story_markdown=story_markdown,
        story_json=story_json,
        evidence_pack=evidence_pack,
        report_videos=report_videos,
        provider=provider,
        text_model=text_model,
        highlight_thinking=highlight_thinking,
        target_seconds=target_seconds,
        max_segments=max_segments,
    )
    storyline_path = output / "highlight_storyline.json"
    storyline_path.write_text(json.dumps(storyline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plan = build_highlight_plan(storyline, report_videos, target_seconds=target_seconds, max_segments=max_segments)
    plan_path = output / "highlight_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    video_path = output / "highlight_video.mp4"
    log(f"Rendering Highlight Video with ffmpeg ({len(plan['segments'])} segment(s))...")
    render_highlight_video(plan, video_path)

    html_path = output / "highlight_video.html"
    html_path.write_text(render_highlight_html(video_path.name, plan), encoding="utf-8")
    log("Highlight Video files written.")

    return {
        "highlight_storyline_path": str(storyline_path),
        "highlight_plan_path": str(plan_path),
        "highlight_video_path": str(video_path),
        "highlight_video_html_path": str(html_path),
        "provider": provider,
        "text_model": text_model,
        "target_seconds": target_seconds,
        "segment_count": len(plan["segments"]),
        "estimated_duration_sec": plan["estimated_duration_sec"],
    }


def build_highlight_storyline(
    story_markdown: str,
    story_json: dict[str, Any],
    evidence_pack: dict[str, Any],
    report_videos: list[dict[str, Any]],
    provider: str,
    text_model: str,
    target_seconds: int,
    max_segments: int,
    highlight_thinking: bool | None = None,
) -> dict[str, Any]:
    resolved_thinking = DEFAULT_HIGHLIGHT_THINKING if highlight_thinking is None else highlight_thinking
    policy = highlight_generation_policy(
        provider=provider,
        text_model=text_model,
        target_seconds=target_seconds,
        max_segments=max_segments,
        enable_thinking=resolved_thinking,
    )
    if policy.provider not in {"aliyun"}:
        raise RuntimeError("Highlight Video v1 requires provider='aliyun'. No fallback selector is used.")
    llm = create_text_llm(policy.provider, model=policy.text_model, enable_thinking=policy.enable_thinking)
    response = llm.generate_text(
        _highlight_system_prompt(max_segments=max_segments),
        _highlight_user_prompt(story_markdown, story_json, evidence_pack, report_videos, target_seconds, max_segments),
    )
    parsed = _parse_json_text(response)
    if not parsed:
        raise RuntimeError("Highlight Video model did not return valid JSON.")
    return _normalize_highlight_storyline(parsed, report_videos, target_seconds=target_seconds, max_segments=max_segments)


def build_highlight_plan(
    storyline: dict[str, Any],
    report_videos: list[dict[str, Any]],
    target_seconds: int,
    max_segments: int,
) -> dict[str, Any]:
    videos_by_id = {str(item["clip_id"]): item for item in report_videos}
    used = set()
    segments = []
    for index, beat in enumerate(storyline.get("beats", [])[:max_segments], start=1):
        clip_id = str(beat.get("clip_id") or "")
        if clip_id in used:
            raise RuntimeError(f"Duplicate clip_id in Highlight Video plan: {clip_id}")
        video = videos_by_id.get(clip_id)
        if not video:
            raise RuntimeError(f"Unknown clip_id in Highlight Video plan: {clip_id}")
        used.add(clip_id)
        duration = float(video.get("duration_sec") or 0) or _probe_duration(Path(video["path"]))
        start_sec, end_sec = _segment_range(beat, duration)
        segments.append(
            {
                "order": index,
                "beat_id": str(beat.get("beat_id") or f"beat_{index:02d}"),
                "clip_id": clip_id,
                "source_path": video["path"],
                "local_time_range": video.get("local_time_range", ""),
                "summary": video.get("summary", ""),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
                "source_rotation": video.get("rotation"),
                "orientation_filter": video.get("orientation_filter", ""),
                "caption": _one_line(beat.get("caption") or beat.get("story_beat") or video.get("summary"), 28),
                "story_beat": _one_line(beat.get("story_beat"), 80),
                "reason": _one_line(beat.get("reason"), 120),
            }
        )
    if not segments:
        raise RuntimeError("Highlight Video model returned no usable segments.")
    segments.sort(key=lambda item: (_clip_id_sort_key(item["clip_id"]), item["order"]))
    for index, segment in enumerate(segments, start=1):
        segment["order"] = index
    title = _short_title(str(storyline.get("title") or _first_story_title(storyline) or "今日高光"))
    date_label = _one_line(storyline.get("date_label"), 24)
    estimated = 2.4 + sum(item["duration_sec"] for item in segments) + 2.2
    return {
        "schema_version": "highlight_video_plan.v1",
        "title": title,
        "date_label": date_label,
        "summary": _one_line(storyline.get("summary"), 120),
        "target_seconds": target_seconds,
        "estimated_duration_sec": round(estimated, 3),
        "render": {
            "width": OUTPUT_SIZE[0],
            "height": OUTPUT_SIZE[1],
            "fps": 30,
            "audio": "keep_original_resampled_stereo",
        },
        "segments": segments,
    }


def render_highlight_video(plan: dict[str, Any], output_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to render Highlight Video.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        work = Path(temp_dir)
        parts = []
        title_card = work / "part_000_title.mp4"
        _render_text_card(
            title=plan.get("title") or "今日高光",
            subtitle=plan.get("date_label") or "PhoneLifeAgent",
            output_path=title_card,
            duration=2.4,
        )
        parts.append(title_card)
        for index, segment in enumerate(plan.get("segments", []), start=1):
            part_path = work / f"part_{index:03d}.mp4"
            _render_video_segment(segment, part_path)
            parts.append(part_path)
        end_card = work / f"part_{len(parts):03d}_end.mp4"
        _render_text_card(
            title=plan.get("title") or "这一天的高光",
            subtitle="PhoneLifeAgent",
            output_path=end_card,
            duration=2.2,
        )
        parts.append(end_card)

        concat_path = work / "concat.txt"
        concat_path.write_text("".join(f"file '{_ffconcat_path(path)}'\n" for path in parts), encoding="utf-8")
        planned_duration = _planned_render_duration(plan)
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-fflags",
                "+genpts",
                "-t",
                f"{planned_duration:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )


def render_highlight_html(video_name: str, plan: dict[str, Any]) -> str:
    rows = []
    for segment in plan.get("segments", []):
        rows.append(
            "<li>"
            f"<b>{html.escape(str(segment.get('caption') or ''))}</b>"
            f"<span>{html.escape(str(segment.get('local_time_range') or ''))}</span>"
            f"<p>{html.escape(str(segment.get('reason') or segment.get('summary') or ''))}</p>"
            "</li>"
        )
    return (
        '<!doctype html><meta charset="utf-8"><title>Highlight Video</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<style>"
        "body{margin:0;background:#050505;color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,sans-serif}"
        "main{width:min(100%,780px);margin:0 auto;padding:28px 16px 52px}"
        "video{width:100%;border-radius:22px;background:#000;display:block}"
        "h1{font-size:30px;margin:22px 0 6px}p{color:#c9c9d1;line-height:1.65}"
        "ol{padding-left:1.2rem}li{margin:16px 0}li span{display:block;color:#8f96a3;font-size:13px;margin-top:2px}"
        "</style>"
        f'<main><video src="{html.escape(video_name)}" controls playsinline></video>'
        f"<h1>{html.escape(str(plan.get('title') or 'Highlight Video'))}</h1>"
        f"<p>{html.escape(str(plan.get('summary') or ''))}</p>"
        "<ol>"
        + "".join(rows)
        + "</ol></main>\n"
    )


def _planned_render_duration(plan: dict[str, Any]) -> float:
    return round(2.4 + sum(float(item.get("duration_sec") or 0) for item in plan.get("segments", [])) + 2.2, 3)


def _load_report_videos(media_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    videos = []
    for item in media_manifest.get("selected_report_videos", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path") or ""))
        if not path.exists():
            continue
        record = {
            "clip_id": str(item.get("clip_id") or path.stem),
            "path": str(path),
            "local_time_range": _one_line(item.get("local_time_range"), 80),
            "summary": _one_line(item.get("summary"), 180),
            "duration_sec": _probe_duration(path),
            "rotation": _probe_rotation(path),
            "orientation_filter": str(item.get("orientation_filter") or ""),
        }
        videos.append(record)
    videos.sort(key=lambda row: _clip_id_sort_key(row["clip_id"]))
    return videos


def _normalize_highlight_storyline(
    parsed: dict[str, Any],
    report_videos: list[dict[str, Any]],
    target_seconds: int,
    max_segments: int,
) -> dict[str, Any]:
    valid_ids = {str(item["clip_id"]) for item in report_videos}
    beats = []
    for index, item in enumerate(parsed.get("beats") or parsed.get("segments") or [], start=1):
        if not isinstance(item, dict):
            continue
        clip_id = str(item.get("clip_id") or "")
        if clip_id not in valid_ids:
            raise RuntimeError(f"Highlight Video model selected invalid clip_id: {clip_id}")
        beats.append(
            {
                "beat_id": str(item.get("beat_id") or f"beat_{index:02d}"),
                "order": int(item.get("order") or index),
                "story_beat": _one_line(item.get("story_beat") or item.get("description") or item.get("summary"), 100),
                "clip_id": clip_id,
                "start_sec": _float_or_none(item.get("start_sec")),
                "end_sec": _float_or_none(item.get("end_sec")),
                "caption": _one_line(item.get("caption") or item.get("title"), 28),
                "reason": _one_line(item.get("reason"), 120),
            }
        )
    beats = beats[:max_segments]
    if len(beats) < 1:
        raise RuntimeError("Highlight Video model returned no beats.")
    beats.sort(key=lambda row: (row["order"], _clip_id_sort_key(row["clip_id"])))
    target_count = _target_segment_count(target_seconds, max_segments, len(report_videos))
    if len(beats) < target_count:
        used = {str(beat["clip_id"]) for beat in beats}
        for video in _supplemental_highlight_videos(report_videos, used, target_count - len(beats)):
            index = len(beats) + 1
            beats.append(
                {
                    "beat_id": f"beat_{index:02d}",
                    "order": index,
                    "story_beat": _one_line(video.get("summary"), 100),
                    "clip_id": str(video["clip_id"]),
                    "start_sec": None,
                    "end_sec": None,
                    "caption": _one_line(video.get("summary"), 28),
                    "reason": "自动补足高光覆盖，避免长 session 只剪到少量片段。",
                }
            )
            used.add(str(video["clip_id"]))
    beats = beats[:max_segments]
    return {
        "schema_version": "highlight_video_storyline.v1",
        "title": _short_title(str(parsed.get("title") or "今日高光")),
        "date_label": _one_line(parsed.get("date_label"), 24),
        "summary": _one_line(parsed.get("summary") or parsed.get("storyline"), 120),
        "target_seconds": int(target_seconds),
        "max_segments": int(max_segments),
        "beats": beats,
    }


def _highlight_system_prompt(max_segments: int) -> str:
    return "".join(highlight_schema_rules(max_segments) + highlight_style_rules() + highlight_evidence_rules())


def highlight_schema_rules(max_segments: int) -> list[str]:
    return [
        "你是 PhoneLifeAgent 的 Highlight Video 剪辑导演。",
        "你必须基于最终 Life Story 和真实 report video clips 设计高光视频剪辑计划。",
        "只能选择输入列表里的 clip_id，不能编造视频。",
        "优先选择 7 到 ",
        f"{max_segments} 个片段；除非可用素材不足，不要少于 7 个。必须覆盖主要转折和前中后时间段。",
        "每个 clip_id 最多用一次。",
        "只输出简体中文 JSON，不要 markdown，不要英文叙述。",
        "字段：title、date_label、summary、beats。",
        "beats 每项字段：order、story_beat、clip_id、start_sec、end_sec、caption、reason。",
        "start_sec/end_sec 是相对该 report video 的秒数，优先 3-6 秒；如果片段本身很短可以使用 0 到视频结尾。",
    ]


def highlight_style_rules() -> list[str]:
    return [
        "title、summary、story_beat、caption、reason 要保持第一人称生活记录视角；可以写“我看到/我记录到”。",
    ]


def highlight_evidence_rules() -> list[str]:
    return [
        "不要写“用户/拍摄者/father/daughter/the child”。",
    ]


def _target_segment_count(target_seconds: int, max_segments: int, available_count: int) -> int:
    desired = 7 if target_seconds >= 35 else 5
    return max(1, min(max_segments, available_count, desired))


def _supplemental_highlight_videos(report_videos: list[dict[str, Any]], used_ids: set[str], needed: int) -> list[dict[str, Any]]:
    candidates = [video for video in report_videos if str(video.get("clip_id")) not in used_ids]
    if needed <= 0 or not candidates:
        return []
    if len(candidates) <= needed:
        return candidates
    if needed == 1:
        return [candidates[len(candidates) // 2]]
    step = (len(candidates) - 1) / (needed - 1)
    selected = []
    seen = set()
    for index in range(needed):
        candidate_index = round(index * step)
        if candidate_index not in seen:
            selected.append(candidates[candidate_index])
            seen.add(candidate_index)
    return selected[:needed]


def _highlight_user_prompt(
    story_markdown: str,
    story_json: dict[str, Any],
    evidence_pack: dict[str, Any],
    report_videos: list[dict[str, Any]],
    target_seconds: int,
    max_segments: int,
) -> str:
    clips = [
        {
            "clip_id": item["clip_id"],
            "duration_sec": item["duration_sec"],
            "local_time_range": item.get("local_time_range"),
            "summary": item.get("summary"),
        }
        for item in report_videos
    ]
    return (
        f"目标总时长：约 {target_seconds} 秒；最多片段数：{max_segments}。\n\n"
        "Final Life Story:\n"
        f"{_trim_text(story_markdown, 8000)}\n\n"
        "Story metadata:\n"
        f"{json.dumps({key: story_json.get(key) for key in ['time_range', 'source_counts', 'story_model']}, ensure_ascii=False)}\n\n"
        "Evidence time range:\n"
        f"{json.dumps(evidence_pack.get('time_range', {}), ensure_ascii=False)}\n\n"
        "Available report video clips:\n"
        f"{json.dumps(clips, ensure_ascii=False, indent=2)}\n"
    )


def _segment_range(beat: dict[str, Any], duration: float) -> tuple[float, float]:
    duration = max(0.1, float(duration or 0.1))
    start = _float_or_none(beat.get("start_sec"))
    end = _float_or_none(beat.get("end_sec"))
    if start is None or end is None or end <= start:
        if duration <= 10:
            return 0.0, duration
        clip_len = min(6.0, max(3.0, duration * 0.6))
        start = max(0.0, (duration - clip_len) / 2)
        end = min(duration, start + clip_len)
    start = max(0.0, min(float(start), duration - 0.1))
    end = max(start + 0.1, min(float(end), duration))
    if end - start > 7.0:
        end = min(duration, start + 7.0)
    if end - start < 3.0 and duration >= 3.0:
        end = min(duration, start + 3.0)
    return start, end


def _render_video_segment(segment: dict[str, Any], output_path: Path) -> None:
    source = Path(str(segment["source_path"]))
    caption_path = output_path.with_suffix(".caption.png")
    _render_caption_overlay(str(segment.get("caption") or ""), caption_path)
    orientation_filter = str(segment.get("orientation_filter") or "")
    orientation = f"{orientation_filter}," if orientation_filter else _source_orientation_filter(source)
    vf = (
        f"[0:v]{orientation}"
        "setpts=PTS-STARTPTS,"
        f"scale={OUTPUT_SIZE[0]}:{OUTPUT_SIZE[1]}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_SIZE[0]}:{OUTPUT_SIZE[1]}[base];"
        "[base][1:v]overlay=0:0:format=auto,"
        "fps=30,format=yuv420p[v]"
    )
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{float(segment['start_sec']):.3f}",
        "-noautorotate",
        "-i",
        str(source),
        "-loop",
        "1",
        "-i",
        str(caption_path),
        "-t",
        f"{float(segment['duration_sec']):.3f}",
        "-filter_complex",
        vf,
        "-map",
        "[v]",
        "-map",
        "0:a:0?",
        "-af",
        "asetpts=PTS-STARTPTS,aresample=44100,pan=stereo|FL=c0|FR=c0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "128k",
        "-shortest",
        str(output_path),
    ]
    _run(command)


def _render_caption_overlay(text: str, output_path: Path) -> None:
    image = Image.new("RGBA", OUTPUT_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(48)
    lines = _wrap_text(_one_line(text, 36), font, OUTPUT_SIZE[0] - 360, draw)[:2]
    line_h = 62
    box_h = len(lines) * line_h + 54
    x0, y0 = 150, OUTPUT_SIZE[1] - box_h - 78
    x1, y1 = OUTPUT_SIZE[0] - 150, y0 + box_h
    draw.rounded_rectangle((x0, y0, x1, y1), radius=28, fill=(0, 0, 0, 132))
    y = y0 + 28
    for line in lines:
        width = draw.textlength(line, font=font)
        draw.text(((OUTPUT_SIZE[0] - width) / 2, y), line, font=font, fill=(255, 255, 255, 245))
        y += line_h
    image.save(output_path)


def _render_text_card(title: str, subtitle: str, output_path: Path, duration: float) -> None:
    image_path = output_path.with_suffix(".png")
    image = Image.new("RGB", OUTPUT_SIZE, (8, 10, 14))
    draw = ImageDraw.Draw(image)
    title_font = _font(78)
    subtitle_font = _font(40)
    title_lines = _wrap_text(title, title_font, OUTPUT_SIZE[0] - 420, draw)[:3]
    subtitle_lines = _wrap_text(subtitle, subtitle_font, OUTPUT_SIZE[0] - 520, draw)[:2]
    total_h = len(title_lines) * 104 + len(subtitle_lines) * 58 + 40
    y = (OUTPUT_SIZE[1] - total_h) // 2
    for line in title_lines:
        w = draw.textlength(line, font=title_font)
        draw.text(((OUTPUT_SIZE[0] - w) / 2, y), line, font=title_font, fill=(245, 247, 250))
        y += 104
    y += 24
    for line in subtitle_lines:
        w = draw.textlength(line, font=subtitle_font)
        draw.text(((OUTPUT_SIZE[0] - w) / 2, y), line, font=subtitle_font, fill=(142, 194, 255))
        y += 58
    image.save(image_path)
    _run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t",
            f"{duration:.3f}",
            "-vf",
            "fps=30,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-b:a",
            "128k",
            "-shortest",
            str(output_path),
        ]
    )


def _drawtext_filter(text: str) -> str:
    clean = _escape_drawtext(_one_line(text, 28))
    fontfile = _font_file()
    return (
        "drawtext="
        f"fontfile='{fontfile}':"
        f"text='{clean}':"
        "fontsize=54:fontcolor=white:"
        "box=1:boxcolor=black@0.48:boxborderw=28:"
        "x=(w-text_w)/2:y=h-210"
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _font_file() -> str:
    for path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if Path(path).exists():
            return path.replace("'", r"\'")
    return ""


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    lines = []
    current = ""
    for char in str(text):
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _probe_duration(path: Path) -> float:
    if not shutil.which("ffprobe"):
        return 0.0
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        return round(float(result.stdout.strip()), 3)
    except ValueError:
        return 0.0


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
    if not streams:
        return {}
    return streams[0]


def _probe_rotation(path: Path) -> int:
    stream = _probe_video_stream(path)
    if not stream:
        return 0
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


def _source_orientation_filter(path: Path) -> str:
    stream = _probe_video_stream(path)
    rotation = _rotation_from_stream(stream)
    normalized = rotation % 360
    if normalized == 180:
        return "hflip,vflip,"
    return ""


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


def _run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[-2000:])


def _parse_json_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _trim_text(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 60].rstrip() + "\n...[trimmed]...\n" + text[-40:].lstrip()


def _one_line(value: Any, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip("，。,. ") + "。"


def _short_title(title: str) -> str:
    title = re.sub(r"[#*_`]", "", title).strip()
    return title if len(title) <= 16 else title[:16].rstrip("，。,. ")


def _first_story_title(storyline: dict[str, Any]) -> str:
    for beat in storyline.get("beats", []):
        if beat.get("caption"):
            return str(beat["caption"])
    return ""


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip_id_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (10**9, text)


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def _ffconcat_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")
