from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .context_store import ContextStore, ProductPaths
from .model_engine import create_text_llm
from .pipeline_defaults import DEFAULT_MAX_STORY_KEYFRAMES, DEFAULT_STORY_TEXT_MODEL, DEFAULT_STORY_THINKING, story_generation_policy


DEFAULT_STORY_MODEL = DEFAULT_STORY_TEXT_MODEL
GENERIC_STORY_TITLES = {"我的个人生活故事报告", "PhoneLifeAgent Life Story"}
STORY_UI_HEADING_MAP = {
    "按时间的故事线": "故事线",
    "TODO": "待办",
    "TODO Candidates": "待办",
    "值得记住的长期记忆": "值得记住",
    "Memory Candidates": "值得记住",
}
FRAME_TEXT_STOPWORDS = {
    "今天",
    "一个",
    "一些",
    "这个",
    "那个",
    "画面",
    "片段",
    "镜头",
    "场景",
    "故事",
    "生活",
    "时候",
    "自己",
    "我们",
    "他们",
    "东西",
}


def build_story_products(
    audio_products_dir: Path | None,
    video_products_dir: Path | None,
    location_products_dir: Path | None,
    motion_products_dir: Path | None,
    output_dir: Path,
    provider: str = "aliyun",
    story_model: str = DEFAULT_STORY_MODEL,
    story_thinking: bool | None = None,
    max_keyframes: int = DEFAULT_MAX_STORY_KEYFRAMES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    def log(message: str) -> None:
        if progress:
            progress(message)

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    log("Final Story: loading pipeline products...")
    store = ContextStore(
        ProductPaths(
            audio_products_dir=audio_products_dir,
            video_products_dir=video_products_dir,
            location_products_dir=location_products_dir,
            motion_products_dir=motion_products_dir,
        )
    )
    log("Final Story: building evidence pack...")
    evidence_pack = build_story_evidence_pack(store, max_keyframes=max_keyframes)
    evidence_pack_path = output / "story_evidence_pack.json"
    evidence_pack_path.write_text(json.dumps(evidence_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("Final Story: building model input...")
    story_input = build_story_input(evidence_pack)
    story_input_path = output / "story_input.txt"
    story_input_path.write_text(story_input, encoding="utf-8")

    report_markdown = generate_life_story(
        story_input,
        provider=provider,
        story_model=story_model,
        story_thinking=story_thinking,
        progress=progress,
    )
    markdown_path = output / "life_story.md"
    markdown_path.write_text(report_markdown, encoding="utf-8")

    log("Final Story: rendering HTML...")
    html_path = output / "life_story.html"
    html_path.write_text(render_story_html(report_markdown, evidence_pack), encoding="utf-8")

    story_json = {
        "schema_version": "life_story.v1",
        "provider": provider,
        "story_model": story_model,
        "time_range": evidence_pack.get("time_range", {}),
        "report_markdown": report_markdown,
        "evidence_pack_path": str(evidence_pack_path),
        "story_input_path": str(story_input_path),
        "html_path": str(html_path),
        "source_counts": evidence_pack.get("source_counts", {}),
        "media": evidence_pack.get("media", {}),
    }
    json_path = output / "life_story.json"
    json_path.write_text(json.dumps(story_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "story_evidence_pack_path": str(evidence_pack_path),
        "story_input_path": str(story_input_path),
        "life_story_json_path": str(json_path),
        "life_story_markdown_path": str(markdown_path),
        "life_story_html_path": str(html_path),
        "provider": provider,
        "story_model": story_model,
        "timeline_item_count": len(evidence_pack.get("timeline", [])),
        "selected_keyframe_count": len(evidence_pack.get("media", {}).get("selected_keyframes", [])),
    }


def build_story_evidence_pack(store: ContextStore, max_keyframes: int = 16) -> dict[str, Any]:
    time_range = store.global_time_range()
    timeline = build_fused_timeline(store)
    selected_keyframes = store.get_keyframes(max_count=max_keyframes)
    route_map = store.location_timeline.get("overall_map_image")
    return {
        "schema_version": "story_evidence_pack.v1",
        "time_range": {
            **time_range,
            "start_local_time": _format_local_time(time_range.get("start_utc_sec")),
            "end_local_time": _format_local_time(time_range.get("end_utc_sec")),
        },
        "source_counts": {
            "audio_segments": len(store.audio_timeline.get("segments", [])),
            "audio_events": len(store.audio_timeline.get("events", [])),
            "video_clips": len(store.video_timeline.get("clips", [])),
            "video_events": len(store.video_timeline.get("events", [])),
            "location_segments": len(store.location_timeline.get("segments", [])),
            "motion_segments": len(store.motion_timeline.get("segments", [])),
            "selected_keyframes": len(selected_keyframes),
        },
        "source_story_inputs": {
            "audio_story_input": _trim_text(store.audio_story_input, 12000),
            "video_story_input": _trim_text(store.video_story_input, 14000),
        },
        "global_context": {
            "location_compact_raw": _trim_text(store.location_compact_raw, 7000),
            "motion_compact_raw": _trim_text(store.motion_compact_raw, 5000),
        },
        "timeline": timeline,
        "todos": _collect_candidates(store, "todo_candidates"),
        "memories": _collect_candidates(store, "memory_candidates"),
        "media": {
            "overall_route_map": route_map,
            "selected_keyframes": selected_keyframes,
            "selected_report_videos": store.video_media_manifest.get("selected_report_videos", []),
        },
    }


def build_fused_timeline(store: ContextStore) -> list[dict[str, Any]]:
    items = []
    for event in store.audio_timeline.get("events", []):
        items.append(_timeline_record("audio_event", event, store.query_context(event.get("absolute_start_utc_sec"), event.get("absolute_end_utc_sec"))))
    for event in store.video_timeline.get("events", []):
        items.append(_timeline_record("video_event", event, store.query_context(event.get("absolute_start_utc_sec"), event.get("absolute_end_utc_sec"))))
    for clip in store.video_timeline.get("clips", []):
        if not any(str(item.get("source_id")) == str(clip.get("clip_id")) and item.get("source_type") == "video_event" for item in items):
            items.append(_timeline_record("video_clip", clip, store.query_context(clip.get("start_utc_sec"), clip.get("end_utc_sec"))))
    for segment in store.audio_timeline.get("segments", []):
        if not any(str(item.get("source_id")) == str(segment.get("audio_id")) and item.get("source_type") == "audio_event" for item in items):
            items.append(_timeline_record("audio_segment", segment, store.query_context(segment.get("start_utc_sec"), segment.get("end_utc_sec"))))
    items.sort(key=lambda item: (item.get("start_utc_sec") is None, item.get("start_utc_sec") or 0))
    return items


def build_story_input(evidence_pack: dict[str, Any]) -> str:
    lines = [
        "PhoneLifeAgent Final Story Input",
        f"Time range: {evidence_pack.get('time_range', {}).get('start_local_time')} -> {evidence_pack.get('time_range', {}).get('end_local_time')}",
        f"Source counts: {json.dumps(evidence_pack.get('source_counts', {}), ensure_ascii=False)}",
        "",
        "Audio Story Input",
        evidence_pack.get("source_story_inputs", {}).get("audio_story_input", "").strip() or "(empty)",
        "",
        "Video Story Input",
        evidence_pack.get("source_story_inputs", {}).get("video_story_input", "").strip() or "(empty)",
        "",
        "Global Location Context",
        evidence_pack.get("global_context", {}).get("location_compact_raw", "").strip() or "(empty)",
        "",
        "Global Motion Context",
        evidence_pack.get("global_context", {}).get("motion_compact_raw", "").strip() or "(empty)",
        "",
        "Fused Timeline Evidence",
    ]
    for item in evidence_pack.get("timeline", []):
        lines.append(_timeline_line(item))
    todos = evidence_pack.get("todos", [])
    if todos:
        lines.append("\nTODO Candidates")
        for item in todos:
            lines.append(f"- {item.get('source_type')} {item.get('source_id')}: {_one_line(item.get('candidate'))}")
    memories = evidence_pack.get("memories", [])
    if memories:
        lines.append("\nMemory Candidates")
        for item in memories:
            lines.append(f"- {item.get('source_type')} {item.get('source_id')}: {_one_line(item.get('candidate'))}")
    media = evidence_pack.get("media", {})
    lines.extend(["", "Media Evidence", f"Overall route map: {media.get('overall_route_map')}"])
    for frame in media.get("selected_keyframes", []):
        lines.append(
            f"- keyframe {frame.get('keyframe_id')} clip={frame.get('clip_id')} time={frame.get('local_time')} path={frame.get('keyframe_path')} caption={_one_line(frame.get('caption') or frame.get('reason'))}"
        )
    return "\n".join(lines).strip() + "\n"


def generate_life_story(
    story_input: str,
    provider: str,
    story_model: str,
    story_thinking: bool | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    def log(message: str) -> None:
        if progress:
            progress(message)

    resolved_thinking = DEFAULT_STORY_THINKING if story_thinking is None else story_thinking
    policy = story_generation_policy(provider=provider, text_model=story_model, enable_thinking=resolved_thinking)
    text_model = create_text_llm(policy.provider, model=policy.text_model, enable_thinking=policy.enable_thinking)
    log(f"Final Story: calling {policy.text_model} for draft...")
    draft = text_model.generate_text(_story_system_prompt(), story_input)
    if provider in {"mock", "none"}:
        return draft
    log("Final Story: reviewing factual claims...")
    return text_model.generate_text(_claim_review_system_prompt(), _claim_review_user_prompt(draft, story_input))


def render_story_html(markdown_text: str, evidence_pack: dict[str, Any]) -> str:
    media = evidence_pack.get("media", {})
    route_map = media.get("overall_route_map")
    keyframes = [frame for frame in media.get("selected_keyframes", []) if frame.get("keyframe_path")]
    ui_markdown = _story_markdown_for_ui(markdown_text)
    title = _story_title(ui_markdown)
    subtitle = _story_subtitle(ui_markdown)
    date_label = _story_date_label(evidence_pack)
    body = _markdown_to_story_cards(ui_markdown, route_map=route_map, keyframes=keyframes)
    route_overview = _route_overview_html(ui_markdown, route_map)
    hero = _hero_collage(_selected_storyline_keyframes(ui_markdown, keyframes) or keyframes)
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\"><title>PhoneLifeAgent Life Story</title>"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<style>"
        ":root{color-scheme:dark;--bg:#050505;--panel:#0d0d0f;--panel2:#151518;--text:#f4f4f5;--muted:#a1a1aa;--line:#2a2a2e;--accent:#7cc7ff}"
        "*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#1a2334 0,#050505 34rem);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;line-height:1.72}"
        ".shell{width:min(100%,980px);margin:0 auto;padding:18px 14px 44px}.phone-card{border:1px solid var(--line);border-radius:30px;background:linear-gradient(180deg,rgba(18,18,22,.96),rgba(3,3,4,.98));overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.46)}"
        ".hero{padding:26px 22px 8px}.collage{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px;border-radius:26px;background:#f6f1e9}.collage img{width:100%;height:176px;object-fit:cover;border-radius:20px;display:block}.collage img:nth-child(3){height:156px}.collage img:nth-child(4){height:156px}"
        ".hero-copy{padding:30px 6px 26px}.kicker{color:var(--accent);font-weight:700;font-size:14px;margin-bottom:14px}.title{font-size:38px;line-height:1.12;letter-spacing:0;margin:0 0 20px}.subtitle{font-size:22px;line-height:1.65;color:#d4d4d8;margin:0}.meta{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:14px;border-top:1px dashed var(--line);padding-top:18px;margin-top:26px}.dot{width:5px;height:5px;border-radius:50%;background:#3f3f46}"
        ".section{padding:30px 28px;border-top:1px solid rgba(255,255,255,.08)}.section h2{font-size:25px;line-height:1.25;margin:0 0 18px}.section h3{font-size:19px;line-height:1.35;margin:22px 0 10px}.section p,.section li{font-size:18px;color:#d7d7dc}.section p{margin:0 0 16px}.section ul,.section ol{padding-left:1.15rem;margin:0}.section li{margin:0 0 12px}.section strong{color:#fff}.muted{color:var(--muted)}"
        ".story-flow{display:flex;flex-direction:column;gap:28px}.story-moment{display:grid;grid-template-columns:minmax(0,1fr) minmax(260px,.88fr);gap:24px;align-items:center}.story-moment:nth-child(even){grid-template-columns:minmax(260px,.88fr) minmax(0,1fr)}.story-moment:nth-child(even) .story-copy{order:2}.story-moment:nth-child(even) .story-media{order:1}.story-copy h3{font-size:22px;margin:0 0 12px}.story-time{display:block;color:var(--accent);font-size:14px;font-weight:700;margin-bottom:4px}.story-copy p{font-size:18px;color:#d7d7dc;margin:0}.story-media img{width:100%;height:260px;object-fit:cover;display:block;border-radius:20px;border:1px solid var(--line)}"
        ".image-block{margin:22px 0 4px}.image-block img{width:100%;display:block;border-radius:18px;border:1px solid var(--line);object-fit:cover}.caption{font-size:13px;color:var(--muted);margin-top:8px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:18px}.gallery figure{margin:0}.gallery img{width:100%;height:150px;object-fit:cover;border-radius:16px;border:1px solid var(--line);display:block}.gallery figcaption{font-size:12px;line-height:1.4;color:var(--muted);margin-top:6px}"
        "@media (max-width:720px){.shell{width:min(100%,560px)}.story-moment,.story-moment:nth-child(even){display:block}.story-moment:nth-child(even) .story-copy,.story-moment:nth-child(even) .story-media{order:initial}.story-media{margin-top:16px}.story-media img{height:210px}}"
        "@media (max-width:480px){.shell{padding:0}.phone-card{border-radius:0;border-left:0;border-right:0}.title{font-size:34px}.subtitle{font-size:20px}.section{padding:26px 22px}.collage img{height:148px}.collage img:nth-child(3),.collage img:nth-child(4){height:132px}}"
        "</style></head><body><main class=\"shell\"><article class=\"phone-card\">"
        + "<header class=\"hero\">"
        + hero
        + "<div class=\"hero-copy\"><div class=\"kicker\">生活回放</div>"
        + f"<h1 class=\"title\">{html.escape(title)}</h1>"
        + f"<p class=\"subtitle\">{html.escape(subtitle)}</p>"
        + f"<div class=\"meta\"><span>PhoneLifeAgent</span><span class=\"dot\"></span><span>{html.escape(date_label)}</span></div>"
        + "</div></header>"
        + route_overview
        + body
        + "</article>"
        + "</main></body></html>\n"
    )


def _story_markdown_for_ui(markdown_text: str) -> str:
    lines = []
    skip_section = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        heading_level = _heading_level(line)
        heading = _heading_text(line)
        if heading and _is_hidden_ui_section(heading):
            skip_section = True
            continue
        if heading and heading_level <= 3:
            skip_section = False
        if skip_section:
            continue
        cleaned = _clean_ui_line(line)
        if cleaned or (lines and lines[-1]):
            lines.append(cleaned)
    return "\n".join(lines).strip()


def _story_title(markdown_text: str) -> str:
    first_h1 = ""
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            first_h1 = line[2:].strip()
            break
    if first_h1 and first_h1 not in GENERIC_STORY_TITLES:
        return first_h1
    return "今天的小片段"


def _story_subtitle(markdown_text: str) -> str:
    capture = False
    paragraphs = []
    seen_h1 = False
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            seen_h1 = True
            continue
        if seen_h1 and not paragraphs and stripped and not stripped.startswith("#"):
            paragraphs.append(stripped)
            break
        if stripped.startswith("## "):
            capture = "总览" in stripped or "Overview" in stripped
            continue
        if capture and stripped and not stripped.startswith("#") and not stripped.startswith("- "):
            paragraphs.append(stripped)
        if capture and len("".join(paragraphs)) > 130:
            break
    text = _clean_ui_line("".join(paragraphs)) or "把这一天里值得回看的片段，整理成一段可以慢慢读的生活记录。"
    return _one_line(_lively_text(text), 96)


def _story_date_label(evidence_pack: dict[str, Any]) -> str:
    start = evidence_pack.get("time_range", {}).get("start_local_time")
    if not start or start == "-":
        return "Life Story"
    return str(start).split(" ")[0]


def _hero_collage(keyframes: list[dict[str, Any]]) -> str:
    frames = [frame for frame in keyframes if frame.get("keyframe_path")][:4]
    if not frames:
        return ""
    images = []
    for frame in frames:
        caption = html.escape(_one_line(frame.get("caption") or frame.get("reason")))
        path = html.escape(_asset_src(str(frame.get("keyframe_path"))))
        images.append(f'<img src="{path}" alt="{caption}">')
    return '<div class="collage">' + "".join(images) + "</div>"


def _markdown_to_story_cards(markdown_text: str, route_map: str | None, keyframes: list[dict[str, Any]]) -> str:
    sections = _split_markdown_sections(markdown_text)
    cards = []
    inserted_gallery = False
    for title, lines in sections:
        if not title:
            continue
        if title.startswith("# "):
            intro_storyline = _storyline_candidate_lines(lines)
            if intro_storyline:
                content = _storyline_html(intro_storyline, keyframes=keyframes)
                if content.strip():
                    cards.append(f'<section class="section"><h2>故事线</h2>{content}</section>')
                    inserted_gallery = True
            continue
        heading = title[3:].strip() if title.startswith("## ") else title.lstrip("# ").strip()
        if heading in {"一天总览", "Overview"} or ("地点" in heading and "路线" in heading):
            continue
        if _is_story_ui_hidden_heading(heading):
            continue
        if "故事线" in heading:
            content = _storyline_html(lines, keyframes=keyframes)
            inserted_gallery = True
            if content.strip():
                cards.append(f'<section class="section"><h2>{html.escape(_display_heading(heading))}</h2>{content}</section>')
            continue
        content = _markdown_lines_to_html(_compact_section_lines(heading, lines))
        if content.strip():
            cards.append(f'<section class="section"><h2>{html.escape(_display_heading(heading))}</h2>{content}</section>')
    return "\n".join(cards)


def _display_heading(heading: str) -> str:
    clean = heading.strip()
    return STORY_UI_HEADING_MAP.get(clean, clean)


def _is_story_ui_hidden_heading(heading: str) -> bool:
    return any(token in heading for token in ["关键事件", "高光", "Highlight", "Key Events"])


def _storyline_candidate_lines(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if _bold_line_text(line.strip()):
            return lines[index:]
    return []


def _selected_storyline_keyframes(markdown_text: str, keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames_by_path = {str(frame.get("keyframe_path")): frame for frame in keyframes if frame.get("keyframe_path")}
    frames = list(frames_by_path.values())
    used_paths: set[str] = set()
    selected = []
    for lines in _storyline_line_groups(markdown_text):
        for moment in _storyline_moments(lines):
            media_path, _ = _storyline_media(moment, frames, used_paths, fallback_index=len(selected))
            if media_path and media_path in frames_by_path:
                used_paths.add(media_path)
                selected.append(frames_by_path[media_path])
    return selected


def _storyline_line_groups(markdown_text: str) -> list[list[str]]:
    groups = []
    for title, lines in _split_markdown_sections(markdown_text):
        if title.startswith("# "):
            candidate = _storyline_candidate_lines(lines)
            if candidate:
                groups.append(candidate)
            continue
        heading = title[3:].strip() if title.startswith("## ") else title.lstrip("# ").strip()
        if "故事线" in heading:
            groups.append(lines)
    return groups


def _storyline_html(lines: list[str], keyframes: list[dict[str, Any]]) -> str:
    moments = _storyline_moments(lines)
    if not moments:
        return _markdown_lines_to_html(lines)
    frames = [frame for frame in keyframes if frame.get("keyframe_path")]
    used_paths: set[str] = set()
    rendered = []
    for index, moment in enumerate(moments):
        media_path, media_caption = _storyline_media(moment, frames, used_paths, fallback_index=index)
        media = ""
        if media_path:
            used_paths.add(media_path)
            src = html.escape(_asset_src(media_path))
            media = (
                '<div class="story-media">'
                f'<img src="{src}" alt="{html.escape(media_caption)}">'
                "</div>"
            )
        rendered.append(
            '<div class="story-moment">'
            f'<div class="story-copy"><h3>{_storyline_heading_html(moment["title"])}</h3>{_markdown_lines_to_html(_compact_moment_body(moment["body"]))}</div>'
            + media
            + "</div>"
        )
    return '<div class="story-flow">' + "".join(rendered) + "</div>"


def _storyline_moments(lines: list[str]) -> list[dict[str, Any]]:
    moments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        title = _bold_line_text(line)
        if title:
            if current:
                moments.append(current)
            current = {"title": title, "body": []}
            continue
        if current is None:
            current = {"title": "片段", "body": []}
        current["body"].append(line)
    if current:
        moments.append(current)
    return moments


def _storyline_media(moment: dict[str, Any], frames: list[dict[str, Any]], used_paths: set[str], fallback_index: int) -> tuple[str | None, str]:
    if not frames:
        return None, ""
    start_second, end_second = _title_time_seconds(str(moment.get("title") or ""))
    if start_second is None or end_second is None:
        return None, ""
    candidates = _storyline_media_candidates(frames, used_paths, start_second, end_second, expansion_sec=0)
    if not candidates:
        candidates = _storyline_media_candidates(frames, used_paths, start_second, end_second, expansion_sec=20)
    if not candidates:
        candidates = _storyline_media_candidates(frames, used_paths, start_second, end_second, expansion_sec=180)
    if not candidates:
        candidates = _storyline_media_fallback_candidates(frames, used_paths)
    if not candidates:
        return None, ""
    midpoint = (start_second + end_second) / 2
    moment_text = str(moment.get("title") or "") + " " + " ".join(moment.get("body") or [])
    candidates.sort(
        key=lambda frame: (
            abs((_frame_local_second(frame) or midpoint) - midpoint) - _frame_quality(frame) * 24 - min(1.0, _frame_text_bonus(moment_text, frame) * 0.1),
            -_frame_quality(frame),
        )
    )
    frame = candidates[0]
    return str(frame.get("keyframe_path")), _one_line(frame.get("caption") or frame.get("reason"), 60)


def _storyline_media_fallback_candidates(frames: list[dict[str, Any]], used_paths: set[str]) -> list[dict[str, Any]]:
    candidates = []
    for frame in frames:
        path = str(frame.get("keyframe_path") or "")
        if path and path not in used_paths and frame.get("accepted", True) is not False:
            candidates.append(frame)
    return candidates


def _storyline_media_candidates(
    frames: list[dict[str, Any]],
    used_paths: set[str],
    start_second: int,
    end_second: int,
    expansion_sec: int,
) -> list[dict[str, Any]]:
    candidates = []
    for frame in frames:
        path = str(frame.get("keyframe_path") or "")
        if not path or path in used_paths or frame.get("accepted", True) is False:
            continue
        second = _frame_local_second(frame)
        if second is None:
            continue
        if start_second - expansion_sec <= second <= end_second + expansion_sec:
            candidates.append(frame)
    return candidates


def _route_overview_html(markdown_text: str, route_map: str | None) -> str:
    summary = _route_summary(markdown_text)
    if not summary and not route_map:
        return ""
    image_html = _image_block(route_map, "路线总览") if _valid_image_path(route_map) else '<p class="muted">路线图未生成，仍保留文字路线总结。</p>'
    return (
        '<section class="section route-overview"><h2>今天走到哪儿了</h2>'
        f'<p>{html.escape(summary)}</p>'
        + image_html
        + "</section>"
    )


def _route_summary(markdown_text: str) -> str:
    route_lines = []
    capture = False
    found_route_section = False
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = _heading_text(stripped)
            if capture:
                break
            capture = "地点" in heading and "路线" in heading
            found_route_section = found_route_section or capture
            continue
        if capture and stripped.startswith("- "):
            route_lines.append(re.sub(r"^- ", "", stripped))
        elif capture and stripped:
            route_lines.append(stripped)
    text = " ".join(_clean_ui_line(line) for line in route_lines[:3])
    if not found_route_section:
        return ""
    if not text:
        return "路线从出发点、途经地到归家点串成一条完整的小日子轨迹。"
    return _one_line(_lively_text(text), 120)


def _valid_image_path(path: str | None) -> bool:
    if not path:
        return False
    try:
        with Image.open(Path(path).expanduser()) as image:
            image.verify()
            return image.format in {"PNG", "JPEG"}
    except Exception:
        return False


def _compact_section_lines(heading: str, lines: list[str]) -> list[str]:
    max_items = 4
    if "TODO" in heading or "待办" in heading:
        max_items = 3
    if "记忆" in heading or "值得记住" in heading:
        max_items = 3
    compacted = []
    item_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if compacted and compacted[-1]:
                compacted.append("")
            continue
        if stripped.startswith("- ") or re.match(r"^\d+\.\s+", stripped):
            item_count += 1
            if item_count > max_items:
                continue
            compacted.append(_one_line(stripped, 105))
        else:
            compacted.append(_one_line(stripped, 115))
    return compacted


def _compact_moment_body(lines: list[str]) -> list[str]:
    text = _lively_text(" ".join(_clean_ui_line(line) for line in lines if line.strip()))
    if not text:
        return []
    return [_one_line(text, 118)]


def _title_time_seconds(title: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})", title)
    if not match:
        return None, None
    start = int(match.group(1)) * 3600 + int(match.group(2)) * 60
    end = int(match.group(3)) * 3600 + int(match.group(4)) * 60 + 59
    return start, end


def _frame_local_second(frame: dict[str, Any]) -> int | None:
    local_time = str(frame.get("local_time") or "")
    match = re.search(r"(\d{1,2}):(\d{2})(?::\d{2})?$", local_time)
    if not match:
        return None
    seconds_match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})$", local_time)
    seconds = int(seconds_match.group(3)) if seconds_match else 0
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + seconds


def _frame_quality(frame: dict[str, Any]) -> float:
    try:
        return float(frame.get("quality_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _frame_text_bonus(moment_text: str, frame: dict[str, Any]) -> float:
    frame_text = str(frame.get("caption") or "") + " " + str(frame.get("reason") or "")
    moment_tokens = _meaningful_text_tokens(moment_text)
    frame_tokens = _meaningful_text_tokens(frame_text)
    shared = moment_tokens & frame_tokens
    if not shared:
        return 0.0
    long_shared = [token for token in shared if len(token) >= 3]
    medium_shared = [token for token in shared if len(token) == 2]
    score = min(len(long_shared), 3) * 1.8 + min(len(medium_shared), 2) * 0.7
    return round(min(score, 6.0), 3)


def _meaningful_text_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(text)):
        clean = token.strip().lower()
        if not clean:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", clean):
            for size in (2, 3, 4):
                for index in range(0, max(0, len(clean) - size + 1)):
                    part = clean[index : index + size]
                    if part in FRAME_TEXT_STOPWORDS:
                        continue
                    tokens.add(part)
            continue
        if len(clean) < 2:
            continue
        if clean in FRAME_TEXT_STOPWORDS:
            continue
        tokens.add(clean)
    return tokens


def _lively_heading(text: str) -> str:
    text = re.sub(r"^\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}\s*\|\s*", "", text).strip()
    return text


def _storyline_heading_html(text: str) -> str:
    match = re.match(r"^(\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})\s*\|\s*(.+)$", str(text).strip())
    if not match:
        return _inline_markdown(_lively_heading(str(text)))
    time_range, title = match.groups()
    return f'<span class="story-time">{html.escape(time_range)}</span>{_inline_markdown(title.strip())}'


def _lively_text(text: str) -> str:
    replacements = {
        "进行了一次": "走了一段",
        "整体而言，": "",
        "显示为": "像是",
        "运动状态": "脚步",
        "完成": "做完",
        "场景转换": "换了个场景",
        "随遇而安且略带浪漫色彩": "松弛又有点浪漫",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _bold_line_text(line: str) -> str:
    match = re.fullmatch(r"\*\*(.+?)\*\*", line.strip())
    return match.group(1).strip() if match else ""


def _split_markdown_sections(markdown_text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
            if current_title or current_lines:
                sections.append((current_title, current_lines))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_title or current_lines:
        sections.append((current_title, current_lines))
    return sections


def _markdown_lines_to_html(lines: list[str]) -> str:
    rendered = []
    list_type = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if list_type:
                rendered.append(f"</{list_type}>")
                list_type = ""
            continue
        if line.startswith("### "):
            if list_type:
                rendered.append(f"</{list_type}>")
                list_type = ""
            rendered.append(f"<h3>{_inline_markdown(line[4:].strip())}</h3>")
        elif re.match(r"^\d+\.\s+", line):
            if list_type != "ol":
                if list_type:
                    rendered.append(f"</{list_type}>")
                rendered.append("<ol>")
                list_type = "ol"
            item_text = re.sub(r"^\d+\.\s+", "", line)
            rendered.append(f"<li>{_inline_markdown(item_text)}</li>")
        elif line.startswith("- "):
            if list_type != "ul":
                if list_type:
                    rendered.append(f"</{list_type}>")
                rendered.append("<ul>")
                list_type = "ul"
            rendered.append(f"<li>{_inline_markdown(line[2:].strip())}</li>")
        else:
            if list_type:
                rendered.append(f"</{list_type}>")
                list_type = ""
            rendered.append(f"<p>{_inline_markdown(line)}</p>")
    if list_type:
        rendered.append(f"</{list_type}>")
    return "\n".join(rendered)


def _image_block(path: str, caption: str) -> str:
    src = html.escape(_asset_src(path))
    return (
        '<div class="image-block">'
        f'<img src="{src}" alt="{html.escape(caption)}">'
        f'<div class="caption">{html.escape(caption)}</div>'
        "</div>"
    )


def _keyframe_gallery(keyframes: list[dict[str, Any]]) -> str:
    frames = [frame for frame in keyframes if frame.get("keyframe_path")]
    if not frames:
        return ""
    items = []
    for frame in frames:
        caption = html.escape(_one_line(frame.get("caption") or frame.get("reason"), 52))
        path = html.escape(_asset_src(str(frame.get("keyframe_path"))))
        items.append(f'<figure><img src="{path}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    return '<div class="gallery">' + "".join(items) + "</div>"


def _asset_src(path: str) -> str:
    if not path:
        return ""
    try:
        candidate = Path(path).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve().as_uri()
    except (OSError, ValueError):
        return path
    return path


def _heading_text(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return stripped.lstrip("#").strip()
    return ""


def _heading_level(line: str) -> int:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return 99
    return len(stripped) - len(stripped.lstrip("#"))


def _is_hidden_ui_section(heading: str) -> bool:
    lowered = heading.lower()
    hidden_exact = {
        "人物和对话",
        "物品",
        "情绪和状态",
        "开放问题",
        "people and conversations",
        "objects",
        "emotion and state",
        "open questions",
        "video evidence",
        "location & motion evidence",
        "location and motion evidence",
    }
    if lowered in hidden_exact:
        return True
    return any(token in lowered for token in ["证据", "evidence", "媒体索引", "media evidence", "source evidence"])


def _clean_ui_line(line: str) -> str:
    line = re.sub(r"\*\([^)]*(?:Audio|Clip|loc_seg|motion_seg|keyframe|route map)[^)]*\)\*", "", line)
    line = re.sub(r"\([^)]*(?:Audio|Clip|loc_seg|motion_seg|keyframe|route map)[^)]*\)", "", line)
    line = re.sub(r"（[^）]*(?:Audio|Clip|loc_seg|motion_seg|keyframe|route map)[^）]*）", "", line)
    line = re.sub(r"\[[^\]]*(?:Report Video|Keyframe|Map|Audio|Clip)[^\]]*\]\([^)]*\)", "", line)
    line = re.sub(r"\b(?:Audio|Clip)\s*\d+(?:/\d+)?\s*中的", "", line)
    line = re.sub(r"\b(?:Audio|Clip)\s*\d+(?:[-/]\d+)?\b", "", line)
    line = _replace_motion_codes(line)
    line = re.sub(r"`/[^`]+`", "", line)
    line = re.sub(r"`([^`]*(?:loc_seg|motion_seg|Audio|Clip|keyframe)[^`]*)`", "", line)
    line = line.replace("`", "")
    line = re.sub(r"/Users/[^\s)]+", "", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def _replace_motion_codes(line: str) -> str:
    replacements = {
        "walking_like": "步行",
        "phone_handling": "手持手机晃动",
        "steady_motion": "平稳移动",
        "stationary": "静止",
        "running_or_shaking": "剧烈晃动",
    }
    for code, label in replacements.items():
        line = line.replace(code, label)
    return line


def _inline_markdown(text: str) -> str:
    escaped = html.escape(_clean_ui_line(text))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def story_schema_rules() -> list[str]:
    return [
        "输入来自 audio、video、location、motion 的确定性证据包。",
        "请输出 markdown，不要输出 JSON。",
        "只输出这些章节：一天总览、故事线、关键事件、地点和路线、待办、值得记住。",
        "每个章节必须使用二级标题，例如：## 一天总览、## 故事线。",
        "一天总览不超过 90 个汉字。",
        "故事线控制在 4 到 6 段，每段 1 到 2 句，每段标题必须带时间，格式为：**14:10 - 14:15 | 生动短标题**。",
        "关键事件最多 5 条；待办最多 3 条；值得记住最多 3 条。",
        "人物、物品、情绪状态、开放问题、证据索引都作为背后信息理解，不要作为章节输出。",
    ]


def story_style_rules() -> list[str]:
    return [
        "目标是生成一份给本人阅读的生活小记，不是分析报告。",
        "文风要轻松、生动、有画面感，像生活杂志里的短篇回放；不要流水账，不要啰嗦。",
        "标题要具体、有情绪，避免“个人生活故事报告”这类模板标题。",
        "必须使用简体中文和第一人称生活小记文风，不要写“用户”“拍摄者”“父亲”。",
        "每个故事线段落至少保留一个自然的第一人称视角表达，例如“我看到”“我听到”“我记录到”“我在家里感受到”。",
    ]


def story_evidence_rules() -> list[str]:
    return [
        "不要在正文展示原始文件路径、audio_id、clip_id、loc_seg、keyframe path。",
        "严格区分第一人称事实、旁人对话、视觉猜测：明确是本人行为时写“我做了”；摄像头视角证据可以写“我看到/我记录到”；音频证据可以写“我听到”。",
        "听到旁人说话、背景声、邻居聊天时，写“旁边有人/家里有人/声音里听到”，不要写成我参与了。",
        "看到物品时，只能写可见物本身；不要推断用途、所有权、身份关系或后续行动。",
        "用户报告里不要保留“可能是/像是/疑似”的物品用途猜测；这类不确定信息留给背后问答。",
        "如果证据只是模型猜测，使用“像是/可能/不确定”，不要写成确定事实。",
        "地点和运动信息用于增强叙事，但遇到冲突或低置信度要写不确定。",
        "不要编造没有证据的人名、关系、地点、结论。",
    ]


def story_review_rules() -> list[str]:
    return [
        "输入包含一个 Life Story 草稿和原始证据包。",
        "你的任务是重写草稿，保留生活化文风，但删除或降级没有证据支持的第一人称结论。",
        "只有 audio/video/location/motion 明确支持的行为，才能写成“我做了”。",
        "摄像头视角可以写“我看到/我记录到”；音频证据可以写“我听到”。不要把这些全部改成无主语旁观句。",
        "旁人对话、背景音、路边喊话，默认写成“旁边听到/家里有人/声音里听到”，不能写成我参与。",
        "看到物品只能写可见物本身，不要推断用途、所有权、家庭关系或归属。",
        "最终用户报告不要展示物品用途猜测；如果只能写“像是/可能/疑似”，就删除猜测，只保留可见事实。",
        "如果某个 claim 找不到直接证据，就删除，或改成更保守的描述。",
        "最终文风仍然要像本人生活小记，故事线每段尽量保留“我看到/我听到/我记录到”等第一人称视角。",
        "输出 markdown，每个章节必须使用二级标题，章节保持：一天总览、故事线、关键事件、地点和路线、待办、值得记住。",
        "不要输出校验过程，不要输出 JSON，不要输出证据索引。",
    ]


def _story_system_prompt() -> str:
    parts = ["你是 PhoneLifeAgent 的最终 Life Story 生成模块。"]
    parts.extend(story_schema_rules())
    parts.extend(story_style_rules())
    parts.extend(story_evidence_rules())
    return "".join(parts)


def _claim_review_system_prompt() -> str:
    parts = ["你是 PhoneLifeAgent 的最终事实校验编辑。"]
    parts.extend(story_review_rules())
    return "".join(parts)


def _claim_review_user_prompt(draft: str, story_input: str) -> str:
    return (
        "Life Story Draft\n"
        "================\n"
        f"{draft.strip()}\n\n"
        "Evidence Pack Input\n"
        "===================\n"
        f"{_trim_text(story_input, 36000)}\n"
    )


def _timeline_record(source_type: str, source: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    start, end = _source_time_range(source)
    return {
        "source_type": source_type,
        "source_id": source.get("audio_id") or source.get("clip_id") or source.get("event_id"),
        "event_id": source.get("event_id"),
        "start_utc_sec": start,
        "end_utc_sec": end,
        "local_time_range": source.get("local_time_range") or _format_local_range(start, end),
        "summary": source.get("summary") or source.get("scene_summary") or source.get("clip_summary") or source.get("life_story_hint"),
        "event_type": source.get("event_type") or source.get("time_granularity"),
        "evidence_refs": source.get("evidence_refs", []),
        "context_brief": _context_brief(context),
    }


def _context_brief(context: dict[str, Any]) -> dict[str, Any]:
    location_segments = context.get("location", {}).get("segments", [])
    motion_segments = context.get("motion", {}).get("segments", [])
    return {
        "locations": [_location_label(item) for item in location_segments[:3]],
        "motions": [_motion_label(item) for item in motion_segments[:3]],
        "keyframes": [
            {
                "clip_id": item.get("clip_id"),
                "local_time": item.get("local_time"),
                "keyframe_path": item.get("keyframe_path"),
                "caption": item.get("caption"),
            }
            for item in context.get("video", {}).get("keyframes", [])[:3]
        ],
    }


def _location_label(segment: dict[str, Any]) -> str:
    amap = segment.get("amap") or {}
    places = []
    if amap.get("address"):
        places.append(str(amap.get("address")))
    roads = amap.get("roads") or []
    if roads:
        places.append("roads=" + ",".join(str(item) for item in roads[:3]))
    return f"{segment.get('start_local_time')}->{segment.get('end_local_time')} {segment.get('movement')} {segment.get('quality')} {'; '.join(places)}"


def _motion_label(segment: dict[str, Any]) -> str:
    return (
        f"{segment.get('start_local_time')}->{segment.get('end_local_time')} "
        f"{segment.get('state')} {segment.get('intensity')} {segment.get('stability')}"
    )


def _collect_candidates(store: ContextStore, key: str) -> list[dict[str, Any]]:
    candidates = []
    for item in store.audio_timeline.get(key, []):
        candidates.append({"source_type": "audio", "source_id": item.get("audio_id"), "candidate": item.get("candidate")})
    for item in store.video_timeline.get(key, []):
        candidates.append({"source_type": "video", "source_id": item.get("clip_id"), "candidate": item.get("candidate")})
    return candidates


def _timeline_line(item: dict[str, Any]) -> str:
    context = item.get("context_brief", {})
    locations = " | ".join(context.get("locations", []))
    motions = " | ".join(context.get("motions", []))
    return (
        f"- {item.get('local_time_range')} | {item.get('source_type')} {item.get('source_id')} | "
        f"{_one_line(item.get('event_type'))} | {_one_line(item.get('summary'))} | "
        f"location={_one_line(locations, 320)} | motion={_one_line(motions, 220)}"
    )


def _markdown_to_simple_html(markdown_text: str) -> str:
    lines = []
    in_list = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                lines.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h1>{html.escape(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h2>{html.escape(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h3>{html.escape(line[4:].strip())}</h3>")
        elif line.startswith("- "):
            if not in_list:
                lines.append("<ul>")
                in_list = True
            lines.append(f"<li>{html.escape(line[2:].strip())}</li>")
        else:
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        lines.append("</ul>")
    return "\n".join(lines)


def _source_time_range(source: dict[str, Any]) -> tuple[float | None, float | None]:
    start = _float_or_none(source.get("start_utc_sec") or source.get("absolute_start_utc_sec"))
    end = _float_or_none(source.get("end_utc_sec") or source.get("absolute_end_utc_sec"))
    if start is None and end is None:
        point = _float_or_none(source.get("absolute_utc_sec"))
        return point, point
    return start, end


def _format_local_range(start: float | None, end: float | None) -> str:
    return f"{_format_local_time(start)} - {_format_local_time(end)}"


def _format_local_time(utc_sec: Any) -> str:
    value = _float_or_none(utc_sec)
    if value is None:
        return "-"
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _trim_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 80].rstrip() + "\n...[trimmed for final story input]...\n" + text[-60:].lstrip()


def _one_line(value: Any, limit: int = 220) -> str:
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
