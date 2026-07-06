from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .model_engine import create_text_llm
from .pipeline_defaults import (
    DEFAULT_COMIC_IMAGE_MODEL,
    DEFAULT_COMIC_MAX_PANELS,
    DEFAULT_COMIC_MAX_REFERENCES_PER_PANEL,
    DEFAULT_COMIC_TEXT_MODEL,
    DEFAULT_COMIC_THINKING,
    comic_generation_policy,
)
from .settings_store import DEFAULT_DASHSCOPE_OPENAI_BASE_URL, apply_api_settings, load_api_settings


DEFAULT_COMIC_REFERENCE_COUNT = DEFAULT_COMIC_MAX_REFERENCES_PER_PANEL
DEFAULT_ARK_IMAGE_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"


def build_comic_products(
    run_dir: Path,
    output_dir: Path | None = None,
    provider: str = "aliyun",
    text_model: str = DEFAULT_COMIC_TEXT_MODEL,
    comic_thinking: bool | None = None,
    image_model: str = DEFAULT_COMIC_IMAGE_MODEL,
    max_reference_images: int | None = None,
    image_provider: str | None = None,
    max_panels: int = DEFAULT_COMIC_MAX_PANELS,
    comic_style: str = "daily_cartoon",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    def log(message: str) -> None:
        if progress:
            progress(message)

    run = run_dir.expanduser().resolve()
    output = (output_dir or run / "comic").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if provider == "aliyun":
        apply_api_settings(load_api_settings(Path.cwd()))

    log("Reading Story, evidence, and video media manifest...")
    story_markdown = _read_text(run / "story" / "life_story.md")
    story_json = _read_json(run / "story" / "life_story.json")
    evidence_pack = _read_json(run / "story" / "story_evidence_pack.json")
    media_manifest = _read_json(run / "video" / "products" / "video_story_media_manifest.json")

    log("Building Daily Comic storyline...")
    storyline = build_comic_storyline(
        story_markdown=story_markdown,
        story_json=story_json,
        evidence_pack=evidence_pack,
        media_manifest=media_manifest,
        provider=provider,
        text_model=text_model,
        comic_thinking=comic_thinking,
        max_panels=max_panels,
    )
    storyline_path = output / "comic_storyline.json"
    storyline_path.write_text(json.dumps(storyline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    storyboard = comic_storyboard_from_storyline(storyline, evidence_pack)
    storyboard_path = output / "comic_storyboard.json"
    storyboard_path.write_text(json.dumps(storyboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    panel_count = max(1, len(storyline.get("panels", [])))
    total_reference_limit = panel_count if not max_reference_images else max(1, min(panel_count, int(max_reference_images)))
    log(f"Selecting one reference keyframe for each comic panel ({panel_count} panel(s))...")
    reference_plan = build_comic_panel_reference_plan(
        storyline=storyline,
        evidence_pack=evidence_pack,
        media_manifest=media_manifest,
        provider=provider,
        text_model=text_model,
        comic_thinking=comic_thinking,
        max_references_per_panel=DEFAULT_COMIC_MAX_REFERENCES_PER_PANEL,
        max_total_references=total_reference_limit,
    )
    selected_refs = [Path(item["path"]) for item in reference_plan["selected_references"]]
    log(f"Normalizing {len(selected_refs)} reference image(s)...")
    normalized_refs = normalize_reference_images(selected_refs, output / "refs")
    for item, normalized_path in zip(reference_plan["selected_references"], normalized_refs):
        item["normalized_path"] = str(normalized_path)
    reference_plan_path = output / "comic_reference_plan.json"
    reference_plan_path.write_text(
        json.dumps(
            {
                "schema_version": "daily_comic_reference_plan.v3",
                "max_reference_images": total_reference_limit,
                "source_paths": [str(path) for path in selected_refs],
                "normalized_paths": [str(path) for path in normalized_refs],
                **reference_plan,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    selected_refs_path = output / "selected_references.json"
    selected_refs_path.write_text(reference_plan_path.read_text(encoding="utf-8"), encoding="utf-8")

    prompt = build_seedream_comic_prompt(storyline, reference_plan, comic_style=comic_style)
    prompt_path = output / "comic_image_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    seedream_prompt_path = output / "seedream_comic_prompt.txt"
    seedream_prompt_path.write_text(prompt, encoding="utf-8")

    draft_path = output / "daily_comic_reference_contact_sheet.png"
    log("Rendering reference contact sheet...")
    render_freeform_reference_sheet(normalized_refs, draft_path, storyline)
    panel_path = output / "daily_comic_panel.png"
    response_path = output / "daily_comic_image_response.json"
    resolved_image_provider = image_provider or ("ark" if provider == "aliyun" else "mock")
    if resolved_image_provider == "ark":
        log("Waiting for Seedream image generation...")
        image_response = generate_comic_panel_with_seedream(
            prompt=prompt,
            reference_images=normalized_refs,
            output_path=panel_path,
            response_path=response_path,
            image_model=image_model,
        )
    else:
        log("Rendering mock comic panel...")
        Image.open(draft_path).save(panel_path, quality=94)
        image_response = {
            "status": "mock",
            "mode": "seedream_reference_sheet_only",
            "draft_path": str(draft_path),
            "prompt_preview": prompt[:300],
        }
        response_path.write_text(json.dumps(image_response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    card_path = output / "daily_comic_card.png"
    log("Rendering storybook comic card...")
    render_comic_card(panel_path, storyboard, card_path, include_text=True)
    current_path = output / "daily_comic.png"
    Image.open(card_path).save(current_path, quality=94)
    html_path = output / "daily_comic.html"
    html_path.write_text(render_comic_html(card_path.name), encoding="utf-8")
    log("Daily Comic files written.")

    return {
        "comic_storyline_path": str(storyline_path),
        "comic_storyboard_path": str(storyboard_path),
        "comic_prompt_path": str(prompt_path),
        "seedream_comic_prompt_path": str(seedream_prompt_path),
        "comic_reference_plan_path": str(reference_plan_path),
        "selected_references_path": str(selected_refs_path),
        "layout_draft_path": str(draft_path),
        "daily_comic_panel_path": str(panel_path),
        "daily_comic_card_path": str(card_path),
        "daily_comic_path": str(current_path),
        "daily_comic_html_path": str(html_path),
        "provider": provider,
        "image_provider": resolved_image_provider,
        "text_model": text_model,
        "image_model": image_model,
        "panel_count": len(storyline.get("panels", [])),
        "reference_image_count": len(normalized_refs),
        "image_response": image_response,
    }


def build_comic_storyline(
    story_markdown: str,
    story_json: dict[str, Any] | None,
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
    provider: str,
    text_model: str,
    comic_thinking: bool | None = None,
    max_panels: int = DEFAULT_COMIC_MAX_PANELS,
) -> dict[str, Any]:
    story_json = story_json or {}
    max_panels = max(1, min(DEFAULT_COMIC_MAX_PANELS, int(max_panels or DEFAULT_COMIC_MAX_PANELS)))
    fallback = _fallback_comic_storyline(story_markdown, story_json, evidence_pack, media_manifest, max_panels=max_panels)
    if provider != "aliyun":
        return fallback
    try:
        resolved_thinking = DEFAULT_COMIC_THINKING if comic_thinking is None else comic_thinking
        policy = comic_generation_policy(
            provider=provider,
            text_model=text_model,
            max_panels=max_panels,
            enable_thinking=resolved_thinking,
        )
        llm = create_text_llm(policy.provider, model=policy.text_model, enable_thinking=policy.enable_thinking)
        response = llm.generate_text(
            _storyline_system_prompt(max_panels),
            _storyline_user_prompt(story_markdown, story_json, evidence_pack, media_manifest, max_panels=max_panels),
        )
        parsed = _parse_json_text(response)
    except Exception as exc:
        return {**fallback, "storyline_model_error": str(exc)}
    if not parsed:
        return {**fallback, "storyline_model_error": "failed_to_parse_json"}
    return _normalize_comic_storyline(parsed, fallback, max_panels=max_panels)


def comic_storyboard_from_storyline(storyline: dict[str, Any], evidence_pack: dict[str, Any]) -> dict[str, Any]:
    panels = storyline.get("panels", [])
    return {
        "schema_version": "daily_comic_storyboard.v2",
        "title": _short_title(str(storyline.get("title") or "今天的小片段")),
        "caption": _short_caption(str(storyline.get("caption") or storyline.get("storyline") or "")),
        "tag": _label_text(storyline.get("tag"), "生活漫画"),
        "date_label": _normalize_date_label(storyline.get("date_label") or _date_label(evidence_pack)),
        "storyline": _one_line(storyline.get("storyline")),
        "panels": [_one_line(item.get("story_beat") or item.get("visual_focus")) for item in panels if isinstance(item, dict)],
        "visual_anchors": _merge_unique(
            [
                anchor
                for item in panels
                if isinstance(item, dict)
                for anchor in _string_list(item.get("required_elements"))
            ],
            _string_list(storyline.get("visual_anchors")),
        )[:16],
        "forbidden": _string_list(storyline.get("forbidden"))[:8]
        or ["不添加与 Story 无关的生活事件", "不强化证据不足的人物关系", "不虚构具体对话"],
    }


def build_comic_panel_reference_plan(
    storyline: dict[str, Any],
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
    provider: str = "none",
    text_model: str = DEFAULT_COMIC_TEXT_MODEL,
    comic_thinking: bool | None = None,
    max_references_per_panel: int = DEFAULT_COMIC_MAX_REFERENCES_PER_PANEL,
    max_total_references: int | None = None,
) -> dict[str, Any]:
    frames = _unique_comic_candidate_frames(evidence_pack, media_manifest)
    panels = [item for item in storyline.get("panels", []) if isinstance(item, dict)]
    llm_choices = _llm_reference_choices(
        panels,
        frames,
        provider=provider,
        text_model=text_model,
        comic_thinking=comic_thinking,
    )
    used_paths: set[str] = set()
    panel_rows = []
    selected_refs = []
    panel_quotas = _panel_reference_quotas(panels, max_total_references, max_references_per_panel)
    for panel_index, panel in enumerate(panels):
        remaining = None if max_total_references is None else max(0, max_total_references - len(selected_refs))
        if remaining == 0:
            panel_rows.append(
                {
                    "panel_id": str(panel.get("panel_id")),
                    "story_beat": panel.get("story_beat"),
                    "visual_focus": panel.get("visual_focus"),
                    "references": [],
                    "missing": True,
                    "reason": "已达到总参考图上限。",
                }
            )
            continue
        panels_left_after = max(0, len(panels) - panel_index - 1)
        if remaining is None:
            allowed_for_panel = max_references_per_panel
        else:
            reserved_for_later = min(panels_left_after, remaining)
            allowed_for_panel = max(1, remaining - reserved_for_later)
        chosen = _select_panel_reference_frames(
            panel,
            frames,
            used_paths=used_paths,
            llm_choice_ids=llm_choices.get(str(panel.get("panel_id")), []),
            max_count=min(panel_quotas[panel_index], allowed_for_panel) if remaining is not None else max_references_per_panel,
        )
        row_refs = []
        panel_role = _panel_role(panel, panel_index, len(panels))
        for frame, match in chosen:
            path = str(frame.get("keyframe_path") or "")
            if not path:
                continue
            used_paths.add(path)
            item = {
                "panel_id": str(panel.get("panel_id")),
                "panel_role": panel_role,
                "path": path,
                "clip_id": str(frame.get("clip_id") or ""),
                "keyframe_id": frame.get("keyframe_id"),
                "local_time": frame.get("local_time"),
                "quality_score": frame.get("quality_score"),
                "score": round(float(match.get("score") or 0.0), 4),
                "match_features": match.get("match_features", []),
                "score_breakdown": match.get("score_breakdown", {}),
                "reason": match.get("reason", ""),
                "clip_summary": _one_line(frame.get("clip_summary")),
            }
            selected_refs.append(item)
            row_refs.append(item)
        panel_rows.append(
            {
                "panel_id": str(panel.get("panel_id")),
                "panel_role": panel_role,
                "story_beat": panel.get("story_beat"),
                "visual_focus": panel.get("visual_focus"),
                "references": row_refs,
                "missing": not row_refs,
            }
        )
    return {
        "selection_strategy": "storyline_panel_reference.v3",
        "panel_count": len(panels),
        "candidate_count": len(frames),
        "panels": panel_rows,
        "selected_references": selected_refs,
        "missing_panel_ids": [row["panel_id"] for row in panel_rows if row["missing"]],
    }


def _panel_reference_quotas(
    panels: list[dict[str, Any]],
    max_total_references: int | None,
    max_references_per_panel: int,
) -> list[int]:
    if not panels:
        return []
    if max_total_references is None:
        return [max_references_per_panel for _ in panels]
    total = max(1, max_total_references)
    base = max(1, min(max_references_per_panel, total // len(panels)))
    quotas = [base for _ in panels]
    remaining = max(0, total - sum(quotas))
    priority = sorted(
        range(len(panels)),
        key=lambda index: (
            len(_string_list(panels[index].get("required_elements"))),
            _panel_role(panels[index], index, len(panels)) in {"action_peak", "closing"},
        ),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for index in priority:
            if remaining <= 0:
                break
            if quotas[index] >= max_references_per_panel:
                continue
            quotas[index] += 1
            remaining -= 1
            changed = True
        if not changed:
            break
    return quotas


def _fallback_comic_storyline(
    story_markdown: str,
    story_json: dict[str, Any],
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
    max_panels: int,
) -> dict[str, Any]:
    story_text = story_markdown or str(story_json.get("report_markdown") or "")
    title = _first_heading(story_text) or "今天的小片段"
    overview = _story_overview(story_text) or _first_body_sentence(story_text)
    beats = _storyline_beats_from_markdown(story_text)
    beats = _expand_multi_scene_beats(beats)
    candidate_count = len(beats)
    if not beats:
        beats = [_one_line(overview) or "一天中的代表性片段"]
    selected_beats = _limit_storyline_beats(beats, max_panels=max_panels)
    frames = media_manifest.get("selected_keyframes", []) or evidence_pack.get("media", {}).get("selected_keyframes", [])
    panels = []
    for index, beat in enumerate(selected_beats, start=1):
        panel_id = f"panel_{index:02d}"
        panels.append(
            {
                "panel_id": panel_id,
                "order": index,
                "time_hint": _time_hint_from_text(beat),
                "story_beat": _one_line(beat),
                "visual_focus": _visual_focus_from_text(beat),
                "required_elements": _required_elements_from_text(beat),
                "avoid": ["不要添加没有证据的新事件", "不要把路人关系画成亲密关系"],
                "reference_queries": _reference_queries_from_text(beat),
                "importance_reason": "来自最终 Life Story 的时间线或关键事件。",
            }
        )
    return {
        "schema_version": "daily_comic_storyline.v2",
        "title": _short_title(title),
        "caption": _short_caption(overview),
        "tag": "生活漫画",
        "date_label": _date_label(evidence_pack),
        "storyline": _one_line(overview),
        "max_panels": max_panels,
        "candidate_panel_count": candidate_count or len(panels),
        "panel_count": len(panels),
        "selection_reason": "fallback 从最终 Story 的时间线/关键事件抽取；超过上限时保留起承转合和结尾。",
        "visual_anchors": _visual_anchors_from_media(media_manifest)[:12],
        "forbidden": ["不添加与 Story 无关的生活事件", "不强化证据不足的人物关系", "不虚构具体对话"],
        "panels": panels,
        "available_keyframe_count": len(frames),
    }


def _normalize_comic_storyline(parsed: dict[str, Any], fallback: dict[str, Any], max_panels: int) -> dict[str, Any]:
    panels = parsed.get("panels")
    normalized_panels = []
    if isinstance(panels, list):
        for index, item in enumerate(panels, start=1):
            if not isinstance(item, dict):
                item = {"story_beat": _one_line(item)}
            story_beat = _one_line(item.get("story_beat") or item.get("description") or item.get("caption") or item.get("text"))
            if not story_beat:
                continue
            normalized_panels.append(
                {
                    "panel_id": str(item.get("panel_id") or f"panel_{index:02d}"),
                    "order": int(item.get("order") or index),
                    "time_hint": _one_line(item.get("time_hint")),
                    "story_beat": story_beat,
                    "visual_focus": _one_line(item.get("visual_focus") or story_beat),
                    "required_elements": _string_list(item.get("required_elements"))[:8],
                    "avoid": _string_list(item.get("avoid"))[:6],
                    "reference_queries": _string_list(item.get("reference_queries"))[:8],
                    "importance_reason": _one_line(item.get("importance_reason")),
                }
            )
    if not normalized_panels:
        normalized_panels = fallback["panels"]
    normalized_panels = _expand_storyline_panels(normalized_panels, max_panels=max_panels)[:max_panels]
    for index, item in enumerate(normalized_panels, start=1):
        item["panel_id"] = f"panel_{index:02d}"
        item["order"] = index
        if not item.get("required_elements"):
            item["required_elements"] = _required_elements_from_text(item["story_beat"])
        if not item.get("reference_queries"):
            item["reference_queries"] = _reference_queries_from_text(item["story_beat"])
        if not item.get("avoid"):
            item["avoid"] = ["不要添加没有证据的新事件"]
    result = {
        **fallback,
        **{key: value for key, value in parsed.items() if key != "panels" and value},
        "schema_version": "daily_comic_storyline.v2",
        "max_panels": max_panels,
        "panel_count": len(normalized_panels),
        "panels": normalized_panels,
    }
    result["title"] = _short_title(str(result.get("title") or fallback["title"]))
    result["caption"] = _short_caption(str(result.get("caption") or fallback["caption"]))
    result["tag"] = _label_text(result.get("tag"), "生活漫画")
    result["date_label"] = _normalize_date_label(result.get("date_label") or fallback.get("date_label"))
    result["storyline"] = _one_line(result.get("storyline") or fallback.get("storyline"))
    result["forbidden"] = _merge_unique(_string_list(result.get("forbidden")), fallback.get("forbidden", []))[:8]
    return result


def _expand_storyline_panels(panels: list[dict[str, Any]], max_panels: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for panel in panels:
        if len(expanded) >= max_panels:
            break
        remaining_capacity = max_panels - len(expanded)
        split_panels = _split_panel_on_scene_transition(panel) if remaining_capacity >= 2 else []
        if split_panels:
            expanded.extend(split_panels[:remaining_capacity])
        else:
            expanded.append(panel)
    return expanded


def _split_panel_on_scene_transition(panel: dict[str, Any]) -> list[dict[str, Any]]:
    beat = _one_line(panel.get("story_beat"))
    if not beat:
        return []
    parts = _split_scene_transition_text(beat)
    if len(parts) < 2:
        return []
    primary, secondary = parts[0], parts[1]
    if not _has_scene_transition(primary, secondary):
        return []
    return [
        {
            **panel,
            "story_beat": primary,
            "visual_focus": _visual_focus_from_text(primary),
            "required_elements": _required_elements_from_text(primary),
            "reference_queries": _reference_queries_from_text(primary),
        },
        {
            **panel,
            "story_beat": secondary,
            "visual_focus": _visual_focus_from_text(secondary),
            "required_elements": _required_elements_from_text(secondary),
            "reference_queries": _reference_queries_from_text(secondary),
        },
    ]


def _split_scene_transition_text(text: str) -> list[str]:
    for pattern in [r"(.+?后，)(.+)", r"(.+?之后，)(.+)", r"(.+?随后)(.+)", r"(.+?然后)(.+)", r"(.+?最后)(.+)"]:
        match = re.match(pattern, text)
        if not match:
            continue
        first = _one_line(match.group(1))
        second = _one_line(match.group(2))
        if first and second:
            return [first, second]
    return [text]


def _has_scene_transition(first: str, second: str) -> bool:
    group = {"home_indoor", "store_indoor", "building_entry", "outdoor_residential", "park_walk", "intersection"}
    first_tags = _scene_type_tags(first) & group
    second_tags = _scene_type_tags(second) & group
    if not first_tags or not second_tags:
        return False
    return first_tags != second_tags


def _storyline_system_prompt(max_panels: int) -> str:
    return (
        "你是 PhoneLifeAgent 的 Daily Comic 分镜导演。"
        "根据最终 Life Story 设计一张漫画页的分镜，不要编造新事件。"
        f"分镜数量由故事内容决定，最多 {max_panels} 个；如果候选超过上限，筛掉重复或视觉弱的片段。"
        "优先保留故事转折、视觉差异、事实重要节点和收尾。"
        "必须使用简体中文输出 JSON，不要 markdown，不要英文叙述。"
        "story_beat、visual_focus、caption 要保持第一人称生活记录视角；可以写“我看到/我记录到”，不要写“用户/拍摄者/father/daughter/the child”。"
        "字段：title、caption、tag、date_label、storyline、candidate_panel_count、selection_reason、forbidden、panels。"
        "caption 是最终漫画下方的故事书式短文，60-100 个中文字符，必须覆盖主要转折和结尾。"
        "panels 每项字段：time_hint、story_beat、visual_focus、required_elements、avoid、reference_queries、importance_reason。"
    )


def _storyline_user_prompt(
    story_markdown: str,
    story_json: dict[str, Any],
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
    max_panels: int,
) -> str:
    media_lines = []
    for frame in media_manifest.get("selected_keyframes", [])[:24]:
        media_lines.append(
            f"- keyframe={frame.get('keyframe_id')} clip={frame.get('clip_id')} time={frame.get('local_time')} summary={_one_line(frame.get('clip_summary'))}"
        )
    return (
        f"分镜上限：{max_panels}\n\n"
        "Final Life Story:\n"
        f"{_trim_text(story_markdown, 9000)}\n\n"
        "Life Story metadata:\n"
        f"{json.dumps({key: story_json.get(key) for key in ['time_range', 'source_counts', 'story_model']}, ensure_ascii=False)}\n\n"
        "Evidence time range:\n"
        f"{json.dumps(evidence_pack.get('time_range', {}), ensure_ascii=False)}\n\n"
        "Visual evidence candidates:\n"
        + "\n".join(media_lines)
        + "\n\n要求：分镜要像一天的漫画故事，不要把证据里不确定的东西升级成事实。"
    )


def _storyline_beats_from_markdown(markdown_text: str) -> list[str]:
    sections = _split_markdown_sections_simple(markdown_text)
    timeline_beats = []
    event_beats = []
    for heading, lines in sections:
        if heading in {"按时间的故事线", "故事线"}:
            timeline_beats.extend(_beats_from_lines(lines))
        elif heading in {"关键事件"}:
            event_beats.extend(_beats_from_lines(lines))
    if timeline_beats:
        return timeline_beats
    if event_beats:
        return event_beats
    overview_beats = []
    for heading, lines in sections:
        if heading in {"一天总览", "总览", "今日总览"}:
            overview_beats.extend(_beats_from_lines(lines))
    return overview_beats


def _split_markdown_sections_simple(markdown_text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    lines: list[str] = []
    for raw in markdown_text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            if heading or lines:
                sections.append((heading, lines))
            heading = line.lstrip("#").strip()
            lines = []
        elif line:
            lines.append(line)
    if heading or lines:
        sections.append((heading, lines))
    return sections


def _beats_from_lines(lines: list[str]) -> list[str]:
    beats = []
    current_title = ""
    current_body: list[str] = []
    for line in lines:
        if line.startswith("**") and "|" in line:
            if current_title or current_body:
                beats.append(_one_line(f"{current_title} {' '.join(current_body)}"))
            current_title = re.sub(r"^\*\*|\*\*$", "", line).strip()
            current_body = []
        elif line.startswith("- "):
            beats.append(line[2:].strip())
        elif current_title:
            current_body.append(line)
        else:
            parts = re.split(r"(?<=[。！？])", line)
            beats.extend([part.strip() for part in parts if part.strip()])
    if current_title or current_body:
        beats.append(_one_line(f"{current_title} {' '.join(current_body)}"))
    return [item for item in beats if item]


def _limit_storyline_beats(beats: list[str], max_panels: int) -> list[str]:
    if len(beats) <= max_panels:
        return beats
    if max_panels <= 2:
        return beats[:max_panels]
    selected = [beats[0], beats[-1]]
    middle = beats[1:-1]
    slots = max_panels - 2
    if slots > 0 and middle:
        step = max(1, len(middle) / slots)
        for index in range(slots):
            selected.insert(-1, middle[min(len(middle) - 1, int(index * step))])
    return selected[:max_panels]


def _expand_multi_scene_beats(beats: list[str]) -> list[str]:
    expanded = []
    for beat in beats:
        text = _one_line(beat)
        split_match = re.search(r"(随后|然后|接着|最后)", text)
        if split_match and len(text) >= 90:
            marker = split_match.group(1)
            first, second = text.split(marker, 1)
            if len(first) >= 28 and len(second) >= 18:
                expanded.append(first.strip())
                time_hint = _time_hint_from_text(text)
                prefix = f"{time_hint} | " if time_hint else ""
                expanded.append(prefix + marker + second.strip())
                continue
        expanded.append(text)
    return expanded


def _time_hint_from_text(text: str) -> str:
    match = re.search(r"\b(\d{1,2}:\d{2})(?:\s*[-~—]\s*(\d{1,2}:\d{2}))?", text)
    if not match:
        return ""
    return " - ".join(part for part in match.groups() if part)


def _visual_focus_from_text(text: str) -> str:
    clean = re.sub(r"\*\*", "", text).strip()
    clean = re.sub(r"^\d{1,2}:\d{2}\s*[-~—]\s*\d{1,2}:\d{2}\s*\|\s*", "", clean)
    return _short_caption(clean)


def _required_elements_from_text(text: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_]{3,}", text)
    stop = {
        "这个",
        "故事",
        "时间",
        "随后",
        "然后",
        "接着",
        "最后",
        "直到",
        "觉得",
        "一段",
        "一天",
        "画面",
        "分镜",
        "片段",
        "主角",
        "自己",
    }
    elements = []
    for term in terms:
        if term in stop or re.fullmatch(r"\d{1,2}:\d{2}", term) or term in elements:
            continue
        elements.append(term)
    return elements[:8]


def _reference_queries_from_text(text: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_]{3,}", text)
    stop = {"这个", "故事", "时间", "随后", "直到", "觉得", "一段", "一天", "画面"}
    queries = []
    for term in terms:
        if term in stop or term in queries:
            continue
        queries.append(term)
    return queries[:8]


def _unique_comic_candidate_frames(evidence_pack: dict[str, Any], media_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    frames = []
    frames.extend(media_manifest.get("selected_keyframes", []))
    frames.extend(evidence_pack.get("media", {}).get("selected_keyframes", []))
    seen: set[str] = set()
    unique = []
    for frame in frames:
        path = str(frame.get("keyframe_path") or "")
        if not path or path in seen or not Path(path).exists():
            continue
        seen.add(path)
        unique.append(frame)
    return unique


def _llm_reference_choices(
    panels: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    provider: str,
    text_model: str,
    comic_thinking: bool | None = None,
) -> dict[str, list[str]]:
    if provider != "aliyun" or not panels or not frames:
        return {}
    try:
        resolved_thinking = DEFAULT_COMIC_THINKING if comic_thinking is None else comic_thinking
        policy = comic_generation_policy(provider=provider, text_model=text_model, enable_thinking=resolved_thinking)
        llm = create_text_llm(policy.provider, model=policy.text_model, enable_thinking=policy.enable_thinking)
        response = llm.generate_text(_reference_selector_system_prompt(), _reference_selector_user_prompt(panels, frames))
        parsed = _parse_json_text(response)
    except Exception:
        return {}
    choices: dict[str, list[str]] = {}
    items = parsed.get("selections") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return choices
    for item in items:
        if not isinstance(item, dict):
            continue
        panel_id = str(item.get("panel_id") or "")
        keyframe_ids = _string_list(item.get("keyframe_ids"))
        if panel_id and keyframe_ids:
            choices[panel_id] = keyframe_ids[:1]
    return choices


def _reference_selector_system_prompt() -> str:
    return (
        "你是 Daily Comic 的关键帧选择器。"
        "为每个漫画分镜从真实 keyframes 中精确选择 1 张最能支持事实的参考图。"
        "必须综合判断分镜语义、时间接近、视觉元素匹配、故事代表性、场景边界和与其他分镜的差异。"
        "要理解 opening、transition、action_peak、closeup、environment、closing 这类分镜角色，不要按固定地点词表硬匹配。"
        "相邻但不同空间的场景不能混选；纯物件特写、室内收尾、室外过渡、天气高潮都要区分开。"
        "只输出简体中文 JSON，字段 selections；每项包含 panel_id、keyframe_ids、reason。"
        "keyframe_ids 每项只放 1 个 id。"
    )


def _reference_selector_user_prompt(panels: list[dict[str, Any]], frames: list[dict[str, Any]]) -> str:
    panel_rows = [
        {
            "panel_id": panel.get("panel_id"),
            "time_hint": panel.get("time_hint"),
            "story_beat": panel.get("story_beat"),
            "visual_focus": panel.get("visual_focus"),
            "required_elements": panel.get("required_elements"),
            "reference_queries": panel.get("reference_queries"),
        }
        for panel in panels
    ]
    frame_rows = [
        {
            "keyframe_id": frame.get("keyframe_id"),
            "clip_id": frame.get("clip_id"),
            "local_time": frame.get("local_time"),
            "quality_score": frame.get("quality_score"),
            "summary": _one_line(frame.get("clip_summary") or frame.get("caption") or frame.get("selection_reason")),
        }
        for frame in frames[:32]
    ]
    return json.dumps({"panels": panel_rows, "keyframes": frame_rows}, ensure_ascii=False, indent=2)


def _select_panel_reference_frames(
    panel: dict[str, Any],
    frames: list[dict[str, Any]],
    used_paths: set[str],
    llm_choice_ids: list[str],
    max_count: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_keyframe = {str(frame.get("keyframe_id") or ""): frame for frame in frames}
    panel_role = _scene_role_from_text(_panel_search_text(panel), 1, 3)
    for keyframe_id in llm_choice_ids:
        frame = by_keyframe.get(str(keyframe_id))
        if not frame:
            continue
        path = str(frame.get("keyframe_path") or "")
        if not path or path in used_paths:
            continue
        return [
            (
                frame,
                {
                    "score": 100.0,
                    "match_features": ["llm_choice", "role_match"],
                    "score_breakdown": {
                        "llm_priority": 100.0,
                        "quality": round(float(frame.get("quality_score") or 0.0), 4),
                    },
                    "reason": "LLM 精确选择：最匹配该分镜的真实参考图。",
                    "panel_role": panel_role,
                },
            )
        ]

    scored = []
    panel_text = _panel_search_text(panel)
    panel_time = _minutes_from_time_hint(str(panel.get("time_hint") or ""))
    panel_scene_tags = _scene_type_tags(panel_text)
    panel_required_text = " ".join(_string_list(panel.get("required_elements")))
    panel_query_text = " ".join(_string_list(panel.get("reference_queries")))
    for frame in frames:
        path = str(frame.get("keyframe_path") or "")
        if not path or path in used_paths:
            continue
        frame_text = _frame_search_text(frame)
        frame_scene_tags = _scene_type_tags(frame_text)
        overlap = _keyword_overlap_score(panel_text, frame_text)
        required_overlap = _keyword_overlap_score(panel_required_text, frame_text) * 2.0
        query_overlap = _keyword_overlap_score(panel_query_text, frame_text)
        role_score = _scene_tag_match_score(panel_scene_tags, frame_scene_tags)
        quality_score = float(frame.get("quality_score") or 0)
        score = quality_score + overlap + required_overlap + query_overlap + role_score
        time_score = 0.0
        frame_time = _minutes_from_local_time(str(frame.get("local_time") or ""))
        if panel_time is not None and frame_time is not None:
            time_score = max(0.0, 1.0 - abs(panel_time - frame_time) / 12.0)
            score += time_score
        matched_terms = [
            term
            for term in _merge_unique(_string_list(panel.get("required_elements")), _string_list(panel.get("reference_queries")))
            if term and term.lower() in frame_text
        ][:6]
        scored.append(
            (
                score,
                frame,
                {
                    "score": round(score, 4),
                    "match_features": [
                        feature
                        for feature, active in [
                            ("time_proximity", time_score > 0),
                            ("role_match", role_score > 0),
                            ("visual_keyword_overlap", overlap > 0 or required_overlap > 0 or query_overlap > 0),
                            ("quality", quality_score > 0),
                        ]
                        if active
                    ],
                    "score_breakdown": {
                        "quality": round(quality_score, 4),
                        "keyword_overlap": round(overlap, 4),
                        "required_elements": round(required_overlap, 4),
                        "reference_queries": round(query_overlap, 4),
                        "role_match": round(role_score, 4),
                        "time_proximity": round(time_score, 4),
                    },
                    "reason": (
                        f"片段角色={panel_role}；"
                        f"命中元素={', '.join(matched_terms) if matched_terms else '弱'}；"
                        f"时间接近度={time_score:.2f}；画质={quality_score:.2f}。"
                    ),
                    "panel_role": panel_role,
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _, frame, match in scored[:max_count]:
        selected.append((frame, match))
        break
    if selected:
        return selected
    return []


def _required_element_score(required_elements: list[str], frame_text: str) -> float:
    return _keyword_overlap_score(" ".join(required_elements), frame_text) * 2.0


def _panel_search_text(panel: dict[str, Any]) -> str:
    return " ".join(
        [
            str(panel.get("time_hint") or ""),
            str(panel.get("story_beat") or ""),
            str(panel.get("visual_focus") or ""),
            " ".join(_string_list(panel.get("required_elements"))),
            " ".join(_string_list(panel.get("reference_queries"))),
        ]
    ).lower()


def _keyword_overlap_score(left: str, right: str) -> float:
    terms = [term for term in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_]{3,}", left) if len(term) >= 2]
    score = 0.0
    seen = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        if term.lower() in right:
            score += 1.2
    return score


def _scene_type_tags(text: str) -> set[str]:
    lowered = text.lower()
    tags: set[str] = set()
    if any(word in lowered for word in ["室内", "家中", "屋里", "回家", "收尾", "整理", "门厅", "走廊", "入口内侧", "鞋柜", "储物"]):
        tags.add("home_indoor")
    if any(word in lowered for word in ["店内", "商店", "超市", "餐馆", "柜台", "货架", "商品区", "冷柜", "收银", "室内公共空间"]):
        tags.add("store_indoor")
    if any(word in lowered for word in ["门口", "门外", "入口", "出口", "大厅", "楼门", "楼栋", "站台", "车站", "过渡区域", "电梯厅"]):
        tags.add("building_entry")
    if any(word in lowered for word in ["路口", "斑马线", "信号灯", "路牌", "十字路口", "地下通道", "道路交汇", "街口"]):
        tags.add("intersection")
    if any(word in lowered for word in ["草坪", "树木", "树荫", "绿化", "步道", "小径", "河边", "广场", "户外空地", "绿地"]):
        tags.add("park_walk")
    if any(word in lowered for word in ["雨伞", "伞面", "雨滴", "撑伞", "积水", "暴雨", "大雨", "冰雹", "雨声", "降水", "雪", "风"]):
        tags.add("rain_action")
    if any(word in lowered for word in ["特写", "手里", "手持", "局部", "近景", "近距离"]):
        tags.add("detail_closeup")
    if any(word in lowered for word in ["居民楼", "道路", "街道", "人行道", "电动车", "自行车", "住宅", "车内", "驾驶", "方向盘", "通勤路线"]):
        tags.add("outdoor_residential")
    return tags


def _panel_role(panel: dict[str, Any], index: int, total: int) -> str:
    return _scene_role_from_text(_panel_search_text(panel), index, total)


def _scene_role_from_text(text: str, index: int, total: int) -> str:
    lowered = (text or "").lower()
    if index == 0:
        return "opening"
    if index == total - 1:
        return "closing"
    if any(word in lowered for word in ["突然", "暴雨", "冰雹", "冲进", "折返", "告别", "高潮", "转折"]):
        return "action_peak"
    if any(word in lowered for word in ["特写", "局部", "近景", "手里", "雨滴", "物件", "细节"]):
        return "closeup"
    if any(word in lowered for word in ["路上", "经过", "前往", "返回", "穿过", "走向", "过渡", "路口", "门口"]):
        return "transition"
    if any(word in lowered for word in ["街道", "客厅", "房间", "车内", "环境", "场景", "室内", "户外"]):
        return "environment"
    return "transition"


def _scene_role_match_score(panel_role: str, frame_tags: set[str]) -> float:
    role_map = {
        "opening": {"outdoor_residential", "building_entry", "environment"},
        "transition": {"outdoor_residential", "building_entry", "intersection", "park_walk"},
        "action_peak": {"rain_action", "intersection", "building_entry"},
        "closeup": {"detail_closeup", "rain_action", "home_indoor", "store_indoor"},
        "environment": {"park_walk", "outdoor_residential", "store_indoor", "home_indoor"},
        "closing": {"home_indoor", "building_entry", "store_indoor"},
    }
    targets = role_map.get(panel_role, set())
    return 1.6 if frame_tags & targets else 0.0


def _storyboard_time_score(fragment_text: str, frame_local_time: str) -> float:
    fragment_time = _minutes_from_time_hint(fragment_text)
    frame_time = _minutes_from_local_time(frame_local_time)
    if fragment_time is None or frame_time is None:
        return 0.0
    return max(0.0, 1.0 - abs(fragment_time - frame_time) / 12.0)


def _scene_tag_match_score(panel_tags: set[str], frame_tags: set[str]) -> float:
    if not panel_tags or not frame_tags:
        return 0.0
    shared = panel_tags & frame_tags
    score = len(shared) * 1.6
    conflicting_groups = [
        {"home_indoor", "store_indoor", "building_entry", "outdoor_residential", "park_walk", "intersection"},
        {"detail_closeup", "home_indoor", "store_indoor", "building_entry", "outdoor_residential", "park_walk", "intersection"},
    ]
    for group in conflicting_groups:
        panel_group = panel_tags & group
        frame_group = frame_tags & group
        if panel_group and frame_group and not shared.intersection(group):
            score -= 2.4
    if "detail_closeup" in panel_tags and "detail_closeup" not in frame_tags:
        score -= 1.2
    return score


def _minutes_from_time_hint(value: str) -> int | None:
    match = re.search(r"(\d{1,2}):(\d{2})", value)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _minutes_from_local_time(value: str) -> int | None:
    match = re.search(r"(\d{1,2}):(\d{2})(?::\d{2})?", value)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def build_comic_storyboard(
    story_markdown: str,
    story_json: dict[str, Any] | None,
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
    provider: str,
    text_model: str,
) -> dict[str, Any]:
    story_json = story_json or {}
    fallback = _fallback_storyboard(story_markdown, story_json, evidence_pack, media_manifest)
    if provider != "aliyun":
        return fallback
    policy = comic_generation_policy(provider=provider, text_model=text_model)
    llm = create_text_llm(policy.provider, model=policy.text_model, enable_thinking=policy.enable_thinking)
    response = llm.generate_text(_storyboard_system_prompt(), _storyboard_user_prompt(story_markdown, story_json, evidence_pack, media_manifest))
    parsed = _parse_json_text(response)
    if not parsed:
        return {**fallback, "storyboard_model_error": "failed_to_parse_json", "storyboard_model_text": response[:2000]}
    return _normalize_storyboard(parsed, fallback)


def select_comic_reference_images(
    storyboard: dict[str, Any],
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
    max_count: int = DEFAULT_COMIC_MAX_PANELS,
) -> list[Path]:
    plan = build_comic_reference_selection(storyboard, evidence_pack, media_manifest, max_count=max_count)
    return [Path(item["path"]) for item in plan["selected_references"]]


def build_comic_reference_selection(
    storyboard: dict[str, Any],
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
    max_count: int = DEFAULT_COMIC_MAX_PANELS,
) -> dict[str, Any]:
    frames = _unique_comic_candidate_frames(evidence_pack, media_manifest)
    storyboard_fragments = _storyboard_scene_fragments(storyboard, max_count=max_count)
    selected: list[dict[str, Any]] = []
    selection_rows = []
    used_paths: set[str] = set()
    used_clips: set[str] = set()
    for index, fragment in enumerate(storyboard_fragments):
        match = _best_storyboard_frame_match(fragment, frames, used_paths, used_clips, prefer_unused_clip=True)
        if not match:
            match = _best_storyboard_frame_match(fragment, frames, used_paths, used_clips, prefer_unused_clip=False)
        if not match:
            selection_rows.append(
                {
                    "fragment_id": fragment["fragment_id"],
                    "panel_role": fragment["panel_role"],
                    "status": "missing",
                    "reason": "没有找到足够贴近该故事片段的关键帧。",
                }
            )
            continue
        frame = match["frame"]
        path = str(frame.get("keyframe_path") or "")
        used_paths.add(path)
        used_clips.add(str(frame.get("clip_id") or ""))
        row = {
            "fragment_id": fragment["fragment_id"],
            "panel_role": fragment["panel_role"],
            "path": path,
            "clip_id": str(frame.get("clip_id") or ""),
            "keyframe_id": frame.get("keyframe_id"),
            "local_time": frame.get("local_time"),
            "quality_score": frame.get("quality_score"),
            "score": round(float(match["score"]), 4),
            "match_features": match["match_features"],
            "score_breakdown": match["score_breakdown"],
            "reason": match["reason"],
            "clip_summary": _one_line(frame.get("clip_summary") or frame.get("caption") or frame.get("selection_reason")),
        }
        selected.append(row)
        selection_rows.append({**row, "status": "selected"})
        if len(selected) >= max_count:
            break

    return {
        "selection_strategy": "storyboard_fragment_coverage.v2",
        "candidate_count": len(frames),
        "fragment_count": len(storyboard_fragments),
        "fragments": storyboard_fragments,
        "fragment_selections": selection_rows,
        "selected_references": selected,
        "missing_fragments": [item["fragment_id"] for item in selection_rows if item.get("status") == "missing"],
    }


def _frame_search_text(frame: dict[str, Any]) -> str:
    return " ".join(
        str(frame.get(key) or "")
        for key in [
            "clip_summary",
            "caption",
            "reason",
            "selection_reason",
            "local_time",
            "source_video_path",
            "report_video_path",
        ]
    ).lower()


def _storyboard_scene_fragments(storyboard: dict[str, Any], max_count: int) -> list[dict[str, Any]]:
    panel_texts = _string_list(storyboard.get("panels"))
    fragments = []
    total = max(1, min(max_count, len(panel_texts) or max_count))
    for index, panel_text in enumerate(panel_texts[:total]):
        fragments.append(
            {
                "fragment_id": f"fragment_{index + 1:02d}",
                "panel_role": _scene_role_from_text(panel_text, index, total),
                "text": panel_text,
                "visual_anchors": _string_list(storyboard.get("visual_anchors")),
                "query_terms": _storyboard_terms({"panels": [panel_text], "visual_anchors": storyboard.get("visual_anchors", [])}),
            }
        )
    if not fragments:
        text = _comic_story_text(storyboard)
        fragments.append(
            {
                "fragment_id": "fragment_01",
                "panel_role": "opening",
                "text": text,
                "visual_anchors": _string_list(storyboard.get("visual_anchors")),
                "query_terms": _storyboard_terms(storyboard),
            }
        )
    return fragments


def _best_storyboard_frame_match(
    fragment: dict[str, Any],
    frames: list[dict[str, Any]],
    used_paths: set[str],
    used_clips: set[str],
    prefer_unused_clip: bool,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    for frame in frames:
        path = str(frame.get("keyframe_path") or "")
        clip_id = str(frame.get("clip_id") or "")
        if not path or path in used_paths:
            continue
        if prefer_unused_clip and clip_id in used_clips:
            continue
        match = _score_storyboard_frame(fragment, frame, clip_penalty=clip_id in used_clips)
        if match["score"] > best_score:
            best = match
            best_score = float(match["score"])
    return best


def _score_storyboard_frame(fragment: dict[str, Any], frame: dict[str, Any], clip_penalty: bool) -> dict[str, Any]:
    fragment_text = str(fragment.get("text") or "")
    frame_text = _frame_search_text(frame)
    panel_role = str(fragment.get("panel_role") or "transition")
    query_terms = _string_list(fragment.get("query_terms"))
    anchor_terms = _string_list(fragment.get("visual_anchors"))
    keyword_overlap = _keyword_overlap_score(" ".join(query_terms), frame_text)
    anchor_overlap = _keyword_overlap_score(" ".join(anchor_terms), frame_text)
    role_score = _scene_role_match_score(panel_role, _scene_type_tags(frame_text))
    time_score = _storyboard_time_score(fragment_text, str(frame.get("local_time") or ""))
    quality_score = float(frame.get("quality_score") or 0)
    uniqueness_penalty = 0.45 if clip_penalty else 0.0
    total = quality_score + keyword_overlap + anchor_overlap * 0.8 + role_score + time_score - uniqueness_penalty
    matched_terms = [term for term in query_terms if term and term.lower() in frame_text][:6]
    features = []
    if time_score > 0:
        features.append("time_proximity")
    if role_score > 0:
        features.append("role_match")
    if keyword_overlap > 0:
        features.append("visual_keyword_overlap")
    if anchor_overlap > 0:
        features.append("anchor_overlap")
    if quality_score > 0:
        features.append("quality")
    reason = (
        f"片段角色={panel_role}；关键词命中={', '.join(matched_terms) if matched_terms else '弱'}；"
        f"时间接近度={time_score:.2f}；画质={quality_score:.2f}。"
    )
    return {
        "frame": frame,
        "score": round(total, 4),
        "match_features": features,
        "score_breakdown": {
            "quality": round(quality_score, 4),
            "keyword_overlap": round(keyword_overlap, 4),
            "anchor_overlap": round(anchor_overlap, 4),
            "role_match": round(role_score, 4),
            "time_proximity": round(time_score, 4),
            "clip_diversity_penalty": round(uniqueness_penalty, 4),
        },
        "reason": reason,
    }


def _comic_story_text(storyboard: dict[str, Any]) -> str:
    return " ".join(
        [
            str(storyboard.get("title") or ""),
            str(storyboard.get("caption") or ""),
            str(storyboard.get("storyline") or ""),
            " ".join(_string_list(storyboard.get("panels"))),
            " ".join(_string_list(storyboard.get("visual_anchors"))),
        ]
    )


def _comic_visual_targets(storyboard: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            str(storyboard.get("storyline") or ""),
            " ".join(_string_list(storyboard.get("panels"))),
            " ".join(_string_list(storyboard.get("visual_anchors"))),
        ]
    )
    candidates = []
    if any(word in text for word in ["伞", "撑伞", "雨伞"]):
        candidates.append("umbrella")
    if any(word in text for word in ["店内", "商店", "超市", "货架", "冷柜", "柜台", "室内公共空间"]):
        candidates.append("retail_interior")
    if any(word in text for word in ["雨", "积水", "湿滑", "雨声", "冰雹", "雹"]):
        candidates.append("precipitation")
    if any(word in text for word in ["路口", "路牌", "信号灯", "马路", "斑马线", "街口", "道路"]):
        candidates.append("street_transition")
    if any(word in text for word in ["冰雹", "雹"]):
        candidates.append("hail")
    if any(word in text for word in ["室内", "回家", "整理", "包袋", "衣物", "储物", "鞋柜", "门厅"]):
        candidates.append("home_return")
    if any(word in text for word in ["树", "绿化", "步道", "草地", "户外空地", "广场"]):
        candidates.append("green_outdoor")
    ordered = ["umbrella", "retail_interior", "precipitation", "street_transition", "home_return", "hail", "green_outdoor"]
    return [item for item in ordered if item in candidates]


def _best_frame_for_target(
    scored: list[tuple[float, str, Path, str, dict[str, Any]]],
    target: str,
    selected: list[Path],
    used_clips: set[str],
) -> tuple[float, str, Path, str, dict[str, Any]] | None:
    best = None
    best_score = -1.0
    for score, clip_id, path, text, frame in scored:
        if path in selected:
            continue
        if clip_id in used_clips:
            continue
        target_score = _target_match_score(target, text)
        if target_score <= 0:
            continue
        combined = score + target_score
        if combined > best_score:
            best = (score, clip_id, path, text, frame)
            best_score = combined
    return best


def _target_match_score(target: str, text: str) -> float:
    keywords = {
        "umbrella": ["雨伞", "撑伞", "折叠伞", "伞面"],
        "retail_interior": ["店内", "商店", "超市", "货架", "冷柜", "柜台", "室内公共空间"],
        "precipitation": ["雨天", "雨声", "大雨", "积水", "湿滑", "雨滴", "冰雹", "雹"],
        "street_transition": ["路口", "路牌", "信号灯", "马路", "斑马线", "道路", "车辆", "街口"],
        "hail": ["冰雹", "雹"],
        "home_return": ["室内", "回家", "储物", "衣物", "包袋", "鞋柜", "门厅", "整理"],
        "green_outdoor": ["小径", "草地", "树木", "绿化", "步道", "广场"],
    }
    score = 0.0
    for keyword in keywords.get(target, []):
        if keyword in text:
            score += 2.0
    return score


def normalize_reference_images(paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = []
    for index, path in enumerate(paths, start=1):
        image = Image.open(path).convert("RGB")
        if image.height > image.width:
            image = image.rotate(-90, expand=True)
        image.thumbnail((1536, 1536))
        output_path = output_dir / f"ref_{index:02d}.jpg"
        image.save(output_path, quality=90, optimize=True)
        normalized.append(output_path)
    return normalized


def build_comic_image_prompt(storyboard: dict[str, Any]) -> str:
    scene_fragments = "；".join(str(item) for item in storyboard.get("panels", [])[:8])
    visual_anchors = "、".join(str(item) for item in storyboard.get("visual_anchors", [])[:10])
    fact_boundaries = "、".join(str(item) for item in storyboard.get("forbidden", [])[:10])
    lines = [
        "生成一张正方形的生活漫画场景拼图，像一天里的几个记忆碎片自然组合成一张完整画面。",
        "画面可以错落、叠放、留白、局部遮挡或轻微融合，由你根据故事自由安排构图。",
        "参考图用于理解真实经历里的场景、物品、天气、道路、室内外氛围。请把参考内容重新绘制成统一漫画风格，减少照片贴片感。",
        f"故事主线：{storyboard.get('storyline', '')}",
        f"生活片段：{scene_fragments}",
        f"真实视觉锚点：{visual_anchors}",
        "风格：温暖生活漫画、干净手绘线条、柔和色彩、轻松手账感、细节清楚，像私人日记里的生活碎片合集。",
        "人物可以自然出现，但不要强化证据不足的人物关系；画面里的标识、招牌和路牌可抽象成色块或符号，避免抢走生活氛围。",
        f"事实边界：{fact_boundaries}。这些信息只用于避免过度想象，画面仍然保持自然和有趣。",
    ]
    return "\n".join(lines)


def build_layout_refine_prompt(storyboard: dict[str, Any]) -> str:
    scene_fragments = "；".join(str(item) for item in storyboard.get("panels", [])[:8])
    visual_anchors = "、".join(str(item) for item in storyboard.get("visual_anchors", [])[:10])
    fact_boundaries = "、".join(str(item) for item in storyboard.get("forbidden", [])[:6])
    return (
        "参考输入图中的 6 个生活片段，重新设计一张完整的生活漫画拼图。\n"
        "输入图只是素材索引板，不是最终排版；请重新安排版式，让格子大小有主次变化，不要照抄 3x2 方格。\n"
        "保留每个片段的主要内容和事实关系，可以调整每个片段的位置、大小和裁切，让画面更像完成度高的生活漫画页。\n"
        "整体铺满正方形画布，没有大块空白；分隔线清楚自然，画面不要出现说明文字。\n"
        "每个区域要有不同内容，不要把不同素材合并成同一种街景；保留室内外、物品、道路、雨伞、店内等差异。\n"
        "请统一重绘成温暖、干净、轻松的手绘漫画风格。\n"
        f"故事主线：{storyboard.get('storyline', '')}\n"
        f"生活片段：{scene_fragments}\n"
        f"真实视觉锚点：{visual_anchors}\n"
        f"事实边界：{fact_boundaries}。这些只用于避免过度想象，最终画面要自然、有生活感。"
    )


def build_seedream_comic_prompt(storyline: dict[str, Any], reference_plan: dict[str, Any], comic_style: str = "daily_cartoon") -> str:
    panels = []
    refs_by_panel = {
        str(row.get("panel_id")): row.get("references", [])
        for row in reference_plan.get("panels", [])
        if isinstance(row, dict)
    }
    for panel in storyline.get("panels", []):
        if not isinstance(panel, dict):
            continue
        panel_id = str(panel.get("panel_id"))
        ref_summaries = [
            _one_line(item.get("clip_summary") or item.get("reason"))
            for item in refs_by_panel.get(panel_id, [])
            if isinstance(item, dict)
        ]
        panels.append(
            f"{panel.get('order')}. {panel.get('story_beat')} | 画面重点：{panel.get('visual_focus')} | "
            f"真实元素：{'、'.join(_string_list(panel.get('required_elements')))} | "
            f"对应参考图摘要：{'；'.join(ref_summaries[:1])}"
        )
    layout_guidance = _comic_page_layout_guidance(storyline)
    hero_refs, support_refs = _hero_and_support_panel_summaries(storyline, panels)
    forbidden = "、".join(_string_list(storyline.get("forbidden"))[:8])
    style_line = {
        "daily_cartoon": "忠于参考图的半写实漫画风格迁移：保留原图构图、镜头视角、透视关系、人物比例、物体数量和真实空间结构；只把照片质感转换成清晰手绘线条、柔和色块和轻微漫画化光影。人物不要 Q 版、不要大眼、不要夸张表情；背景不要童话化或动画化，保持真实生活场景，只减少噪点和杂乱纹理。",
    }.get(comic_style, comic_style)
    prompt_lines = []
    prompt_lines.extend(comic_schema_rules())
    prompt_lines.extend(comic_style_rules(style_line, layout_guidance))
    prompt_lines.extend(comic_evidence_rules())
    prompt_lines.extend(
        [
            f"标题：{storyline.get('title')}",
            f"故事主线：{storyline.get('storyline')}",
            f"事实边界：{forbidden}",
            "主参考图：",
            *hero_refs,
            "",
            "辅助参考图：",
            *support_refs,
            "",
            "分镜内容：",
            *panels,
        ]
    )
    return "\n".join(prompt_lines)


def comic_schema_rules() -> list[str]:
    return [
        "请根据输入的真实参考图，生成一张正方形 Daily Comic 漫画故事页。这是一整张完成图，不是素材板，也不是联系表。",
        "每个分镜必须是独立场景，所有关键情节都要全部画出来，不要跳过主线转折。",
        "分镜数量按故事自然决定，不必等于参考图数量。",
        "不要画成规则九宫格、平均网格、整齐表格、海报拼贴或 contact sheet。",
    ]


def comic_style_rules(style_line: str, layout_guidance: str) -> list[str]:
    return [
        "请把分镜自然组织成一张活泼但克制的故事书漫画页：整体要像真正排版过的漫画页。",
        "分镜尺寸要有明显主次：至少 1 个较大的主镜头，2 到 3 个中等镜头，其余可以是窄长、小幅或不规则辅助镜头。",
        "允许少量斜线切分、弧线边框或错落拼接，让版式更灵活，但边界仍要清楚可读。",
        f"页面布局建议：{layout_guidance}",
        f"漫画风格：{style_line}",
        "主角人物请接近真实生活照片里的普通人：真人比例、自然身高、自然手脚和衣着，五官只做极轻微漫画化。",
        "背景请做轻度风格迁移：用手绘线条和概括色块替代照片噪点，但保留真实地点结构、天气氛围和关键物件。",
    ]


def comic_evidence_rules() -> list[str]:
    return [
        "请优先忠于每张参考图：保留原始镜头角度、画面裁切、主要物体位置，以及道路、室内陈设、出入口、建筑等空间关系，不要重新设计成全新场景。",
        "不同参考图如果对应不同空间或镜头，就分别画成不同分镜；不要合并成同一个连续大画面。",
        "可以把同一段连续动作的参考图融合成一个更强的单一分镜，但必须保持同一空间和同一情节段落。",
        "人物只画参考图中确实出现或故事分镜明确需要的人；不要新增陌生人、围观者、男女主角或额外家庭成员。",
        "必须保留每个分镜对应参考图中的真实场景和物品关系；没有证据的对话、人物关系、地点招牌、人物数量不要新增。",
        "画面内完全不要生成任何文字：不要标题、不要分镜标签、不要对白气泡、不要拟声词、不要字幕、不要时间戳、不要水印、不要二维码、不要模型签名。",
        "真实世界中的路牌、招牌、商品包装、支付提示、屏幕内容也不要生成可读文字，请抽象成不可识别的色块、模糊线条或简单符号。",
        "剧情只能通过画面、人物动作、天气、物品和场景关系表达，不要用文字解释。",
    ]


def _comic_page_layout_guidance(storyline: dict[str, Any]) -> str:
    panels = [item for item in storyline.get("panels", []) if isinstance(item, dict)]
    if not panels:
        return "使用 1 个较大的主分镜带动节奏，其余分镜大小有变化，避免整齐平均。"
    opening = panels[0]
    ending = panels[-1]
    climax = max(
        panels,
        key=lambda item: (
            any(word in _one_line(item.get("story_beat")) for word in ["冰雹", "暴雨", "突然", "转折", "冲进", "避险"]),
            len(_string_list(item.get("required_elements"))),
        ),
    )
    return (
        f"建议把第 {climax.get('order')} 格作为较大的高潮分镜；"
        f"第 {opening.get('order')} 格和第 {ending.get('order')} 格可用中等大小承担开场与收尾；"
        "其余分镜穿插成宽窄不同的辅助镜头，至少出现一条非完全水平垂直的分割边，但整体仍然清楚易读。"
    )


def _hero_and_support_panel_summaries(storyline: dict[str, Any], panel_lines: list[str]) -> tuple[list[str], list[str]]:
    raw_panels = [item for item in storyline.get("panels", []) if isinstance(item, dict)]
    if not raw_panels:
        return (["- 无"], ["- 无"])
    climax = max(
        raw_panels,
        key=lambda item: (
            any(word in _one_line(item.get("story_beat")) for word in ["冰雹", "暴雨", "突然", "转折", "冲进", "避险"]),
            len(_string_list(item.get("required_elements"))),
        ),
    )
    picks = []
    seen = set()
    for candidate in [raw_panels[0], climax, raw_panels[-1]]:
        panel_id = str(candidate.get("panel_id"))
        if panel_id in seen:
            continue
        seen.add(panel_id)
        picks.append(panel_id)
    hero_refs = [f"- {line}" for line in panel_lines if line.split(".", 1)[0] and any(line.startswith(f"{raw.get('order')}.") for raw in raw_panels if str(raw.get("panel_id")) in picks)]
    support_refs = [f"- {line}" for line in panel_lines if line not in [item[2:] for item in hero_refs]]
    return (hero_refs or ["- 无"], support_refs or ["- 无"])


def generate_comic_panel_with_seedream(
    prompt: str,
    reference_images: list[Path],
    output_path: Path,
    response_path: Path,
    image_model: str = DEFAULT_COMIC_IMAGE_MODEL,
) -> dict[str, Any]:
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("VOLCENGINE_API_KEY") or os.environ.get("SEEDREAM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY for Seedream comic image generation.")
    if not reference_images:
        raise RuntimeError("At least one reference image is required for Seedream comic image generation.")
    endpoint = _ark_image_endpoint()
    images = []
    for image_path in reference_images[:18]:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        images.append(f"data:image/jpeg;base64,{encoded}")
    payload = {
        "model": image_model,
        "prompt": prompt,
        "image": images,
        "size": "2K",
        "response_format": "url",
        "sequential_image_generation": "disabled",
        "watermark": False,
    }
    started = time.time()
    response = requests.post(
        endpoint,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=300,
    )
    elapsed = round(time.time() - started, 3)
    data = _response_json(response)
    response_path.write_text(
        json.dumps({"elapsed_sec": elapsed, "endpoint": endpoint, **_redact_response(data)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Seedream comic image generation failed: {response.status_code} {json.dumps(data, ensure_ascii=False)[:500]}")
    image_url = _extract_seedream_image_url(data)
    if not image_url:
        raise RuntimeError("Seedream comic image response did not include an image URL.")
    if image_url.startswith("data:image/"):
        output_path.write_bytes(base64.b64decode(image_url.split(",", 1)[1]))
    else:
        image_response = requests.get(image_url, timeout=180)
        image_response.raise_for_status()
        output_path.write_bytes(image_response.content)
    return {"status": "done", "elapsed_sec": elapsed, "usage": data.get("usage", {}), "image_model": image_model}


def render_freeform_reference_sheet(frame_paths: list[Path], output_path: Path, storyline: dict[str, Any] | None = None) -> None:
    if not frame_paths:
        raise RuntimeError("No reference frames available for comic reference sheet rendering.")
    frames = [Image.open(path).convert("RGB") for path in frame_paths[:9]]
    width = height = 2048
    canvas = Image.new("RGB", (width, height), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, height), fill=(246, 241, 231))
    rects = _freeform_sheet_rects(len(frames))
    for index, (image, rect) in enumerate(zip(frames, rects), start=1):
        _paste_cover(canvas, image, rect, radius=24, border=0)
        x0, y0, _, _ = rect
        draw.rounded_rectangle((x0 + 16, y0 + 16, x0 + 70, y0 + 70), radius=14, fill=(255, 255, 255), outline=(28, 28, 28), width=3)
        draw.text((x0 + 32, y0 + 25), str(index), fill=(20, 20, 20), font=_font(28))
    title = _short_title(str((storyline or {}).get("title") or "Daily Comic"))
    draw.text((52, 1978), title, fill=(40, 40, 40), font=_font(44))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def _freeform_sheet_rects(count: int) -> list[tuple[int, int, int, int]]:
    layouts = {
        1: [(120, 120, 1928, 1928)],
        2: [(80, 80, 1968, 980), (220, 1040, 1828, 1960)],
        3: [(60, 80, 1370, 920), (980, 820, 1988, 1570), (120, 1340, 1080, 1988)],
        4: [(60, 70, 1160, 760), (1080, 180, 1988, 1040), (120, 900, 980, 1900), (940, 1160, 1968, 1988)],
        5: [(60, 70, 1190, 690), (1150, 120, 1988, 880), (80, 760, 900, 1500), (820, 820, 1660, 1530), (220, 1490, 1880, 1988)],
        6: [(60, 70, 1120, 640), (1040, 110, 1988, 780), (80, 710, 760, 1430), (700, 760, 1390, 1470), (1320, 830, 1988, 1520), (180, 1500, 1900, 1988)],
        7: [(60, 70, 1020, 600), (960, 90, 1988, 700), (80, 650, 720, 1300), (650, 710, 1320, 1370), (1250, 740, 1988, 1390), (120, 1350, 980, 1988), (900, 1420, 1968, 1988)],
        8: [(60, 70, 940, 570), (880, 90, 1540, 690), (1480, 120, 1988, 760), (80, 640, 760, 1220), (700, 720, 1320, 1340), (1260, 800, 1988, 1420), (120, 1320, 1010, 1988), (940, 1440, 1968, 1988)],
        9: [(60, 70, 860, 560), (800, 90, 1440, 650), (1370, 120, 1988, 760), (80, 620, 700, 1160), (650, 700, 1260, 1300), (1200, 780, 1988, 1350), (120, 1250, 760, 1900), (710, 1340, 1360, 1988), (1300, 1420, 1968, 1988)],
    }
    return layouts[min(max(count, 1), 9)][:count]


def _extract_seedream_image_url(data: dict[str, Any]) -> str | None:
    items = data.get("data")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                if item.get("url"):
                    return str(item["url"])
                if item.get("b64_json"):
                    return _write_data_url_from_b64(str(item["b64_json"]))
    return _extract_image_url(data)


def _ark_image_endpoint() -> str:
    endpoint = os.environ.get("ARK_IMAGE_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/")
    base = os.environ.get("ARK_BASE_URL")
    if not base:
        return DEFAULT_ARK_IMAGE_ENDPOINT
    base = base.rstrip("/")
    if base.endswith("/images/generations"):
        return base
    return base + "/images/generations"


def _write_data_url_from_b64(value: str) -> str:
    return f"data:image/png;base64,{value}"


def _redact_response(data: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(data))
    for item in redacted.get("data", []) if isinstance(redacted.get("data"), list) else []:
        if isinstance(item, dict) and item.get("url"):
            item["url"] = str(item["url"])[:80] + "...[redacted]"
        if isinstance(item, dict) and item.get("b64_json"):
            item["b64_json"] = "[redacted]"
    return redacted


def generate_comic_panel_with_aliyun(
    prompt: str,
    reference_images: list[Path],
    output_path: Path,
    response_path: Path,
    image_model: str,
) -> dict[str, Any]:
    settings = _load_settings()
    if not settings.dashscope_api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY for comic image generation.")
    if not reference_images:
        raise RuntimeError("At least one reference image is required for wan image generation.")

    api_base = (settings.dashscope_openai_base_url or DEFAULT_DASHSCOPE_OPENAI_BASE_URL).replace("/compatible-mode/v1", "/api/v1").rstrip("/")
    url = f"{api_base}/services/aigc/multimodal-generation/generation"
    content = [{"text": prompt}]
    for image_path in reference_images[:9]:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content.append({"image": f"data:image/jpeg;base64,{encoded}"})
    payload = {
        "model": image_model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": {
            "enable_sequential": False,
            "size": "2K",
            "n": 1,
            "watermark": False,
            "seed": int(time.time()) % 2_147_483_647,
        },
    }
    started = time.time()
    response = requests.post(
        url,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings.dashscope_api_key}"},
        json=payload,
        timeout=240,
    )
    elapsed = round(time.time() - started, 3)
    data = _response_json(response)
    response_path.write_text(json.dumps({"elapsed_sec": elapsed, **data}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if response.status_code >= 400:
        raise RuntimeError(f"Aliyun comic image generation failed: {response.status_code} {json.dumps(data, ensure_ascii=False)[:500]}")
    image_url = _extract_image_url(data)
    if not image_url:
        raise RuntimeError("Aliyun comic image response did not include an image URL.")
    image_response = requests.get(image_url, timeout=120)
    image_response.raise_for_status()
    output_path.write_bytes(image_response.content)
    return {"status": "done", "elapsed_sec": elapsed, "usage": data.get("usage", {})}


def cartoonize_reference_images_with_aliyun(
    storyboard: dict[str, Any],
    reference_images: list[Path],
    output_dir: Path,
    response_path: Path,
    image_model: str,
) -> dict[str, Any]:
    if not reference_images:
        raise RuntimeError("At least one reference image is required for comic frame generation.")
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = []
    frame_paths = []
    total_elapsed = 0.0
    for index, image_path in enumerate(reference_images, start=1):
        output_path = output_dir / f"frame_{index:02d}.png"
        meta_path = output_dir / f"frame_{index:02d}.json"
        source_hash = _file_hash(image_path)
        if output_path.exists() and meta_path.exists():
            meta = _read_json(meta_path)
            if meta.get("source_hash") == source_hash and meta.get("image_model") == image_model:
                frame_paths.append(str(output_path))
                responses.append({"index": index, "status": "skipped", "path": str(output_path)})
                continue

        prompt = build_frame_cartoon_prompt(storyboard, index)
        response = generate_single_cartoon_frame_with_aliyun(
            prompt=prompt,
            reference_image=image_path,
            output_path=output_path,
            image_model=image_model,
        )
        meta = {
            "index": index,
            "source_path": str(image_path),
            "source_hash": source_hash,
            "output_path": str(output_path),
            "image_model": image_model,
            "prompt": prompt,
            **response,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        responses.append(meta)
        frame_paths.append(str(output_path))
        total_elapsed += float(response.get("elapsed_sec") or 0)

    result = {
        "status": "done",
        "mode": "cartoonize_frames_then_collage",
        "image_model": image_model,
        "frame_count": len(frame_paths),
        "frame_paths": frame_paths,
        "elapsed_sec": round(total_elapsed, 3),
        "frames": responses,
    }
    response_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def build_frame_cartoon_prompt(storyboard: dict[str, Any], index: int) -> str:
    visual_anchors = "、".join(str(item) for item in storyboard.get("visual_anchors", [])[:8])
    fact_boundaries = "、".join(str(item) for item in storyboard.get("forbidden", [])[:5])
    return (
        "把这张真实关键帧重新绘制成温暖、干净的生活漫画插画。\n"
        "保留原图里的主要构图、场景关系、重要物品和天气氛围；把照片质感转成统一手绘线条和柔和色彩。\n"
        "画面内完全不要生成任何文字、标题、标签、对白气泡、拟声词、字幕、水印或签名。招牌、路牌、屏幕文字请抽象成色块、模糊线条或简单符号。\n"
        "不要新增和原图无关的事件，人物关系保持克制自然。\n"
        f"这一组漫画来自同一天的故事：{storyboard.get('storyline', '')}\n"
        f"整体视觉锚点：{visual_anchors}\n"
        f"事实边界：{fact_boundaries}\n"
        f"当前是第 {index} 张漫画化关键帧，请保证它可以和其他帧拼成同一组 Daily Comic。"
    )


def generate_single_cartoon_frame_with_aliyun(
    prompt: str,
    reference_image: Path,
    output_path: Path,
    image_model: str,
) -> dict[str, Any]:
    settings = _load_settings()
    if not settings.dashscope_api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY for comic image generation.")
    api_base = (settings.dashscope_openai_base_url or DEFAULT_DASHSCOPE_OPENAI_BASE_URL).replace("/compatible-mode/v1", "/api/v1").rstrip("/")
    url = f"{api_base}/services/aigc/multimodal-generation/generation"
    encoded = base64.b64encode(reference_image.read_bytes()).decode("ascii")
    payload = {
        "model": image_model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        {"image": f"data:image/jpeg;base64,{encoded}"},
                    ],
                }
            ]
        },
        "parameters": {
            "enable_sequential": False,
            "size": "2K",
            "n": 1,
            "watermark": False,
            "seed": int(time.time()) % 2_147_483_647,
        },
    }
    started = time.time()
    response = requests.post(
        url,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings.dashscope_api_key}"},
        json=payload,
        timeout=240,
    )
    elapsed = round(time.time() - started, 3)
    data = _response_json(response)
    if response.status_code >= 400:
        raise RuntimeError(f"Aliyun comic frame generation failed: {response.status_code} {json.dumps(data, ensure_ascii=False)[:500]}")
    image_url = _extract_image_url(data)
    if not image_url:
        raise RuntimeError("Aliyun comic frame response did not include an image URL.")
    image_response = requests.get(image_url, timeout=120)
    image_response.raise_for_status()
    output_path.write_bytes(image_response.content)
    return {"status": "done", "elapsed_sec": elapsed, "usage": data.get("usage", {})}


def render_mock_cartoon_frames(reference_images: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for index, path in enumerate(reference_images, start=1):
        image = Image.open(path).convert("RGB")
        image = image.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.CONTOUR)
        output_path = output_dir / f"frame_{index:02d}.png"
        image.save(output_path, quality=92)
        frame_paths.append(output_path)
    return frame_paths


def render_comic_collage(frame_paths: list[Path], output_path: Path, fit_mode: str = "cover") -> None:
    if not frame_paths:
        raise RuntimeError("No comic frames available for collage rendering.")
    frames = [Image.open(path).convert("RGB") for path in frame_paths[:8]]
    width = height = 2048
    rng = random.Random(_collage_seed(frame_paths))
    canvas = Image.new("RGB", (width, height), (22, 20, 18))
    rects = _collage_rects(len(frames), rng)
    for image, rect in zip(frames, rects):
        if fit_mode == "contain":
            _paste_contain(canvas, image, rect)
        else:
            _paste_cover(canvas, image, rect, radius=0, border=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def render_square_grid_draft(frame_paths: list[Path], output_path: Path) -> None:
    if not frame_paths:
        raise RuntimeError("No reference frames available for square grid draft rendering.")
    frames = [Image.open(path).convert("RGB") for path in frame_paths[:6]]
    width = height = 2048
    columns = 3 if len(frames) > 4 else 2
    rows = 2 if len(frames) > 2 else 1
    gap = 8
    cell_w = (width - gap * (columns - 1)) // columns
    cell_h = (height - gap * (rows - 1)) // rows
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    for index, image in enumerate(frames):
        row, col = divmod(index, columns)
        if row >= rows:
            break
        x0 = col * (cell_w + gap)
        y0 = row * (cell_h + gap)
        x1 = width if col == columns - 1 else x0 + cell_w
        y1 = height if row == rows - 1 else y0 + cell_h
        _paste_cover(canvas, image, (x0, y0, x1, y1), radius=0, border=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def _collage_rects(count: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    layouts = {
        1: [[(120, 120, 1928, 1928)]],
        2: [
            [(0, 0, 2048, 1160), (0, 1168, 2048, 2048)],
            [(0, 0, 1260, 2048), (1268, 0, 2048, 2048)],
        ],
        3: [
            [(0, 0, 2048, 1024), (0, 1032, 1020, 2048), (1028, 1032, 2048, 2048)],
            [(0, 0, 1140, 2048), (1148, 0, 2048, 1020), (1148, 1028, 2048, 2048)],
        ],
        4: [
            [(0, 0, 1160, 900), (1168, 0, 2048, 900), (0, 908, 900, 2048), (908, 908, 2048, 2048)],
            [(0, 0, 2048, 740), (0, 748, 1020, 1390), (1028, 748, 2048, 1390), (0, 1398, 2048, 2048)],
        ],
        5: [
            [(0, 0, 2048, 760), (0, 768, 900, 1370), (908, 768, 2048, 1370), (0, 1378, 1220, 2048), (1228, 1378, 2048, 2048)],
            [(0, 0, 1120, 900), (1128, 0, 2048, 700), (0, 908, 700, 2048), (708, 908, 1430, 2048), (1438, 708, 2048, 2048)],
        ],
        6: [
            [(0, 0, 1320, 720), (1328, 0, 2048, 720), (0, 728, 680, 1370), (688, 728, 1360, 1370), (1368, 728, 2048, 1370), (0, 1378, 2048, 2048)],
            [(0, 0, 760, 760), (768, 0, 2048, 760), (0, 768, 980, 1360), (988, 768, 2048, 1360), (0, 1368, 760, 2048), (768, 1368, 2048, 2048)],
        ],
    }
    key = min(max(count, 1), 6)
    return list(rng.choice(layouts[key]))[:count]


def _paste_cover(
    canvas: Image.Image,
    image: Image.Image,
    rect: tuple[int, int, int, int],
    radius: int,
    border: int,
) -> None:
    x0, y0, x1, y1 = rect
    width, height = x1 - x0, y1 - y0
    scale = max(width / image.width, height / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    cropped = resized.crop((left, top, left + width, top + height)).convert("RGBA")
    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    if radius:
        mask_draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    else:
        mask_draw.rectangle((0, 0, width, height), fill=255)
    canvas_rgba = canvas.convert("RGBA")
    cropped.putalpha(mask)
    canvas_rgba.alpha_composite(cropped, (x0, y0))
    canvas.paste(canvas_rgba.convert("RGB"))


def _paste_contain(canvas: Image.Image, image: Image.Image, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    width, height = x1 - x0, y1 - y0
    cell = Image.new("RGB", (width, height), (246, 244, 238))
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    cell.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    canvas.paste(cell, (x0, y0))


def render_placeholder_panel(prompt: str, reference_images: list[Path], output_path: Path) -> dict[str, Any]:
    canvas = Image.new("RGB", (1024, 1024), (248, 243, 234))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((28, 28, 996, 996), radius=44, outline=(30, 30, 30), width=4)
    positions = [(60, 70), (542, 70), (60, 542), (542, 542)]
    for path, (x, y) in zip(reference_images[:4], positions):
        image = Image.open(path).convert("RGB")
        image.thumbnail((390, 390))
        draw.rounded_rectangle((x, y, x + 410, y + 410), radius=28, fill=(255, 255, 255), outline=(40, 40, 40), width=3)
        canvas.paste(image, (x + 10, y + 10))
    draw.text((70, 930), "Mock Daily Comic Panel", fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return {"status": "mock", "prompt_preview": prompt[:300]}


def render_comic_card(panel_path: Path, storyboard: dict[str, Any], output_path: Path, include_text: bool = False) -> None:
    if not include_text:
        image = Image.open(panel_path).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=94)
        return

    width, height = 1080, 1680
    card = Image.new("RGB", (width, height), (0, 0, 0)).convert("RGBA")
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((-120, -220, width + 120, 430), fill=(20, 30, 48, 110))
    card = Image.alpha_composite(card, glow.filter(ImageFilter.GaussianBlur(70)))

    panel_box = (980, 980)
    panel = Image.open(panel_path).convert("RGB")
    panel.thumbnail(panel_box, Image.Resampling.LANCZOS)
    panel_canvas = Image.new("RGB", panel_box, (245, 243, 237))
    paste_x = (panel_box[0] - panel.width) // 2
    paste_y = (panel_box[1] - panel.height) // 2
    panel_canvas.paste(panel, (paste_x, paste_y))
    mask = Image.new("L", panel_box, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, 980, 980), radius=48, fill=255)
    frame = Image.new("RGBA", (1024, 1024), (255, 255, 255, 0))
    frame_draw = ImageDraw.Draw(frame)
    frame_draw.rounded_rectangle((0, 0, 1024, 1024), radius=64, fill=(255, 255, 255, 255))
    card.alpha_composite(frame, (28, 38))
    panel_rgba = panel_canvas.convert("RGBA")
    panel_rgba.putalpha(mask)
    card.alpha_composite(panel_rgba, (50, 60))

    draw = ImageDraw.Draw(card)
    title_font = _font(72)
    body_font = _font(35)
    meta_font = _font(29)
    tag_font = _font(33)
    x, y = 76, 1154
    draw.text((x, y), str(storyboard.get("title") or "今天的小片段"), font=title_font, fill=(255, 255, 255, 255))
    y += 110
    for line in _wrap_px(str(storyboard.get("caption") or ""), body_font, 920, draw)[:6]:
        draw.text((x, y), line, font=body_font, fill=(205, 209, 216, 255))
        y += 54
    draw.line((0, 1544, width, 1544), fill=(28, 28, 30, 255), width=2)
    tag = _label_text(storyboard.get("tag"), "生活漫画")
    draw.text((76, 1584), tag, font=tag_font, fill=(87, 172, 244, 255))
    tag_width = int(draw.textlength(tag, font=tag_font))
    dot_x = min(76 + tag_width + 28, 340)
    draw.ellipse((dot_x, 1601, dot_x + 16, 1617), fill=(46, 46, 48, 255))
    draw.text((dot_x + 40, 1586), str(storyboard.get("date_label") or ""), font=meta_font, fill=(145, 150, 160, 255))
    draw.text((812, 1586), "PhoneLifeAgent", font=meta_font, fill=(235, 235, 235, 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card.convert("RGB").save(output_path, quality=94)


def render_comic_html(card_name: str) -> str:
    return (
        '<!doctype html><meta charset="utf-8"><title>Daily Comic</title>'
        "<style>body{margin:0;background:#050505;color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,sans-serif;display:flex;justify-content:center}"
        "main{max-width:780px;padding:28px 16px 48px}img{width:100%;border-radius:24px;display:block}</style>"
        f'<main><img src="{card_name}" alt="daily comic card"></main>\n'
    )


def _storyboard_system_prompt() -> str:
    return (
        "你是 PhoneLifeAgent 的 Daily Comic 编导。"
        "你必须以最终 Life Story 为主线生成漫画说明，Evidence 只用于校验事实和提供视觉锚点。"
        "只输出简体中文 JSON，不要 markdown，不要英文叙述。"
        "caption、storyline、panels 要保持第一人称生活记录视角；不要写“用户/拍摄者/father/daughter/the child”。"
        "字段：title、caption、tag、date_label、storyline、panels、visual_anchors、forbidden。"
        "caption 必须短，35-60 个中文字符，像用户会愿意分享的一句话；tag 是 2-6 个中文字符，不要井号。"
        "panels 是适合画成场景拼图的生活片段；visual_anchors 是真实物品/场景。"
        "forbidden 写成柔性的事实边界，用来提醒哪些内容证据不足，不要写成机械禁令。"
    )


def _storyboard_user_prompt(
    story_markdown: str,
    story_json: dict[str, Any],
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
) -> str:
    media_lines = []
    for frame in media_manifest.get("selected_keyframes", [])[:12]:
        media_lines.append(
            f"- {frame.get('local_time')} clip={frame.get('clip_id')} summary={_one_line(frame.get('clip_summary'))} caption={_one_line(frame.get('caption'))}"
        )
    return (
        "Final Life Story:\n"
        f"{_trim_text(story_markdown, 7000)}\n\n"
        "Life Story metadata:\n"
        f"{json.dumps({key: story_json.get(key) for key in ['time_range', 'source_counts', 'story_model']}, ensure_ascii=False)}\n\n"
        "Evidence time range:\n"
        f"{json.dumps(evidence_pack.get('time_range', {}), ensure_ascii=False)}\n\n"
        "Visual evidence candidates:\n"
        + "\n".join(media_lines)
        + "\n\n要求：漫画内容必须和 Final Life Story 对齐，不要把视觉证据里无关或不确定内容提升为故事主线。"
    )


def _fallback_storyboard(
    story_markdown: str,
    story_json: dict[str, Any],
    evidence_pack: dict[str, Any],
    media_manifest: dict[str, Any],
) -> dict[str, Any]:
    title = _first_heading(story_markdown) or "今天的小片段"
    date_label = _date_label(evidence_pack)
    summary = _story_overview(story_markdown) or _first_body_sentence(story_markdown) or _one_line(story_json.get("report_markdown"))
    anchors = _visual_anchors_from_media(media_manifest)
    return {
        "schema_version": "daily_comic_storyboard.v1",
        "title": _short_title(title),
        "caption": _short_caption(summary),
        "tag": "生活漫画",
        "date_label": date_label,
        "storyline": _one_line(summary),
        "panels": anchors[:6] or ["一天中的代表性片段"],
        "visual_anchors": anchors[:10],
        "forbidden": ["不添加与 Story 无关的生活事件", "不强化证据不足的人物关系", "不虚构具体对话"],
    }


def _normalize_storyboard(parsed: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    result = {**fallback, **{key: value for key, value in parsed.items() if value}}
    result["schema_version"] = "daily_comic_storyboard.v1"
    result["title"] = _short_title(str(result.get("title") or fallback["title"]))
    result["caption"] = _short_caption(str(result.get("caption") or fallback["caption"]))
    result["tag"] = _label_text(result.get("tag"), fallback.get("tag") or "生活漫画")
    result["date_label"] = _normalize_date_label(result.get("date_label") or fallback["date_label"])
    result["panels"] = [_clean_panel_text(item) for item in _string_list(result.get("panels"))[:6]] or fallback["panels"]
    result["visual_anchors"] = _string_list(result.get("visual_anchors"))[:12] or fallback["visual_anchors"]
    result["forbidden"] = _merge_unique(_string_list(result.get("forbidden")), fallback["forbidden"])[:6]
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def _extract_image_url(data: dict[str, Any]) -> str | None:
    for choice in data.get("output", {}).get("choices", []):
        for item in choice.get("message", {}).get("content", []):
            if item.get("type") == "image" and item.get("image"):
                return str(item["image"])
    return None


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        value = response.json()
        return value if isinstance(value, dict) else {"response": value}
    except ValueError:
        return {"status_code": response.status_code, "text": response.text[:3000]}


def _load_settings() -> Any:
    settings = load_api_settings(Path.cwd())
    if not settings.dashscope_api_key:
        settings = type(settings)(
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            dashscope_openai_base_url=os.environ.get("DASHSCOPE_OPENAI_BASE_URL", ""),
            amap_api_key=os.environ.get("AMAP_API_KEY", ""),
        )
    return settings


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collage_seed(frame_paths: list[Path]) -> int:
    digest = hashlib.sha256()
    for path in frame_paths:
        digest.update(str(path).encode("utf-8"))
        if path.exists():
            digest.update(str(path.stat().st_size).encode("ascii"))
    return int(digest.hexdigest()[:12], 16)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=0)
    return ImageFont.load_default()


def _wrap_px(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    lines = []
    current = ""
    for char in text:
        candidate = current + char
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def _first_heading(markdown_text: str) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _first_body_sentence(markdown_text: str) -> str:
    text = re.sub(r"^#+ .*?$", "", markdown_text, flags=re.M)
    text = re.sub(r"[*_`#>-]", "", text)
    text = " ".join(text.split())
    parts = re.split(r"(?<=[。！？])", text)
    return "".join(parts[:3]).strip() or text[:160]


def _short_title(title: str) -> str:
    title = re.sub(r"[#*_`]", "", title).strip()
    if len(title) <= 12:
        return title
    return title[:12].rstrip("，。,. ")


def _short_caption(text: str) -> str:
    text = _one_line(text)
    if len(text) <= 100:
        return text
    return text[:99].rstrip("，。,. ") + "。"


def _date_label(evidence_pack: dict[str, Any]) -> str:
    time_range = evidence_pack.get("time_range", {})
    local = str(time_range.get("start_local_time") or "")
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", local)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}年{int(month)}月{int(day)}日"


def _normalize_date_label(value: Any) -> str:
    text = _one_line(value)
    match = re.match(r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})", text)
    if not match:
        return text[:20]
    year, month, day = match.groups()
    return f"{year}年{int(month)}月{int(day)}日"


def _label_text(value: Any, fallback: str) -> str:
    labels = _string_list(value)
    if labels:
        text = labels[0]
    else:
        text = _one_line(value) or fallback
    text = re.sub(r"^[\\[\\]'\"]+|[\\[\\]'\"]+$", "", text).strip()
    text = re.sub(r"[#＃].*$", "", text).strip()
    text = re.sub(r"\s+", "", text)
    if not text or any(char in text for char in "{}[]'\""):
        text = fallback
    return text[:8]


def _clean_panel_text(text: str) -> str:
    return re.sub(r"^画面[一二三四五六七八九十\d]+[:：]\s*", "", _one_line(text)).strip()


def _visual_anchors_from_media(media_manifest: dict[str, Any]) -> list[str]:
    anchors = []
    for frame in media_manifest.get("selected_keyframes", [])[:8]:
        summary = _one_line(frame.get("clip_summary"))
        if summary:
            anchors.append(summary)
    return anchors


def _story_overview(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    in_overview = False
    collected = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped.lstrip("#").strip()
            if in_overview:
                break
            in_overview = heading in {"一天总览", "总览", "今日总览"}
            continue
        if in_overview and stripped:
            collected.append(stripped)
    return _one_line(" ".join(collected))


def _storyboard_terms(storyboard: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            str(storyboard.get("storyline") or ""),
            " ".join(_string_list(storyboard.get("panels"))),
            " ".join(_string_list(storyboard.get("visual_anchors"))),
        ]
    )
    candidates = re.findall(r"[\u4e00-\u9fff]{2,6}|[A-Za-z0-9_]{3,}", text)
    stop = {"故事", "画面", "生活", "漫画", "时间", "场景", "主角", "普通", "一段"}
    terms = []
    for item in candidates:
        if item in stop or item in terms:
            continue
        terms.append(item.lower())
    return terms[:30]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_text_item(item) for item in value if _text_item(item)]
    if isinstance(value, dict):
        item = _text_item(value)
        return [item] if item else []
    if isinstance(value, str):
        return [line.strip("- ").strip() for line in value.splitlines() if line.strip()]
    return []


def _text_item(value: Any) -> str:
    if isinstance(value, dict):
        for key in ["description", "caption", "focus", "text", "summary", "name"]:
            if value.get(key):
                return _one_line(value[key])
        return _one_line(" ".join(str(item) for item in value.values() if item))
    return _one_line(value)


def _merge_unique(primary: list[str], secondary: list[str]) -> list[str]:
    merged = []
    for item in [*primary, *secondary]:
        if item and item not in merged:
            merged.append(item)
    return merged


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[trimmed]..."
