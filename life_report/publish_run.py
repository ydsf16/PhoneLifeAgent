from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .comic_products import _read_json as _read_json_dict
from .highlight_video_products import render_highlight_html
from .story_products import render_story_html


def publish_run(run_dir: Path, target_dir: Path) -> dict[str, Any]:
    run = run_dir.expanduser().resolve()
    target = target_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    story_result = _publish_story(run, target / "story")
    comic_result = _publish_comic(run, target / "comic")
    highlight_result = _publish_highlight(run, target / "highlight_video")

    run_index = target / "index.html"
    run_index.write_text(
        _render_run_index(
            run_name=run.name,
            story_href="story/",
            comic_href="comic/",
            highlight_href="highlight_video/",
        ),
        encoding="utf-8",
    )

    demo_root = target.parent
    demo_root.mkdir(parents=True, exist_ok=True)
    demo_index = demo_root / "index.html"
    demo_index.write_text(_render_demo_index(_discover_runs(demo_root)), encoding="utf-8")

    return {
        "run_dir": str(run),
        "target_dir": str(target),
        "story_dir": str(story_result["dir"]),
        "comic_dir": str(comic_result["dir"]),
        "highlight_dir": str(highlight_result["dir"]),
        "run_index_path": str(run_index),
        "demo_index_path": str(demo_index),
    }


def _publish_story(run: Path, target: Path) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    story_markdown_path = run / "story" / "life_story.md"
    evidence_pack_path = run / "story" / "story_evidence_pack.json"
    story_markdown = story_markdown_path.read_text(encoding="utf-8")
    evidence_pack = _read_json_dict(evidence_pack_path)
    media = evidence_pack.setdefault("media", {})

    route_map_path = Path(str(media.get("overall_route_map") or ""))
    if route_map_path.exists():
        copied_map = target / route_map_path.name
        shutil.copy2(route_map_path, copied_map)
        media["overall_route_map"] = copied_map.name

    copied_keyframes = []
    for frame in media.get("selected_keyframes", []):
        if not isinstance(frame, dict):
            continue
        source = Path(str(frame.get("keyframe_path") or ""))
        if not source.exists():
            continue
        keyframes_dir = target / "keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        copied = keyframes_dir / source.name
        shutil.copy2(source, copied)
        copied_frame = dict(frame)
        copied_frame["keyframe_path"] = f"keyframes/{copied.name}"
        copied_keyframes.append(copied_frame)
    media["selected_keyframes"] = copied_keyframes

    html_path = target / "index.html"
    html_path.write_text(render_story_html(story_markdown, evidence_pack), encoding="utf-8")
    (target / "life_story.md").write_text(story_markdown, encoding="utf-8")
    (target / "story_evidence_pack.json").write_text(json.dumps(evidence_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"dir": target, "html_path": html_path}


def _publish_comic(run: Path, target: Path) -> dict[str, Any]:
    source_dir = run / "comic"
    target.mkdir(parents=True, exist_ok=True)
    files = [
        "daily_comic.html",
        "daily_comic.png",
        "daily_comic_card.png",
        "daily_comic_panel.png",
        "comic_storyline.json",
        "comic_storyboard.json",
        "comic_reference_plan.json",
    ]
    for name in files:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target / name)
    refs_source = source_dir / "refs"
    if refs_source.exists():
        shutil.copytree(refs_source, target / "refs", dirs_exist_ok=True)
    html_source = source_dir / "daily_comic.html"
    if html_source.exists():
        (target / "index.html").write_text(html_source.read_text(encoding="utf-8"), encoding="utf-8")
    return {"dir": target, "html_path": target / "index.html"}


def _publish_highlight(run: Path, target: Path) -> dict[str, Any]:
    source_dir = run / "highlight_video"
    target.mkdir(parents=True, exist_ok=True)
    plan_path = source_dir / "highlight_plan.json"
    plan = _read_json_dict(plan_path)
    video_source = source_dir / "highlight_video.mp4"
    if video_source.exists():
        shutil.copy2(video_source, target / "highlight_video.mp4")
    if plan:
        (target / "highlight_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "index.html").write_text(render_highlight_html("highlight_video.mp4", plan), encoding="utf-8")
    else:
        html_source = source_dir / "highlight_video.html"
        if html_source.exists():
            (target / "index.html").write_text(html_source.read_text(encoding="utf-8"), encoding="utf-8")
    return {"dir": target, "html_path": target / "index.html"}


def _discover_runs(demo_root: Path) -> list[str]:
    runs = [path.name for path in demo_root.iterdir() if path.is_dir() and (path / "index.html").exists()]
    return sorted(runs, reverse=True)


def _render_demo_index(run_names: list[str]) -> str:
    items = "".join(f'<li><a href="{name}/">{name}</a></li>' for name in run_names)
    return (
        "<!doctype html><meta charset=\"utf-8\"><title>PhoneLifeAgent Demo</title>"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<style>"
        "body{margin:0;background:#050505;color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,sans-serif}"
        "main{width:min(100%,900px);margin:0 auto;padding:40px 20px 64px}"
        "h1{font-size:34px;margin:0 0 12px}p{color:#b4b4bd;line-height:1.7}ul{padding-left:1.2rem}"
        "li{margin:14px 0}a{color:#7cc7ff;text-decoration:none}"
        "</style>"
        f"<main><h1>PhoneLifeAgent Demo</h1><p>Published runs</p><ul>{items}</ul></main>\n"
    )


def _render_run_index(run_name: str, story_href: str, comic_href: str, highlight_href: str) -> str:
    return (
        "<!doctype html><meta charset=\"utf-8\"><title>PhoneLifeAgent Run</title>"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<style>"
        "body{margin:0;background:#050505;color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,sans-serif}"
        "main{width:min(100%,900px);margin:0 auto;padding:40px 20px 64px}"
        "h1{font-size:34px;margin:0 0 12px}p{color:#b4b4bd;line-height:1.7}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-top:24px}"
        ".card{display:block;padding:20px;border:1px solid #2a2a2e;border-radius:18px;background:#111114;color:#f5f5f5;text-decoration:none}"
        ".card b{display:block;font-size:20px;margin-bottom:8px}.card span{color:#a1a1aa}"
        "</style>"
        f"<main><h1>{run_name}</h1><p>Published story, comic, and highlight video.</p>"
        "<div class=\"grid\">"
        f'<a class="card" href="{story_href}"><b>Story</b><span>Read the life story page</span></a>'
        f'<a class="card" href="{comic_href}"><b>Comic</b><span>Open the comic page</span></a>'
        f'<a class="card" href="{highlight_href}"><b>Highlight Video</b><span>Play the highlight video</span></a>'
        "</div></main>\n"
    )
