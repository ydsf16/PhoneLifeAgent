from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Callable

from .audio_products import build_audio_products
from .location_products import build_location_products
from .model_probe import probe_audio_model
from .motion_products import build_motion_products
from .pipeline_defaults import (
    DEFAULT_AUDIO_CONCURRENCY,
    DEFAULT_AUDIO_MODEL,
    DEFAULT_MAX_STORY_KEYFRAMES,
    DEFAULT_PROVIDER,
    DEFAULT_STORY_THINKING,
    DEFAULT_STORY_TEXT_MODEL,
    DEFAULT_SUMMARY_THINKING,
    DEFAULT_SUMMARY_TEXT_MODEL,
    DEFAULT_VIDEO_CONCURRENCY,
    DEFAULT_VIDEO_MODEL,
)
from .session_preflight import build_session_preflight
from .story_products import build_story_products
from .video_products import build_video_products, probe_video_model


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SessionPipelineConfig:
    session_path: Path
    output_dir: Path | None = None
    provider: str = DEFAULT_PROVIDER
    audio_model: str = DEFAULT_AUDIO_MODEL
    video_model: str = DEFAULT_VIDEO_MODEL
    summary_model: str = DEFAULT_SUMMARY_TEXT_MODEL
    story_model: str = DEFAULT_STORY_TEXT_MODEL
    summary_thinking: bool = DEFAULT_SUMMARY_THINKING
    story_thinking: bool = DEFAULT_STORY_THINKING
    audio_concurrency: int = DEFAULT_AUDIO_CONCURRENCY
    video_concurrency: int = DEFAULT_VIDEO_CONCURRENCY
    use_amap: bool = False
    max_story_keyframes: int = DEFAULT_MAX_STORY_KEYFRAMES
    force_rebuild: bool = False


def default_session_output_dir(session_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{session_path.name}_{stamp}"


def run_session_pipeline(config: SessionPipelineConfig, progress: ProgressCallback | None = None) -> dict[str, Any]:
    pipeline_started = time.time()
    session = config.session_path.expanduser().resolve()
    output = (config.output_dir or default_session_output_dir(session)).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provider = _provider(config.provider)
    text_provider = "aliyun" if provider == "aliyun" else "none"

    def log(message: str) -> None:
        if progress:
            progress(message)

    location_dir = output / "location"
    motion_dir = output / "motion"
    preflight_dir = output / "preflight"
    audio_probe_dir = output / "audio" / "probe"
    audio_products_dir = output / "audio" / "products"
    video_probe_dir = output / "video" / "probe"
    video_products_dir = output / "video" / "products"
    story_dir = output / "story"

    preflight_result = _run_cached_step(
        "Preflight",
        preflight_dir / "quarantine_manifest.json",
        config.force_rebuild,
        log,
        lambda: build_session_preflight(session, preflight_dir),
    )
    quarantine_manifest_path = preflight_dir / "quarantine_manifest.json"

    def run_location() -> dict[str, Any]:
        return _run_cached_step(
            "Location",
            location_dir / "location_timeline.json",
            config.force_rebuild,
            log,
            lambda: build_location_products(session, location_dir, use_amap=config.use_amap),
            cached_result=lambda: _cached_location_products(location_dir),
            cached_valid=lambda: _cached_location_valid(location_dir, use_amap=config.use_amap),
        )

    def run_motion() -> dict[str, Any]:
        return _run_cached_step(
            "Motion",
            motion_dir / "motion_timeline.json",
            config.force_rebuild,
            log,
            lambda: build_motion_products(session, motion_dir),
        )

    def run_audio_chain() -> tuple[dict[str, Any], dict[str, Any]]:
        audio_probe_result = _run_cached_step(
            "Audio Understanding",
            audio_probe_dir / "audio_understandings.json",
            config.force_rebuild,
            log,
            lambda: probe_audio_model(
                session_path=session,
                output_dir=audio_probe_dir,
                provider=provider,
                model=config.audio_model if provider == "aliyun" else None,
                limit_audio=0,
                audio_max_seconds=None,
                concurrency=max(1, config.audio_concurrency),
                location_context_path=location_dir / "clip_location_context.json",
                motion_context_path=motion_dir / "clip_motion_context.json",
                quarantine_manifest_path=quarantine_manifest_path,
            ).to_dict(),
        )
        audio_result_inner = _run_cached_step(
            "Audio Summary",
            audio_products_dir / "audio_story_input.txt",
            config.force_rebuild,
            log,
            lambda: build_audio_products(
                understandings_path=audio_probe_dir / "audio_understandings.json",
                output_dir=audio_products_dir,
                provider=text_provider,
                story_model=config.summary_model,
                summary_thinking=config.summary_thinking,
                location_compact_path=location_dir / "location_compact_raw.txt",
                motion_compact_path=motion_dir / "motion_compact_raw.txt",
            ),
            cached_result=lambda: _cached_audio_products(audio_products_dir),
        )
        return audio_probe_result, audio_result_inner

    def run_video_chain() -> tuple[dict[str, Any], dict[str, Any]]:
        video_probe_result = _run_cached_step(
            "Video Understanding",
            video_probe_dir / "video_understandings.json",
            config.force_rebuild,
            log,
            lambda: _video_probe_to_dict(
                probe_video_model(
                    session_path=session,
                    output_dir=video_probe_dir,
                    provider=provider,
                    model=config.video_model if provider == "aliyun" else None,
                    limit_clips=0,
                    understand_clips=1_000_000,
                    concurrency=max(1, config.video_concurrency),
                    location_context_path=location_dir / "clip_location_context.json",
                    motion_context_path=motion_dir / "clip_motion_context.json",
                    quarantine_manifest_path=quarantine_manifest_path,
                )
            ),
        )
        video_result_inner = _run_cached_step(
            "Video Summary",
            video_products_dir / "video_story_input.txt",
            config.force_rebuild,
            log,
            lambda: build_video_products(
                understandings_path=video_probe_dir / "video_understandings.json",
                output_dir=video_products_dir,
                provider=text_provider,
                story_model=config.summary_model,
                summary_thinking=config.summary_thinking,
                max_story_keyframes=config.max_story_keyframes,
                location_compact_path=location_dir / "location_compact_raw.txt",
                motion_compact_path=motion_dir / "motion_compact_raw.txt",
            ),
            cached_result=lambda: _cached_video_products(video_products_dir),
        )
        return video_probe_result, video_result_inner

    with ThreadPoolExecutor(max_workers=2) as executor:
        location_future = executor.submit(run_location)
        motion_future = executor.submit(run_motion)
        location_result = location_future.result()
        motion_result = motion_future.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        audio_future = executor.submit(run_audio_chain)
        video_future = executor.submit(run_video_chain)
        audio_probe, audio_result = audio_future.result()
        video_probe, video_result = video_future.result()

    story_result = _run_cached_step(
        "Final Story",
        story_dir / "life_story.html",
        config.force_rebuild,
        log,
        lambda: build_story_products(
            audio_products_dir=audio_products_dir,
            video_products_dir=video_products_dir,
            location_products_dir=location_dir,
            motion_products_dir=motion_dir,
            output_dir=story_dir,
            provider=text_provider,
            story_model=config.story_model,
            story_thinking=config.story_thinking,
            max_keyframes=config.max_story_keyframes,
            progress=log,
        ),
        cached_result=lambda: _cached_story_products(story_dir),
        cached_valid=lambda: _cached_story_valid(story_dir, location_dir),
    )

    log("Done")
    result = {
        "output_dir": str(output),
        "session_path": str(session),
        "provider": provider,
        "preflight": preflight_result,
        "summary_model": config.summary_model,
        "story_model": config.story_model,
        "summary_thinking": config.summary_thinking,
        "story_thinking": config.story_thinking,
        "story_html_path": story_result["life_story_html_path"],
        "story_markdown_path": story_result["life_story_markdown_path"],
        "story_json_path": story_result["life_story_json_path"],
        "location": location_result,
        "motion": motion_result,
        "audio_probe": audio_probe,
        "audio_products": audio_result,
        "video_probe": video_probe,
        "video_products": video_result,
        "story_products": story_result,
        "files": [
            "location/location_timeline.json",
            "preflight/quarantine_manifest.json",
            "preflight/session_health.json",
            "motion/motion_timeline.json",
            "audio/probe/audio_understandings.json",
            "audio/products/audio_story_input.txt",
            "video/probe/video_understandings.json",
            "video/products/video_story_input.txt",
            "story/life_story.html",
            "story/life_story.md",
            "story/life_story.json",
        ],
    }
    result["timings"] = _timing_summary(result, wall_clock_sec=time.time() - pipeline_started)
    return result


def _provider(value: str) -> str:
    if value in {"aliyun", "mock"}:
        return value
    raise ValueError(f"Unsupported run-session provider: {value}")


def _run_cached_step(
    name: str,
    marker: Path,
    force_rebuild: bool,
    log: ProgressCallback,
    fn: Callable[[], dict[str, Any]],
    cached_result: Callable[[], dict[str, Any]] | None = None,
    cached_valid: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if marker.exists() and not force_rebuild and (cached_valid is None or cached_valid()):
        message = f"{name} skipped (cached)"
        log(f"TASK|{name}|skipped|cached")
        log(message)
        cached = cached_result() if cached_result else {}
        return {**cached, "skipped": True, "marker": str(marker), "status": "skipped"}
    return _run_timed_step(name, log, fn)


def _run_timed_step(name: str, log: ProgressCallback, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.time()
    log(f"TASK|{name}|running|")
    log(f"{name} running")
    try:
        result = fn()
    except Exception:
        elapsed = time.time() - started
        log(f"TASK|{name}|failed|{elapsed:.1f}s")
        raise
    elapsed = time.time() - started
    log(f"TASK|{name}|done|{elapsed:.1f}s")
    log(f"{name} done in {elapsed:.1f}s")
    if isinstance(result, dict):
        return {**result, "status": result.get("status", "done"), "duration_sec": round(elapsed, 3)}
    return {"status": "done", "duration_sec": round(elapsed, 3), "result": result}


def _video_probe_to_dict(result: Any) -> dict[str, Any]:
    return {
        "output_dir": str(result.output_dir),
        "video_understandings_path": str(result.video_understandings_path),
        "video_understandings_pretty_path": str(result.video_understandings_pretty_path),
        "video_preparation_errors_path": str(result.video_preparation_errors_path),
        "video_model_errors_path": str(result.video_model_errors_path),
        "processed_clip_count": result.processed_clip_count,
        "understood_clip_count": result.understood_clip_count,
        "preparation_error_count": result.preparation_error_count,
        "model_error_count": result.model_error_count,
    }


def _cached_audio_products(output_dir: Path) -> dict[str, Any]:
    return {
        "audio_timeline_path": str(output_dir / "audio_timeline.json"),
        "audio_compact_raw_path": str(output_dir / "audio_compact_raw.txt"),
        "audio_story_input_path": str(output_dir / "audio_story_input.txt"),
    }


def _audio_probe_to_dict(result: Any) -> dict[str, Any]:
    return result.to_dict()


def _cached_video_products(output_dir: Path) -> dict[str, Any]:
    return {
        "video_timeline_path": str(output_dir / "video_timeline.json"),
        "video_compact_raw_path": str(output_dir / "video_compact_raw.txt"),
        "video_story_input_path": str(output_dir / "video_story_input.txt"),
        "video_story_media_manifest_path": str(output_dir / "video_story_media_manifest.json"),
    }


def _cached_location_products(output_dir: Path) -> dict[str, Any]:
    timeline_path = output_dir / "location_timeline.json"
    timeline = _read_json(timeline_path)
    return {
        "location_points_path": str(output_dir / "location_points_clean.json"),
        "location_timeline_path": str(timeline_path),
        "clip_location_context_path": str(output_dir / "clip_location_context.json"),
        "location_compact_raw_path": str(output_dir / "location_compact_raw.txt"),
        "route_geojson_path": str(output_dir / "route.geojson"),
        "overall_map_image": timeline.get("overall_map_image"),
        "amap_enabled": bool(timeline.get("overall_map_image")),
    }


def _cached_story_products(output_dir: Path) -> dict[str, Any]:
    return {
        "story_evidence_pack_path": str(output_dir / "story_evidence_pack.json"),
        "story_input_path": str(output_dir / "story_input.txt"),
        "life_story_json_path": str(output_dir / "life_story.json"),
        "life_story_markdown_path": str(output_dir / "life_story.md"),
        "life_story_html_path": str(output_dir / "life_story.html"),
    }


def _cached_location_valid(output_dir: Path, use_amap: bool) -> bool:
    if not use_amap:
        return True
    timeline = _read_json(output_dir / "location_timeline.json")
    route_map = timeline.get("overall_map_image")
    return bool(route_map and Path(route_map).exists())


def _cached_story_valid(story_dir: Path, location_dir: Path) -> bool:
    route_map = _read_json(location_dir / "location_timeline.json").get("overall_map_image")
    if not route_map:
        return True
    evidence_pack = _read_json(story_dir / "story_evidence_pack.json")
    story_route_map = evidence_pack.get("media", {}).get("overall_route_map")
    return story_route_map == route_map and Path(route_map).exists()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _timing_summary(result: dict[str, Any], wall_clock_sec: float) -> dict[str, Any]:
    stages = {
        "Location": result.get("location", {}),
        "Motion": result.get("motion", {}),
        "Audio Understanding": result.get("audio_probe", {}),
        "Audio Summary": result.get("audio_products", {}),
        "Video Understanding": result.get("video_probe", {}),
        "Video Summary": result.get("video_products", {}),
        "Final Story": result.get("story_products", {}),
    }
    stage_rows = []
    total = 0.0
    for name, data in stages.items():
        duration = float(data.get("duration_sec") or 0.0) if isinstance(data, dict) else 0.0
        total += duration
        stage_rows.append(
            {
                "name": name,
                "status": data.get("status", "unknown") if isinstance(data, dict) else "unknown",
                "duration_sec": round(duration, 3),
            }
        )
    return {"wall_clock_sec": round(wall_clock_sec, 3), "total_tracked_sec": round(total, 3), "stages": stage_rows}
