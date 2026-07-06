from __future__ import annotations

import argparse
from pathlib import Path

from .audio_products import build_audio_products
from .comic_products import build_comic_products
from .highlight_video_products import (
    DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
    DEFAULT_HIGHLIGHT_TARGET_SECONDS,
    build_highlight_video_products,
)
from .location_products import build_location_products
from .model_probe import probe_audio_model
from .motion_products import build_motion_products
from .publish_run import publish_run
from .pipeline_defaults import (
    DEFAULT_AUDIO_CONCURRENCY,
    DEFAULT_AUDIO_MODEL,
    DEFAULT_COMIC_IMAGE_MODEL,
    DEFAULT_COMIC_MAX_PANELS,
    DEFAULT_COMIC_MAX_REFERENCES_PER_PANEL,
    DEFAULT_COMIC_THINKING,
    DEFAULT_HIGHLIGHT_THINKING,
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
from .session_loader import load_sessions
from .session_pipeline import SessionPipelineConfig, run_session_pipeline
from .story_products import build_story_products
from .video_products import build_video_products, probe_video_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="life_report",
        description="PhoneLifeAgent Life Report tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="inspect one or more LifeLogger session folders",
    )
    inspect_parser.add_argument(
        "--sessions",
        nargs="+",
        type=Path,
        required=True,
        help="LifeLogger session directories",
    )

    run_session_parser = subparsers.add_parser(
        "run-session",
        help="run the full PhoneLifeAgent pipeline for one LifeLogger session",
    )
    run_session_parser.add_argument("--session", type=Path, required=True)
    run_session_parser.add_argument("--output", type=Path, default=None)
    run_session_parser.add_argument("--provider", choices=["aliyun", "mock"], default=DEFAULT_PROVIDER)
    run_session_parser.add_argument("--audio-model", default=DEFAULT_AUDIO_MODEL)
    run_session_parser.add_argument("--video-model", default=DEFAULT_VIDEO_MODEL)
    run_session_parser.add_argument("--summary-model", default=DEFAULT_SUMMARY_TEXT_MODEL)
    run_session_parser.add_argument("--story-model", default=DEFAULT_STORY_TEXT_MODEL)
    run_session_parser.add_argument("--summary-thinking", action=argparse.BooleanOptionalAction, default=DEFAULT_SUMMARY_THINKING)
    run_session_parser.add_argument("--story-thinking", action=argparse.BooleanOptionalAction, default=DEFAULT_STORY_THINKING)
    run_session_parser.add_argument("--audio-concurrency", type=int, default=DEFAULT_AUDIO_CONCURRENCY)
    run_session_parser.add_argument("--video-concurrency", type=int, default=DEFAULT_VIDEO_CONCURRENCY)
    run_session_parser.add_argument("--use-amap", action="store_true")
    run_session_parser.add_argument("--force-rebuild", action="store_true")

    preflight_parser = subparsers.add_parser(
        "preflight-session",
        help="inspect one LifeLogger session and quarantine clearly bad media rows",
    )
    preflight_parser.add_argument("--session", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)

    probe_parser = subparsers.add_parser(
        "probe-models",
        help="probe model providers with a small number of real media files",
    )
    probe_parser.add_argument("--session", type=Path, required=True)
    probe_parser.add_argument("--output", type=Path, default=Path("outputs/model_probe"))
    probe_parser.add_argument("--provider", choices=["mock", "aliyun"], default="mock")
    probe_parser.add_argument("--model", default=None)
    probe_parser.add_argument("--limit-audio", type=int, default=1)
    probe_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of audio segments to process in parallel",
    )
    probe_parser.add_argument("--limit-clips", type=int, default=0)
    probe_parser.add_argument(
        "--audio-max-seconds",
        type=int,
        default=0,
        help="trim/convert local m4a files to this many seconds; 0 means full segment",
    )
    probe_parser.add_argument("--location-context", type=Path, default=None, help="clip_location_context.json for per-audio prompt injection")
    probe_parser.add_argument("--motion-context", type=Path, default=None, help="clip_motion_context.json for per-audio prompt injection")
    probe_parser.add_argument("--quarantine-manifest", type=Path, default=None, help="preflight/quarantine_manifest.json")

    audio_products_parser = subparsers.add_parser(
        "build-audio-products",
        help="build audio_timeline.json, audio_compact_raw.txt, and audio_story_input.txt from existing audio understandings",
    )
    audio_products_parser.add_argument("--input", type=Path, required=True, help="audio_understandings.json")
    audio_products_parser.add_argument("--output", type=Path, required=True)
    audio_products_parser.add_argument("--provider", choices=["aliyun", "none"], default="aliyun")
    audio_products_parser.add_argument("--story-model", default=DEFAULT_SUMMARY_TEXT_MODEL)
    audio_products_parser.add_argument("--summary-thinking", action=argparse.BooleanOptionalAction, default=DEFAULT_SUMMARY_THINKING)
    audio_products_parser.add_argument("--location-compact", type=Path, default=None, help="location_compact_raw.txt for audio summary injection")
    audio_products_parser.add_argument("--motion-compact", type=Path, default=None, help="motion_compact_raw.txt for audio summary injection")

    video_parser = subparsers.add_parser(
        "probe-video",
        help="prepare aligned low-resolution AV clips and probe video understanding",
    )
    video_parser.add_argument("--session", type=Path, required=True)
    video_parser.add_argument("--output", type=Path, default=Path("outputs/video_probe"))
    video_parser.add_argument("--provider", choices=["mock", "aliyun"], default="mock")
    video_parser.add_argument("--model", default=None)
    video_parser.add_argument("--limit-clips", type=int, default=3)
    video_parser.add_argument("--understand-clips", type=int, default=1)
    video_parser.add_argument("--concurrency", type=int, default=1)
    video_parser.add_argument("--location-context", type=Path, default=None, help="clip_location_context.json for per-video prompt injection")
    video_parser.add_argument("--motion-context", type=Path, default=None, help="clip_motion_context.json for per-video prompt injection")
    video_parser.add_argument("--quarantine-manifest", type=Path, default=None, help="preflight/quarantine_manifest.json")

    video_products_parser = subparsers.add_parser(
        "build-video-products",
        help="build video_timeline.json, video_compact_raw.txt, and video_story_input.txt from existing video understandings",
    )
    video_products_parser.add_argument("--input", type=Path, required=True, help="video_understandings.json")
    video_products_parser.add_argument("--output", type=Path, required=True)
    video_products_parser.add_argument("--provider", choices=["aliyun", "none"], default="aliyun")
    video_products_parser.add_argument("--story-model", default=DEFAULT_SUMMARY_TEXT_MODEL)
    video_products_parser.add_argument("--summary-thinking", action=argparse.BooleanOptionalAction, default=DEFAULT_SUMMARY_THINKING)
    video_products_parser.add_argument(
        "--max-story-keyframes",
        type=int,
        default=DEFAULT_MAX_STORY_KEYFRAMES,
        help="maximum selected report keyframes to include in video_story_media_manifest.json",
    )
    video_products_parser.add_argument("--location-compact", type=Path, default=None, help="location_compact_raw.txt for video summary injection")
    video_products_parser.add_argument("--motion-compact", type=Path, default=None, help="motion_compact_raw.txt for video summary injection")

    location_products_parser = subparsers.add_parser(
        "build-location-products",
        help="build location evidence and clip context products from geo_location.csv",
    )
    location_products_parser.add_argument("--session", type=Path, required=True)
    location_products_parser.add_argument("--output", type=Path, required=True)
    location_products_parser.add_argument(
        "--use-amap",
        action="store_true",
        help="enrich location facts and maps with AMAP_API_KEY/GAODE_API_KEY",
    )

    motion_products_parser = subparsers.add_parser(
        "build-motion-products",
        help="build motion evidence and clip context products from device_motion.csv",
    )
    motion_products_parser.add_argument("--session", type=Path, required=True)
    motion_products_parser.add_argument("--output", type=Path, required=True)
    motion_products_parser.add_argument("--window-sec", type=float, default=10.0)

    story_products_parser = subparsers.add_parser(
        "build-story-products",
        help="build final Life Story report from audio, video, location, and motion products",
    )
    story_products_parser.add_argument("--audio-products", type=Path, required=True)
    story_products_parser.add_argument("--video-products", type=Path, required=True)
    story_products_parser.add_argument("--location-products", type=Path, required=True)
    story_products_parser.add_argument("--motion-products", type=Path, required=True)
    story_products_parser.add_argument("--output", type=Path, required=True)
    story_products_parser.add_argument("--provider", choices=["aliyun", "none"], default="aliyun")
    story_products_parser.add_argument("--story-model", default=DEFAULT_STORY_TEXT_MODEL)
    story_products_parser.add_argument("--story-thinking", action=argparse.BooleanOptionalAction, default=DEFAULT_STORY_THINKING)
    story_products_parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_STORY_KEYFRAMES)

    comic_products_parser = subparsers.add_parser(
        "build-comic-products",
        help="build Daily Comic outputs from an existing full run directory",
    )
    comic_products_parser.add_argument("--run-dir", type=Path, required=True)
    comic_products_parser.add_argument("--output", type=Path, default=None)
    comic_products_parser.add_argument("--provider", choices=["aliyun", "mock", "none"], default="aliyun")
    comic_products_parser.add_argument("--text-model", default=DEFAULT_STORY_TEXT_MODEL)
    comic_products_parser.add_argument("--comic-thinking", action=argparse.BooleanOptionalAction, default=DEFAULT_COMIC_THINKING)
    comic_products_parser.add_argument("--image-provider", choices=["ark", "mock", "none"], default=None)
    comic_products_parser.add_argument("--image-model", default=DEFAULT_COMIC_IMAGE_MODEL)
    comic_products_parser.add_argument("--max-reference-images", type=int, default=0)
    comic_products_parser.add_argument("--max-panels", type=int, default=DEFAULT_COMIC_MAX_PANELS)
    comic_products_parser.add_argument("--comic-style", default="daily_cartoon")

    highlight_video_parser = subparsers.add_parser(
        "build-highlight-video",
        help="build Highlight Video outputs from an existing full run directory",
    )
    highlight_video_parser.add_argument("--run-dir", type=Path, required=True)
    highlight_video_parser.add_argument("--output", type=Path, default=None)
    highlight_video_parser.add_argument("--provider", choices=["aliyun"], default="aliyun")
    highlight_video_parser.add_argument("--text-model", default=DEFAULT_STORY_TEXT_MODEL)
    highlight_video_parser.add_argument("--highlight-thinking", action=argparse.BooleanOptionalAction, default=DEFAULT_HIGHLIGHT_THINKING)
    highlight_video_parser.add_argument("--target-seconds", type=int, default=DEFAULT_HIGHLIGHT_TARGET_SECONDS)
    highlight_video_parser.add_argument("--max-segments", type=int, default=DEFAULT_HIGHLIGHT_MAX_SEGMENTS)

    publish_run_parser = subparsers.add_parser(
        "publish-run",
        help="publish one generated run into a docs/demo static site layout",
    )
    publish_run_parser.add_argument("--run-dir", type=Path, required=True)
    publish_run_parser.add_argument("--target", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        bundle = load_sessions(args.sessions)
        overview = bundle.to_dict()["overview"]
        print(f"Sessions: {overview['session_count']}")
        print(f"Time range: {overview['start_time']} -> {overview['end_time']}")
        print(f"Duration: {overview['duration']}")
        print(f"Recorded duration: {overview['recorded_duration']}")
        print(f"Clips: {overview['clip_count']}")
        print(f"Audio segments: {overview['audio_count']}")
        print(f"GPS points: {overview['gps_points']}")
        print(f"Motion points: {overview['motion_points']}")
        print(f"Size: {overview['total_size']}")
        print(f"Gaps: {overview['gap_count']}")
        for warning in overview["warnings"]:
            print(f"Warning: {warning}")
        return 0

    if args.command == "run-session":
        result = run_session_pipeline(
            SessionPipelineConfig(
                session_path=args.session,
                output_dir=args.output,
                provider=args.provider,
                audio_model=args.audio_model,
                video_model=args.video_model,
                summary_model=args.summary_model,
                story_model=args.story_model,
                summary_thinking=args.summary_thinking,
                story_thinking=args.story_thinking,
                audio_concurrency=max(1, args.audio_concurrency),
                video_concurrency=max(1, args.video_concurrency),
                use_amap=args.use_amap,
                force_rebuild=args.force_rebuild,
            ),
            progress=lambda message: print(message, flush=True),
        )
        print(f"Wrote full session report to {result['output_dir']}")
        print(f"- story: {result['story_html_path']}")
        print(f"- wall clock: {result.get('timings', {}).get('wall_clock_sec')}s")
        for name in result["files"]:
            print(f"- {name}")
        return 0

    if args.command == "preflight-session":
        result = build_session_preflight(args.session, args.output)
        print(f"Wrote session preflight to {args.output}")
        print(f"- quarantine: {result['quarantine_manifest_path']}")
        print(f"- health: {result['session_health_path']}")
        print(f"- valid audio rows: {result['valid_audio_rows']}")
        print(f"- valid video rows: {result['valid_video_rows']}")
        print(f"- quarantined media rows: {result['quarantined_media_rows']}")
        print(f"- location abnormal rows: {result['location_abnormal_rows']}")
        print(f"- motion abnormal rows: {result['motion_abnormal_rows']}")
        return 0

    if args.command == "probe-models":
        if args.limit_clips:
            print("Video clip probing is not implemented yet; audio probing will run first.")
        result = probe_audio_model(
            session_path=args.session,
            output_dir=args.output,
            provider=args.provider,
            model=args.model,
            limit_audio=args.limit_audio,
            audio_max_seconds=args.audio_max_seconds or None,
            concurrency=max(1, args.concurrency),
            location_context_path=args.location_context,
            motion_context_path=args.motion_context,
            quarantine_manifest_path=args.quarantine_manifest,
        )
        print(f"Wrote model probe outputs to {result.output_dir}")
        print(f"- {result.audio_understandings_path.name}: {result.processed_audio_count} audio segment(s)")
        print(f"- {result.audio_understandings_pretty_path.name}: pretty JSON")
        print(f"- {result.audio_probe_errors_path.name}: {result.audio_error_count} failed segment(s)")
        print("- model_response_cache/")
        return 0

    if args.command == "build-audio-products":
        result = build_audio_products(
            understandings_path=args.input,
            output_dir=args.output,
            story_model=args.story_model,
            provider=args.provider,
            summary_thinking=args.summary_thinking,
            location_compact_path=args.location_compact,
            motion_compact_path=args.motion_compact,
        )
        print(f"Wrote audio products to {args.output}")
        print(f"- audio_timeline.json: {result['audio_event_count']} event(s), {result['audio_moment_count']} moment(s)")
        print("- audio_compact_raw.txt: compact model input")
        print(f"- audio_story_input.txt: model={result['story_model']}")
        return 0

    if args.command == "probe-video":
        result = probe_video_model(
            session_path=args.session,
            output_dir=args.output,
            provider=args.provider,
            model=args.model,
            limit_clips=args.limit_clips,
            understand_clips=args.understand_clips,
            concurrency=max(1, args.concurrency),
            location_context_path=args.location_context,
            motion_context_path=args.motion_context,
            quarantine_manifest_path=args.quarantine_manifest,
        )
        print(f"Wrote video probe outputs to {result.output_dir}")
        print(f"- prepared clips: {result.processed_clip_count}")
        print(f"- video_understandings.json: {result.understood_clip_count} clip(s)")
        print(f"- video_preparation_errors.json: {result.preparation_error_count} skipped clip(s)")
        print(f"- video_model_errors.json: {result.model_error_count} failed model call(s)")
        print("- aligned_audio/")
        print("- model_video/")
        print("- report_video/")
        print("- report_keyframes/")
        return 0

    if args.command == "build-video-products":
        result = build_video_products(
            understandings_path=args.input,
            output_dir=args.output,
            story_model=args.story_model,
            provider=args.provider,
            summary_thinking=args.summary_thinking,
            max_story_keyframes=args.max_story_keyframes,
            location_compact_path=args.location_compact,
            motion_compact_path=args.motion_compact,
        )
        print(f"Wrote video products to {args.output}")
        print(f"- video_timeline.json: {result['video_event_count']} event(s), {result['video_keyframe_count']} keyframe(s)")
        print(f"- video_story_media_manifest.json: {result['selected_story_keyframe_count']} selected keyframe(s)")
        print("- video_compact_raw.txt: compact model input")
        print(f"- video_story_input.txt: model={result['story_model']}")
        return 0

    if args.command == "build-location-products":
        result = build_location_products(
            session_path=args.session,
            output_dir=args.output,
            use_amap=args.use_amap,
        )
        print(f"Wrote location products to {args.output}")
        print(f"- location_points_clean.json: {result['point_count']} point(s)")
        print(f"- location_timeline.json: {result['segment_count']} segment(s)")
        print(f"- clip_location_context.json: {result['video_context_count']} video, {result['audio_context_count']} audio context(s)")
        print(f"- AMAP enrichment: {'enabled' if result['amap_enabled'] else 'disabled'}")
        return 0

    if args.command == "build-motion-products":
        result = build_motion_products(
            session_path=args.session,
            output_dir=args.output,
            window_sec=args.window_sec,
        )
        print(f"Wrote motion products to {args.output}")
        print(f"- motion_features.json: {result['sample_count']} sample(s), {result['window_count']} window(s)")
        print(f"- motion_timeline.json: {result['segment_count']} segment(s)")
        print(f"- clip_motion_context.json: {result['video_context_count']} video, {result['audio_context_count']} audio context(s)")
        return 0

    if args.command == "build-story-products":
        result = build_story_products(
            audio_products_dir=args.audio_products,
            video_products_dir=args.video_products,
            location_products_dir=args.location_products,
            motion_products_dir=args.motion_products,
            output_dir=args.output,
            provider=args.provider,
            story_model=args.story_model,
            story_thinking=args.story_thinking,
            max_keyframes=args.max_keyframes,
        )
        print(f"Wrote story products to {args.output}")
        print(f"- story_evidence_pack.json: {result['timeline_item_count']} timeline item(s)")
        print("- story_input.txt: compact final model input")
        print(f"- life_story.md/html/json: model={result['story_model']}")
        print(f"- selected keyframes: {result['selected_keyframe_count']}")
        return 0

    if args.command == "build-comic-products":
        result = build_comic_products(
            run_dir=args.run_dir,
            output_dir=args.output,
            provider=args.provider,
            text_model=args.text_model,
            comic_thinking=args.comic_thinking,
            image_model=args.image_model,
            max_reference_images=max(1, args.max_reference_images),
            image_provider=args.image_provider,
            max_panels=max(1, args.max_panels),
            comic_style=args.comic_style,
            progress=lambda message: print(message, flush=True),
        )
        print(f"Wrote Daily Comic products to {Path(result['daily_comic_html_path']).parent}")
        print(f"- storyline: {result['comic_storyline_path']}")
        print(f"- storyboard: {result['comic_storyboard_path']}")
        print(f"- references: {result['comic_reference_plan_path']}")
        print(f"- prompt: {result['seedream_comic_prompt_path']}")
        print(f"- panel: {result['daily_comic_panel_path']}")
        print(f"- card: {result['daily_comic_card_path']}")
        print(f"- html: {result['daily_comic_html_path']}")
        print(f"- model: {result['image_provider']} / {result['image_model']}")
        print(f"- panels: {result['panel_count']}")
        print(f"- reference images: {result['reference_image_count']}")
        return 0

    if args.command == "build-highlight-video":
        result = build_highlight_video_products(
            run_dir=args.run_dir,
            output_dir=args.output,
            provider=args.provider,
            text_model=args.text_model,
            highlight_thinking=args.highlight_thinking,
            target_seconds=max(10, args.target_seconds),
            max_segments=max(1, args.max_segments),
            progress=lambda message: print(message, flush=True),
        )
        print(f"Wrote Highlight Video products to {Path(result['highlight_video_html_path']).parent}")
        print(f"- storyline: {result['highlight_storyline_path']}")
        print(f"- plan: {result['highlight_plan_path']}")
        print(f"- video: {result['highlight_video_path']}")
        print(f"- html: {result['highlight_video_html_path']}")
        print(f"- model: {result['provider']} / {result['text_model']}")
        print(f"- segments: {result['segment_count']}")
        print(f"- estimated duration: {result['estimated_duration_sec']}s")
        return 0

    if args.command == "publish-run":
        result = publish_run(
            run_dir=args.run_dir,
            target_dir=args.target,
        )
        print(f"Published run to {result['target_dir']}")
        print(f"- run index: {result['run_index_path']}")
        print(f"- demo index: {result['demo_index_path']}")
        print(f"- story: {result['story_dir']}")
        print(f"- comic: {result['comic_dir']}")
        print(f"- highlight: {result['highlight_dir']}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
