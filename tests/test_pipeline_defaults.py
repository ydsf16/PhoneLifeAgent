import unittest

from apps.desktop_studio.ui_job_defaults import (
    DESKTOP_DEFAULT_AUDIO_CONCURRENCY,
    DESKTOP_DEFAULT_COMIC_MAX_PANELS,
    DESKTOP_DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
    DESKTOP_DEFAULT_HIGHLIGHT_TARGET_SECONDS,
    DESKTOP_DEFAULT_STORY_MODEL,
    DESKTOP_DEFAULT_SUMMARY_MODEL,
    DESKTOP_DEFAULT_VIDEO_CONCURRENCY,
)
from life_report.cli import build_parser
from life_report.pipeline_defaults import (
    DEFAULT_AUDIO_CONCURRENCY,
    DEFAULT_COMIC_MAX_PANELS,
    DEFAULT_COMIC_THINKING,
    DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
    DEFAULT_HIGHLIGHT_THINKING,
    DEFAULT_HIGHLIGHT_TARGET_SECONDS,
    DEFAULT_STORY_THINKING,
    DEFAULT_STORY_TEXT_MODEL,
    DEFAULT_SUMMARY_THINKING,
    DEFAULT_SUMMARY_TEXT_MODEL,
    DEFAULT_VIDEO_CONCURRENCY,
)
from life_report.session_pipeline import SessionPipelineConfig


class PipelineDefaultsTest(unittest.TestCase):
    def test_desktop_defaults_match_pipeline_defaults(self) -> None:
        self.assertEqual(DESKTOP_DEFAULT_SUMMARY_MODEL, DEFAULT_SUMMARY_TEXT_MODEL)
        self.assertEqual(DESKTOP_DEFAULT_STORY_MODEL, DEFAULT_STORY_TEXT_MODEL)
        self.assertEqual(DESKTOP_DEFAULT_AUDIO_CONCURRENCY, DEFAULT_AUDIO_CONCURRENCY)
        self.assertEqual(DESKTOP_DEFAULT_VIDEO_CONCURRENCY, DEFAULT_VIDEO_CONCURRENCY)
        self.assertEqual(DESKTOP_DEFAULT_COMIC_MAX_PANELS, DEFAULT_COMIC_MAX_PANELS)
        self.assertEqual(DESKTOP_DEFAULT_HIGHLIGHT_TARGET_SECONDS, DEFAULT_HIGHLIGHT_TARGET_SECONDS)
        self.assertEqual(DESKTOP_DEFAULT_HIGHLIGHT_MAX_SEGMENTS, DEFAULT_HIGHLIGHT_MAX_SEGMENTS)

    def test_session_pipeline_config_defaults_match_shared_defaults(self) -> None:
        config = SessionPipelineConfig(session_path=__import__("pathlib").Path("/tmp/session"))
        self.assertEqual(config.summary_model, DEFAULT_SUMMARY_TEXT_MODEL)
        self.assertEqual(config.story_model, DEFAULT_STORY_TEXT_MODEL)
        self.assertEqual(config.summary_thinking, DEFAULT_SUMMARY_THINKING)
        self.assertEqual(config.story_thinking, DEFAULT_STORY_THINKING)
        self.assertEqual(config.audio_concurrency, DEFAULT_AUDIO_CONCURRENCY)
        self.assertEqual(config.video_concurrency, DEFAULT_VIDEO_CONCURRENCY)

    def test_cli_defaults_match_shared_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run-session", "--session", "/tmp/session"])
        self.assertEqual(args.summary_model, DEFAULT_SUMMARY_TEXT_MODEL)
        self.assertEqual(args.story_model, DEFAULT_STORY_TEXT_MODEL)
        self.assertEqual(args.summary_thinking, DEFAULT_SUMMARY_THINKING)
        self.assertEqual(args.story_thinking, DEFAULT_STORY_THINKING)
        self.assertEqual(args.audio_concurrency, DEFAULT_AUDIO_CONCURRENCY)
        self.assertEqual(args.video_concurrency, DEFAULT_VIDEO_CONCURRENCY)

        comic_args = parser.parse_args(["build-comic-products", "--run-dir", "/tmp/run"])
        self.assertEqual(comic_args.max_panels, DEFAULT_COMIC_MAX_PANELS)
        self.assertEqual(comic_args.comic_thinking, DEFAULT_COMIC_THINKING)

        highlight_args = parser.parse_args(["build-highlight-video", "--run-dir", "/tmp/run"])
        self.assertEqual(highlight_args.target_seconds, DEFAULT_HIGHLIGHT_TARGET_SECONDS)
        self.assertEqual(highlight_args.max_segments, DEFAULT_HIGHLIGHT_MAX_SEGMENTS)
        self.assertEqual(highlight_args.highlight_thinking, DEFAULT_HIGHLIGHT_THINKING)
