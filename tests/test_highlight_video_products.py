import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from life_report.highlight_video_products import (
    _highlight_system_prompt,
    build_highlight_video_products,
    build_highlight_storyline,
    highlight_evidence_rules,
    highlight_schema_rules,
    highlight_style_rules,
)
from life_report.pipeline_defaults import (
    DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
    DEFAULT_HIGHLIGHT_TARGET_SECONDS,
    DEFAULT_HIGHLIGHT_THINKING,
)


class HighlightVideoProductsTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_build_highlight_video_products_writes_outputs_with_model_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            _write_run_fixture(run_dir)
            model = _FakeTextModel(
                {
                    "title": "雨中高光",
                    "date_label": "2026年6月23日",
                    "summary": "从撑伞出门到雨中折返，再回到玄关。",
                    "beats": [
                        {
                            "order": 1,
                            "story_beat": "撑伞从小区出门。",
                            "clip_id": "1",
                            "start_sec": 0,
                            "end_sec": 4,
                            "caption": "撑伞出门",
                            "reason": "开场动作清楚。",
                        },
                        {
                            "order": 2,
                            "story_beat": "雨中折返，回到玄关。",
                            "clip_id": "2",
                            "start_sec": 1,
                            "end_sec": 5,
                            "caption": "雨中收尾",
                            "reason": "收尾转折明确。",
                        },
                    ],
                }
            )

            with patch("life_report.highlight_video_products.create_text_llm", return_value=model):
                result = build_highlight_video_products(
                    run_dir=run_dir,
                    provider="aliyun",
                    target_seconds=16,
                    max_segments=2,
                )

            self.assertTrue(Path(result["highlight_storyline_path"]).exists())
            self.assertTrue(Path(result["highlight_plan_path"]).exists())
            self.assertTrue(Path(result["highlight_video_path"]).exists())
            self.assertTrue(Path(result["highlight_video_html_path"]).exists())
            width, height = _probe_video_size(Path(result["highlight_video_path"]))
            self.assertEqual((width, height), (1920, 1080))
            audio = _probe_audio_stream(Path(result["highlight_video_path"]))
            self.assertEqual(audio["sample_rate"], "44100")
            self.assertEqual(audio["channels"], 2)
            plan = json.loads(Path(result["highlight_plan_path"]).read_text(encoding="utf-8"))
            self.assertLessEqual(len(plan["segments"]), 2)
            self.assertEqual([item["clip_id"] for item in plan["segments"]], ["1", "2"])
            for segment in plan["segments"]:
                self.assertTrue(Path(segment["source_path"]).exists())
                self.assertLess(segment["start_sec"], segment["end_sec"])
                self.assertIn("caption", segment)
                self.assertIn("reason", segment)

    def test_highlight_storyline_requires_model_provider(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires provider='aliyun'"):
            build_highlight_storyline(
                story_markdown="# Story",
                story_json={},
                evidence_pack={},
                report_videos=[],
                provider="none",
                text_model="mock",
                target_seconds=20,
                max_segments=2,
            )

    def test_highlight_prompt_rules_are_structured(self) -> None:
        prompt = _highlight_system_prompt(DEFAULT_HIGHLIGHT_MAX_SEGMENTS)
        self.assertTrue(highlight_schema_rules(DEFAULT_HIGHLIGHT_MAX_SEGMENTS))
        self.assertTrue(highlight_style_rules())
        self.assertTrue(highlight_evidence_rules())
        self.assertIn("clip_id", prompt)
        self.assertIn("第一人称生活记录视角", prompt)

    def test_highlight_uses_default_policy_for_thinking(self) -> None:
        calls = []

        class Model:
            def generate_text(self, _system_prompt: str, _user_prompt: str) -> str:
                return json.dumps({"title": "ok", "beats": [{"clip_id": "1", "caption": "ok", "story_beat": "ok", "reason": "ok"}]}, ensure_ascii=False)

        def fake_create(provider: str, model=None, enable_thinking: bool = False):
            calls.append(enable_thinking)
            return Model()

        with patch("life_report.highlight_video_products.create_text_llm", side_effect=fake_create):
            with self.assertRaisesRegex(RuntimeError, "invalid clip_id"):
                build_highlight_storyline(
                    story_markdown="# Story",
                    story_json={},
                    evidence_pack={},
                    report_videos=[],
                    provider="aliyun",
                    text_model="mock",
                    target_seconds=DEFAULT_HIGHLIGHT_TARGET_SECONDS,
                    max_segments=DEFAULT_HIGHLIGHT_MAX_SEGMENTS,
                )

        self.assertEqual(calls, [DEFAULT_HIGHLIGHT_THINKING])

    def test_highlight_storyline_rejects_invalid_model_clip_id(self) -> None:
        model = _FakeTextModel({"title": "Bad", "beats": [{"clip_id": "missing", "caption": "bad"}]})
        with patch("life_report.highlight_video_products.create_text_llm", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "invalid clip_id"):
                build_highlight_storyline(
                    story_markdown="# Story",
                    story_json={},
                    evidence_pack={},
                    report_videos=[
                        {
                            "clip_id": "1",
                            "path": "/tmp/clip.mp4",
                            "duration_sec": 4,
                            "summary": "ok",
                            "local_time_range": "",
                        }
                    ],
                    provider="aliyun",
                    text_model="mock",
                    target_seconds=20,
                    max_segments=2,
                )

    def test_highlight_storyline_supplements_short_model_plan(self) -> None:
        model = _FakeTextModel(
            {
                "title": "长序列高光",
                "beats": [
                    {"order": 1, "clip_id": "1", "caption": "开头", "story_beat": "开头", "reason": "开头清楚"},
                    {"order": 2, "clip_id": "4", "caption": "中段", "story_beat": "中段", "reason": "中段清楚"},
                    {"order": 3, "clip_id": "8", "caption": "结尾", "story_beat": "结尾", "reason": "结尾清楚"},
                ],
            }
        )
        report_videos = [
            {
                "clip_id": str(index),
                "path": f"/tmp/clip{index}.mp4",
                "duration_sec": 6,
                "summary": f"片段 {index}",
                "local_time_range": "",
            }
            for index in range(1, 9)
        ]

        with patch("life_report.highlight_video_products.create_text_llm", return_value=model):
            storyline = build_highlight_storyline(
                story_markdown="# Story",
                story_json={},
                evidence_pack={},
                report_videos=report_videos,
                provider="aliyun",
                text_model="mock",
                target_seconds=45,
                max_segments=8,
            )

        self.assertEqual(len(storyline["beats"]), 7)
        self.assertEqual(len({beat["clip_id"] for beat in storyline["beats"]}), 7)
        self.assertTrue(any("自动补足" in beat["reason"] for beat in storyline["beats"]))


class _FakeTextModel:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def generate_text(self, _system_prompt: str, _user_prompt: str) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def _write_run_fixture(run_dir: Path) -> None:
    (run_dir / "story").mkdir(parents=True)
    (run_dir / "video" / "products").mkdir(parents=True)
    video_dir = run_dir / "video" / "probe" / "report_video"
    video_dir.mkdir(parents=True)
    story = "\n".join(
        [
            "## 一天总览",
            "午后撑伞出门，雨中折返，最后回到玄关。",
            "## 按时间的故事线",
            "**14:10 - 14:12 | 撑伞出门**",
            "我从小区出发。",
            "**14:30 - 14:35 | 雨中收尾**",
            "我回到玄关。",
        ]
    )
    (run_dir / "story" / "life_story.md").write_text(story, encoding="utf-8")
    _write_json(run_dir / "story" / "life_story.json", {"report_markdown": story})
    _write_json(run_dir / "story" / "story_evidence_pack.json", {"time_range": {"start_local_time": "2026-06-23 14:10:01"}})
    clips = []
    colors = ["#203060", "#606020"]
    for index, color in enumerate(colors, start=1):
        video_path = video_dir / f"clip_{index}_report.mp4"
        _write_test_video(video_path, color=color, duration=6)
        clips.append(
            {
                "clip_id": str(index),
                "local_time_range": f"2026-06-23 14:{index:02d}:00 - 2026-06-23 14:{index:02d}:06",
                "path": str(video_path),
                "summary": "撑伞出门" if index == 1 else "雨中回家",
            }
        )
    _write_json(run_dir / "video" / "products" / "video_story_media_manifest.json", {"selected_report_videos": clips})


def _write_test_video(path: Path, color: str, duration: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=640x960:r=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _probe_video_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _probe_audio_stream(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(result.stdout)["streams"][0]


if __name__ == "__main__":
    unittest.main()
