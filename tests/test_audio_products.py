from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from life_report.audio_products import _story_input_system_prompt, build_audio_products, build_audio_timeline


class AudioProductsTest(unittest.TestCase):
    def test_build_audio_timeline_converts_relative_time(self) -> None:
        timeline = build_audio_timeline(
            [
                {
                    "audio_id": "1",
                    "source_path": "/tmp/audio_1.m4a",
                    "confidence": 0.9,
                    "understanding": {
                        "event_candidates": [
                            {
                                "event_type": "walking",
                                "time_range": "00:35 - 02:30",
                                "description": "I walked outside.",
                                "confidence": 0.8,
                            }
                        ],
                        "important_moments": [
                            {"time": "00:01", "description": "A short conversation."}
                        ],
                        "transcript": [
                            {"start": "00:01.000", "end": "00:03.000", "text": "hello"}
                        ],
                    },
                    "metadata": {
                        "start_utc_sec": 1000.0,
                        "end_utc_sec": 1300.0,
                        "duration_sec": 300.0,
                    },
                }
            ]
        )

        self.assertEqual(timeline["events"][0]["absolute_start_utc_sec"], 1035.0)
        self.assertEqual(timeline["events"][0]["absolute_end_utc_sec"], 1150.0)
        self.assertEqual(timeline["moments"][0]["absolute_start_utc_sec"], 1001.0)

    def test_build_audio_products_without_provider(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_path = base / "audio_understandings.json"
            output = base / "out"
            location_compact = base / "location_compact_raw.txt"
            motion_compact = base / "motion_compact_raw.txt"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "audio_id": "1",
                            "source_path": "/tmp/audio_1.m4a",
                            "understanding": {"life_story_hint": "我走出门。"},
                            "metadata": {"start_utc_sec": 1000.0, "end_utc_sec": 1300.0},
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            location_compact.write_text("Overall route map: /tmp/route.png\n地点：公园附近", encoding="utf-8")
            motion_compact.write_text("运动：walking_like", encoding="utf-8")

            result = build_audio_products(
                input_path,
                output,
                provider="none",
                location_compact_path=location_compact,
                motion_compact_path=motion_compact,
            )
            self.assertTrue(Path(result["audio_timeline_path"]).exists())
            self.assertTrue(Path(result["audio_compact_raw_path"]).exists())
            self.assertTrue(Path(result["audio_story_input_path"]).exists())
            self.assertTrue(result["context_injected"])
            self.assertIn("地点：公园附近", Path(result["audio_story_input_path"]).read_text(encoding="utf-8"))

    def test_audio_summary_prompt_requires_chinese_first_person(self) -> None:
        prompt = _story_input_system_prompt()

        self.assertIn("必须使用简体中文", prompt)
        self.assertIn("第一人称", prompt)
        self.assertIn("不要写“用户”", prompt)
        self.assertIn("我听到", prompt)

    def test_audio_summary_disables_thinking(self) -> None:
        calls = {}

        class Model:
            def generate_text(self, system_prompt: str, user_prompt: str) -> str:
                return "ok"

        def fake_create(provider: str, model=None, enable_thinking: bool = False):
            calls["enable_thinking"] = enable_thinking
            return Model()

        with patch("life_report.audio_products.create_text_llm", side_effect=fake_create):
            from life_report.audio_products import build_audio_story_input

            self.assertEqual(build_audio_story_input("raw", "qwen3.7-plus", "aliyun"), "ok")

        self.assertFalse(calls["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
