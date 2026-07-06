import json
import re
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from life_report.context_store import ContextStore, ProductPaths
from life_report.pipeline_defaults import DEFAULT_STORY_THINKING
from life_report.story_products import (
    _frame_text_bonus,
    _story_title,
    _claim_review_system_prompt,
    _story_system_prompt,
    build_story_products,
    render_story_html,
    story_evidence_rules,
    story_review_rules,
    story_schema_rules,
    story_style_rules,
)


class StoryProductsTest(unittest.TestCase):
    def test_context_store_queries_time_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            products = _write_minimal_products(base)

            store = ContextStore(products)
            context = store.query_context(1000.0, 1010.0)

            self.assertEqual(context["audio"]["segments"][0]["audio_id"], "1")
            self.assertEqual(context["video"]["clips"][0]["clip_id"], "1")
            self.assertEqual(context["location"]["segments"][0]["segment_id"], "loc1")
            self.assertEqual(context["motion"]["segments"][0]["state"], "walking_like")

    def test_build_story_products_writes_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            products = _write_minimal_products(base)
            output = base / "story"

            result = build_story_products(
                audio_products_dir=products.audio_products_dir,
                video_products_dir=products.video_products_dir,
                location_products_dir=products.location_products_dir,
                motion_products_dir=products.motion_products_dir,
                output_dir=output,
                provider="none",
                max_keyframes=1,
            )

            self.assertTrue(Path(result["story_evidence_pack_path"]).exists())
            self.assertTrue(Path(result["story_input_path"]).exists())
            self.assertTrue(Path(result["life_story_markdown_path"]).exists())
            self.assertTrue(Path(result["life_story_html_path"]).exists())
            self.assertEqual(result["selected_keyframe_count"], 1)
            story_input = Path(result["story_input_path"]).read_text(encoding="utf-8")
            self.assertIn("柳浪家园", story_input)
            self.assertIn("walking_like", story_input)
            self.assertIn("/tmp/kf.jpg", story_input)

    def test_story_prompts_keep_first_person_life_story_style(self) -> None:
        story_prompt = _story_system_prompt()
        review_prompt = _claim_review_system_prompt()

        self.assertTrue(story_schema_rules())
        self.assertTrue(story_style_rules())
        self.assertTrue(story_evidence_rules())
        self.assertTrue(story_review_rules())
        self.assertIn("第一人称生活小记", story_prompt)
        self.assertIn("我看到", story_prompt)
        self.assertIn("我听到", story_prompt)
        self.assertIn("不要输出 JSON", story_prompt)
        self.assertIn("不要把这些全部改成无主语旁观句", review_prompt)
        self.assertIn("本人生活小记", review_prompt)

    def test_final_story_enables_thinking(self) -> None:
        calls = []

        class Model:
            def generate_text(self, system_prompt: str, user_prompt: str) -> str:
                return "## 一天总览\n我记录到一个片段。\n"

        def fake_create(provider: str, model=None, enable_thinking: bool = False):
            calls.append(enable_thinking)
            return Model()

        with patch("life_report.story_products.create_text_llm", side_effect=fake_create):
            from life_report.story_products import generate_life_story

            result = generate_life_story("evidence", provider="aliyun", story_model="qwen3.7-plus")

        self.assertIn("我记录到", result)
        self.assertEqual(calls, [DEFAULT_STORY_THINKING])

    def test_story_title_falls_back_to_generic_without_dataset_specific_heuristics(self) -> None:
        markdown_text = "# 我的个人生活故事报告\n\n我在一个下雨的下午出门散步。\n"
        self.assertEqual(_story_title(markdown_text), "今天的小片段")

    def test_frame_text_bonus_uses_generic_overlap_not_scene_buckets(self) -> None:
        score = _frame_text_bonus(
            "我在车站和孩子告别，然后独自返程。",
            {"caption": "车站送别后的返程镜头", "reason": "孩子与车站场景都清楚可见"},
        )
        self.assertGreater(score, 0.0)

    def test_render_story_html_hides_evidence_from_visible_text(self) -> None:
        html = render_story_html(
            "\n".join(
                [
                    "# 我的个人生活故事报告",
                    "## 一天总览",
                    "我在路上散步。",
                    "## 按时间的故事线",
                    "**14:10 - 14:15 | 出门**",
                    "我拿出雨伞。",
                    "**14:15 - 14:20 | 雨中散步**",
                    "我继续往前走。",
                    "## 人物和对话",
                    "这里是背后分析。",
                    "## 物品",
                    "雨伞。",
                    "## 情绪和状态",
                    "放松。",
                    "## 开放问题",
                    "- **Audio 4 中的对话对象**：是谁在说话？",
                    "- **Clip 8/9 中的便利店**：具体是哪一家？",
                    "## 证据索引",
                    "- **Audio 1**: raw evidence",
                    "- `loc_seg_0001`: raw map",
                ]
            ),
            {
                "time_range": {"start_local_time": "2026-06-23 14:10:01"},
                "media": {
                    "overall_route_map": "/tmp/route.png",
                    "selected_keyframes": [{"keyframe_path": "/tmp/kf.jpg", "caption": "雨伞画面"}],
                },
            },
        )
        visible_text = re.sub(r"<style.*?</style>", "", html, flags=re.S)
        visible_text = re.sub(r"<[^>]+>", " ", visible_text)

        self.assertNotIn("证据索引", visible_text)
        self.assertNotIn("Audio 1", visible_text)
        self.assertNotIn("Audio 4", visible_text)
        self.assertNotIn("Clip 8", visible_text)
        self.assertNotIn("loc_seg", visible_text)
        self.assertNotIn("背后分析", visible_text)
        self.assertNotIn("对话对象", visible_text)
        self.assertNotIn("便利店", visible_text)
        self.assertNotIn("关键事件", visible_text)
        self.assertIn("story-moment", html)
        self.assertIn("我拿出雨伞", visible_text)

    def test_render_story_html_keeps_intro_storyline_moments(self) -> None:
        html = render_story_html(
            "\n".join(
                [
                    "# 雨中的小日子",
                    "",
                    "午后出门散步，雨越下越大。",
                    "",
                    "**14:10 - 14:15 | 撑伞出门**",
                    "我从小区出发，撑开雨伞。",
                    "",
                    "**14:15 - 14:25 | 超市避雨**",
                    "我走进超市，在货架间停了一会儿。",
                    "",
                    "### 关键事件",
                    "- 14:10 撑伞出门。",
                ]
            ),
            {
                "time_range": {"start_local_time": "2026-06-23 14:10:01"},
                "media": {
                    "selected_keyframes": [
                        {
                            "keyframe_path": "/tmp/umbrella.jpg",
                            "caption": "撑伞出门",
                            "local_time": "2026-06-23 14:10:03",
                        },
                        {
                            "keyframe_path": "/tmp/store.jpg",
                            "caption": "超市货架",
                            "local_time": "2026-06-23 14:20:03",
                        },
                    ],
                },
            },
        )

        visible_text = re.sub(r"<style.*?</style>", "", html, flags=re.S)
        visible_text = re.sub(r"<[^>]+>", " ", visible_text)
        self.assertIn("故事线", visible_text)
        self.assertIn("撑伞出门", visible_text)
        self.assertIn("超市避雨", visible_text)
        self.assertEqual(html.count('class="story-moment"'), 2)
        self.assertNotIn("高光片段", visible_text)
        self.assertNotIn("关键事件", visible_text)

    def test_render_story_html_selects_images_by_story_time(self) -> None:
        html = render_story_html(
            "\n".join(
                [
                    "# 雨中的小日子",
                    "## 按时间的故事线",
                    "**14:10 - 14:15 | 撑伞出门**",
                    "我从小区出发。",
                    "**14:20 - 14:25 | 超市避雨**",
                    "我走进超市。",
                ]
            ),
            {
                "time_range": {"start_local_time": "2026-06-23 14:10:01"},
                "media": {
                    "selected_keyframes": [
                        {
                            "keyframe_path": "/tmp/wrong.jpg",
                            "caption": "远处画面",
                            "local_time": "2026-06-23 14:40:03",
                            "quality_score": 1.0,
                        },
                        {
                            "keyframe_path": "/tmp/umbrella.jpg",
                            "caption": "撑伞出门",
                            "local_time": "2026-06-23 14:12:03",
                            "quality_score": 0.7,
                        },
                        {
                            "keyframe_path": "/tmp/store.jpg",
                            "caption": "超市货架",
                            "local_time": "2026-06-23 14:22:03",
                            "quality_score": 0.7,
                        },
                    ],
                },
            },
        )

        self.assertIn("/tmp/umbrella.jpg", html)
        self.assertIn("/tmp/store.jpg", html)
        self.assertNotIn("/tmp/wrong.jpg", html)


def _write_minimal_products(base: Path) -> ProductPaths:
    audio = base / "audio"
    video = base / "video"
    location = base / "location"
    motion = base / "motion"
    for path in [audio, video, location, motion]:
        path.mkdir()

    _write_json(
        audio / "audio_timeline.json",
        {
            "time_range": {"start_utc_sec": 1000.0, "end_utc_sec": 1300.0},
            "segments": [
                {"audio_id": "1", "start_utc_sec": 1000.0, "end_utc_sec": 1300.0, "scene_summary": "我在小区里走路。"}
            ],
            "events": [
                {
                    "event_id": "ae1",
                    "audio_id": "1",
                    "absolute_start_utc_sec": 1002.0,
                    "absolute_end_utc_sec": 1008.0,
                    "summary": "听到脚步声和说话声。",
                }
            ],
            "moments": [],
            "todo_candidates": [{"audio_id": "1", "candidate": "买伞"}],
            "memory_candidates": [],
        },
    )
    (audio / "audio_story_input.txt").write_text("Audio Story Input: 我在柳浪家园附近步行。", encoding="utf-8")
    (audio / "audio_compact_raw.txt").write_text("audio compact", encoding="utf-8")

    _write_json(
        video / "video_timeline.json",
        {
            "time_range": {"start_utc_sec": 1000.0, "end_utc_sec": 1010.0},
            "clips": [
                {
                    "clip_id": "1",
                    "start_utc_sec": 1000.0,
                    "end_utc_sec": 1010.0,
                    "clip_summary": "我拿出雨伞。",
                    "report_video_path": "/tmp/report.mp4",
                }
            ],
            "events": [],
            "keyframes": [
                {
                    "keyframe_id": "kf1",
                    "clip_id": "1",
                    "absolute_utc_sec": 1003.0,
                    "local_time": "2026-06-23 14:10:03",
                    "keyframe_path": "/tmp/kf.jpg",
                    "caption": "雨伞画面",
                }
            ],
            "todo_candidates": [],
            "memory_candidates": [{"clip_id": "1", "candidate": "出门带伞"}],
        },
    )
    _write_json(
        video / "video_story_media_manifest.json",
        {
            "selected_keyframes": [
                {
                    "keyframe_id": "kf1",
                    "clip_id": "1",
                    "local_time": "2026-06-23 14:10:03",
                    "keyframe_path": "/tmp/kf.jpg",
                    "caption": "雨伞画面",
                }
            ],
            "selected_report_videos": [{"clip_id": "1", "path": "/tmp/report.mp4"}],
        },
    )
    (video / "video_story_input.txt").write_text("Video Story Input: 我拿出雨伞。", encoding="utf-8")
    (video / "video_compact_raw.txt").write_text("video compact", encoding="utf-8")

    _write_json(
        location / "location_timeline.json",
        {
            "time_range": {"start_utc_sec": 999.0, "end_utc_sec": 1020.0},
            "overall_map_image": "/tmp/route.png",
            "summary": {"dominant_quality": "good"},
            "segments": [
                {
                    "segment_id": "loc1",
                    "start_utc_sec": 999.0,
                    "end_utc_sec": 1020.0,
                    "start_local_time": "2026-06-23 14:09:59",
                    "end_local_time": "2026-06-23 14:10:20",
                    "movement": "walking",
                    "quality": "good",
                    "amap": {"address": "北京市海淀区柳浪家园", "roads": ["树村北街"]},
                }
            ],
        },
    )
    _write_json(location / "clip_location_context.json", {"video_clips": [], "audio_segments": []})
    (location / "location_compact_raw.txt").write_text("地点：柳浪家园", encoding="utf-8")

    _write_json(
        motion / "motion_timeline.json",
        {
            "time_range": {"start_utc_sec": 999.0, "end_utc_sec": 1020.0},
            "summary": {"dominant_state": "walking_like"},
            "segments": [
                {
                    "segment_id": "motion1",
                    "start_utc_sec": 999.0,
                    "end_utc_sec": 1020.0,
                    "start_local_time": "2026-06-23 14:09:59",
                    "end_local_time": "2026-06-23 14:10:20",
                    "state": "walking_like",
                    "intensity": "moderate",
                    "stability": "shaky",
                }
            ],
        },
    )
    _write_json(motion / "clip_motion_context.json", {"video_clips": [], "audio_segments": []})
    (motion / "motion_compact_raw.txt").write_text("运动：walking_like", encoding="utf-8")

    return ProductPaths(audio, video, location, motion)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
