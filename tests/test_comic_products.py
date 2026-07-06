import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from life_report.comic_products import (
    build_comic_panel_reference_plan,
    build_comic_products,
    build_comic_reference_selection,
    build_comic_storyline,
    build_seedream_comic_prompt,
    comic_evidence_rules,
    comic_schema_rules,
    comic_style_rules,
    render_comic_card,
    render_comic_collage,
    select_comic_reference_images,
    _normalize_comic_storyline,
    _scene_role_from_text,
)
from life_report.pipeline_defaults import DEFAULT_COMIC_MAX_PANELS


class ComicProductsTest(unittest.TestCase):
    def test_comic_storyline_caps_dynamic_panels_at_nine(self) -> None:
        story = "\n".join(
            [
                "# 很长的一天",
                "## 关键事件",
                *[f"- 14:{index:02d} 第 {index} 个生活片段。" for index in range(12)],
            ]
        )

        storyline = build_comic_storyline(
            story_markdown=story,
            story_json={},
            evidence_pack={"time_range": {"start_local_time": "2026-06-23 14:00:00"}},
            media_manifest={"selected_keyframes": []},
            provider="none",
            text_model="mock",
            max_panels=DEFAULT_COMIC_MAX_PANELS,
        )

        self.assertEqual(storyline["candidate_panel_count"], 12)
        self.assertEqual(storyline["panel_count"], DEFAULT_COMIC_MAX_PANELS)
        self.assertEqual(len(storyline["panels"]), DEFAULT_COMIC_MAX_PANELS)

    def test_comic_storyline_splits_cross_scene_panel(self) -> None:
        storyline = _normalize_comic_storyline(
            {
                "title": "回家路上",
                "panels": [
                    {
                        "time_hint": "14:36",
                        "story_beat": "跑进楼栋大厅后，我在玄关整理包袋和鞋子。",
                        "visual_focus": "楼栋大厅转到玄关。",
                        "required_elements": ["楼栋大厅", "玄关", "包袋", "鞋子"],
                    }
                ],
            },
            fallback={
                "title": "回家路上",
                "caption": "回家。",
                "tag": "生活漫画",
                "date_label": "2026年6月23日",
                "storyline": "回家。",
                "forbidden": [],
                "panels": [],
            },
            max_panels=9,
        )

        self.assertEqual(len(storyline["panels"]), 2)
        self.assertIn("楼栋大厅", storyline["panels"][0]["story_beat"])
        self.assertIn("玄关", storyline["panels"][1]["story_beat"])

    def test_scene_role_extraction_is_generic(self) -> None:
        self.assertEqual(_scene_role_from_text("14:10 出门上路，穿过路口。", 0, 5), "opening")
        self.assertEqual(_scene_role_from_text("暴雨突然打下来，我冲进门厅。", 2, 5), "action_peak")
        self.assertEqual(_scene_role_from_text("手里的雨滴和钥匙特写。", 2, 5), "closeup")
        self.assertEqual(_scene_role_from_text("回到屋里收伞整理鞋子。", 4, 5), "closing")

    def test_reference_selection_respects_max_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            frames = []
            for index in range(6):
                image_path = base / f"frame_{index}.jpg"
                Image.new("RGB", (320, 240), (index * 20, 80, 120)).save(image_path)
                frames.append(
                    {
                        "clip_id": str(index),
                        "keyframe_path": str(image_path),
                        "quality_score": 0.9 - index * 0.01,
                        "clip_summary": "雨伞 便利店 玄关" if index == 0 else "普通道路",
                    }
                )

            selected = select_comic_reference_images(
                {"storyline": "雨伞和便利店", "visual_anchors": ["雨伞", "便利店"], "panels": ["撑伞出门", "店内避雨"]},
                {},
                {"selected_keyframes": frames},
                max_count=2,
            )

            self.assertEqual(len(selected), 2)
            self.assertEqual(selected[0].name, "frame_0.jpg")

    def test_reference_selection_outputs_generic_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            summaries = [
                ("1", "居民楼门口刚出发，路边停着车。"),
                ("2", "树下人行道步行，环境开阔。"),
                ("3", "商店货架和冷柜，像是临时进去躲雨。"),
                ("4", "伞面近景，雨滴打在黑伞上。"),
                ("5", "室内鞋架和包袋，已经回到家中。"),
            ]
            frames = []
            for index, (clip_id, summary) in enumerate(summaries, start=1):
                image_path = base / f"frame_{index}.jpg"
                Image.new("RGB", (320, 240), (index * 20, 80, 120)).save(image_path)
                frames.append(
                    {
                        "clip_id": clip_id,
                        "keyframe_id": f"k{index}",
                        "keyframe_path": str(image_path),
                        "quality_score": 0.9,
                        "clip_summary": summary,
                        "local_time": f"2026-06-23 14:{10 + index}:00",
                    }
                )

            plan = build_comic_reference_selection(
                {
                    "storyline": "出门路上遇雨，短暂躲进店里，最后回家收尾。",
                    "visual_anchors": ["出门", "躲雨", "伞面", "回家"],
                    "panels": ["14:11 刚出门", "14:13 进店躲雨", "14:14 伞面雨滴", "14:15 回家收尾"],
                },
                {},
                {"selected_keyframes": frames},
                max_count=4,
            )

            self.assertEqual(len(plan["selected_references"]), 4)
            self.assertEqual(plan["selection_strategy"], "storyboard_fragment_coverage.v2")
            for row in plan["selected_references"]:
                self.assertIn("panel_role", row)
                self.assertIn("match_features", row)
                self.assertIn("score_breakdown", row)
                self.assertIn("reason", row)
                self.assertNotIn("bucket", row)

    def test_panel_reference_selection_prefers_scene_boundary_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            store_path = base / "store.jpg"
            entry_path = base / "entry.jpg"
            foyer_path = base / "foyer.jpg"
            Image.new("RGB", (320, 240), (80, 90, 120)).save(store_path)
            Image.new("RGB", (320, 240), (90, 100, 130)).save(entry_path)
            Image.new("RGB", (320, 240), (100, 110, 140)).save(foyer_path)
            frames = [
                {
                    "clip_id": "1",
                    "keyframe_id": "store_1",
                    "keyframe_path": str(store_path),
                    "local_time": "2026-06-23 14:24:00",
                    "quality_score": 0.95,
                    "clip_summary": "商店店内货架旁，门口外面有雨伞和街道。",
                },
                {
                    "clip_id": "2",
                    "keyframe_id": "entry_1",
                    "keyframe_path": str(entry_path),
                    "local_time": "2026-06-23 14:31:00",
                    "quality_score": 0.92,
                    "clip_summary": "楼门口铁门与指示牌，人在门外避雨准备进楼里。",
                },
                {
                    "clip_id": "3",
                    "keyframe_id": "foyer_1",
                    "keyframe_path": str(foyer_path),
                    "local_time": "2026-06-23 14:33:00",
                    "quality_score": 0.90,
                    "clip_summary": "家中玄关鞋架、包袋，已经回到室内收尾。",
                },
            ]

            plan = build_comic_panel_reference_plan(
                storyline={
                    "panels": [
                        {
                            "panel_id": "panel_01",
                            "order": 1,
                            "time_hint": "14:31",
                            "story_beat": "冲到楼门口准备进楼避雨。",
                            "visual_focus": "楼门口和门外过渡。",
                            "required_elements": ["楼门口", "门外"],
                            "reference_queries": ["楼门口避雨"],
                        },
                        {
                            "panel_id": "panel_02",
                            "order": 2,
                            "time_hint": "14:33",
                            "story_beat": "回到玄关收伞整理包袋。",
                            "visual_focus": "玄关鞋架和包袋。",
                            "required_elements": ["玄关", "鞋架", "包袋"],
                            "reference_queries": ["玄关收尾"],
                        },
                    ]
                },
                evidence_pack={"media": {"selected_keyframes": []}},
                media_manifest={"selected_keyframes": frames},
                provider="none",
                max_references_per_panel=1,
                max_total_references=2,
            )

            self.assertEqual(plan["panels"][0]["references"][0]["keyframe_id"], "entry_1")
            self.assertEqual(plan["panels"][1]["references"][0]["keyframe_id"], "foyer_1")
            self.assertEqual(plan["selected_references"][0]["panel_role"], "opening")
            self.assertIn("score_breakdown", plan["selected_references"][0])

    def test_seedream_prompt_uses_structured_rules(self) -> None:
        prompt = build_seedream_comic_prompt(
            {
                "title": "雨中小漫步",
                "storyline": "撑伞出门，便利店避雨，最后回到玄关。",
                "forbidden": ["不要虚构对话"],
                "panels": [
                    {
                        "panel_id": "panel_01",
                        "order": 1,
                        "story_beat": "撑伞从小区出发。",
                        "visual_focus": "黑色雨伞和小区道路。",
                        "required_elements": ["黑色雨伞", "小区"],
                    }
                ],
            },
            {"panels": [{"panel_id": "panel_01", "references": [{"clip_summary": "小区道路和黑色雨伞"}]}]},
        )

        self.assertTrue(comic_schema_rules())
        self.assertTrue(comic_style_rules("半写实", "布局建议"))
        self.assertTrue(comic_evidence_rules())
        self.assertIn("活泼但克制的故事书漫画页", prompt)
        self.assertIn("不要画成规则九宫格", prompt)
        self.assertIn("完全不要生成任何文字", prompt)
        self.assertIn("不要新增陌生人", prompt)

    def test_build_comic_products_mock_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            _write_run_fixture(run_dir)

            result = build_comic_products(run_dir=run_dir, provider="none")

            self.assertTrue(Path(result["comic_storyboard_path"]).exists())
            self.assertTrue(Path(result["selected_references_path"]).exists())
            self.assertTrue(Path(result["daily_comic_panel_path"]).exists())
            self.assertTrue(Path(result["daily_comic_card_path"]).exists())
            self.assertTrue(Path(result["daily_comic_html_path"]).exists())
            self.assertTrue(Path(result["layout_draft_path"]).exists())
            self.assertEqual(result["reference_image_count"], result["panel_count"])
            with Image.open(result["daily_comic_path"]) as image:
                self.assertEqual(image.size, (1080, 1680))

    def test_render_comic_card_preserves_portrait_panel_aspect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            panel_path = base / "portrait_panel.png"
            Image.new("RGB", (900, 1400), (80, 120, 160)).save(panel_path)
            output_path = base / "daily_comic_card.png"

            render_comic_card(
                panel_path,
                {
                    "title": "雨中小漫步",
                    "caption": "午后刚出门，雷雨就追了上来。我撑着伞穿过公园，又在便利店短暂避雨，最后带着雨声回到玄关。",
                    "tag": "雨中漫步",
                    "date_label": "2026年6月23日",
                },
                output_path,
                include_text=True,
            )

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (1080, 1680))

    def test_render_comic_collage_writes_square_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            frames = []
            for index in range(5):
                image_path = base / f"comic_frame_{index}.png"
                Image.new("RGB", (900 + index * 30, 700), (40 * index, 90, 160)).save(image_path)
                frames.append(image_path)

            output_path = base / "daily_comic_panel.png"
            render_comic_collage(frames, output_path)

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (2048, 2048))


def _write_run_fixture(run_dir: Path) -> None:
    (run_dir / "story").mkdir(parents=True)
    (run_dir / "video" / "products").mkdir(parents=True)
    keyframe_dir = run_dir / "video" / "probe" / "report_keyframes"
    keyframe_dir.mkdir(parents=True)
    story = "\n".join(
        [
            "# 雨中小漫步",
            "## 一天总览",
            "午后出门散步，雷雨突然落下。我撑着黑伞路过便利店，最后回到玄关整理绳子。",
            "## 故事线",
            "**14:10 - 14:12 | 刚出门**",
            "我撑伞从楼下出发。",
            "**14:15 - 14:18 | 进店躲雨**",
            "我短暂走进便利店。",
            "**14:20 - 14:23 | 回家收尾**",
            "我回到玄关收伞。",
        ]
    )
    (run_dir / "story" / "life_story.md").write_text(story, encoding="utf-8")
    _write_json(run_dir / "story" / "life_story.json", {"report_markdown": story, "story_model": "mock"})
    _write_json(
        run_dir / "story" / "story_evidence_pack.json",
        {
            "time_range": {"start_local_time": "2026-06-23 14:10:01"},
            "media": {"selected_keyframes": []},
        },
    )
    frames = []
    rows = [
        ("1", "2026-06-23 14:10:02", "黑色雨伞和楼下道路"),
        ("2", "2026-06-23 14:15:03", "店内货架和冷柜"),
        ("3", "2026-06-23 14:17:10", "伞面上的雨滴近景"),
        ("4", "2026-06-23 14:20:12", "玄关鞋架和包袋"),
        ("5", "2026-06-23 14:12:04", "居民楼门口和道路"),
    ]
    for index, (clip_id, local_time, summary) in enumerate(rows, start=1):
        image_path = keyframe_dir / f"frame_{index}.jpg"
        Image.new("RGB", (320, 240), (20 * index, 80, 130)).save(image_path)
        frames.append(
            {
                "clip_id": clip_id,
                "local_time": local_time,
                "keyframe_id": f"k{index}",
                "keyframe_path": str(image_path),
                "quality_score": 0.95,
                "clip_summary": summary,
                "caption": summary,
            }
        )
    _write_json(run_dir / "video" / "products" / "video_story_media_manifest.json", {"selected_keyframes": frames})


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
