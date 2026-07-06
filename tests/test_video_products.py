import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from life_report.video_products import (
    _attach_video_context,
    _clip_orientation_filter,
    _interval_points,
    _source_orientation_filter,
    build_video_compact_raw,
    build_video_products,
    build_video_story_media_manifest,
    build_video_timeline,
    probe_video_model,
)
from life_report.video_story import _video_story_input_system_prompt


class VideoProductsTest(unittest.TestCase):
    def test_build_video_timeline_converts_keyframe_time(self) -> None:
        timeline = build_video_timeline(
            [
                {
                    "clip_id": "1",
                    "understanding": {
                        "clip_summary": "我在走廊里行走。",
                        "objects": ["钥匙", "门"],
                        "event_candidates": [
                            {
                                "event_type": "walking",
                                "time_range": "00:02 - 00:05",
                                "description": "我走过走廊。",
                            }
                        ],
                        "keyframe_candidates": [
                            {
                                "relative_time_sec": 3.0,
                                "caption": "走廊画面",
                                "keyframe_path": "/tmp/frame.jpg",
                            }
                        ],
                    },
                    "metadata": {
                        "original_video_path": "/tmp/clip.mp4",
                        "report_video_path": "/tmp/report.mp4",
                        "start_utc_sec": 1000.0,
                        "end_utc_sec": 1010.0,
                        "duration_sec": 10.0,
                    },
                }
            ]
        )

        self.assertEqual(timeline["events"][0]["absolute_start_utc_sec"], 1002.0)
        self.assertEqual(timeline["keyframes"][0]["absolute_utc_sec"], 1003.0)
        self.assertEqual(timeline["keyframes"][0]["keyframe_path"], "/tmp/frame.jpg")

    def test_interval_points_extract_every_two_seconds(self) -> None:
        self.assertEqual(_interval_points(5.2), [0.0, 2.0, 4.0])
        self.assertEqual(_interval_points(7.4), [0.0, 2.0, 4.0, 6.0])

    def test_source_orientation_filter_keeps_sideways_metadata_unmodified(self) -> None:
        with unittest.mock.patch(
            "life_report.video_products._probe_video_stream",
            return_value={"width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]},
        ):
            self.assertEqual(_source_orientation_filter(Path("/tmp/video.mp4")), "")

    def test_clip_orientation_filter_ignores_sideways_metadata_for_landscape_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            (session / "motion").mkdir()
            (session / "motion" / "device_motion.csv").write_text(
                "utc_sec,gravity_x,gravity_y,gravity_z\n"
                "100,-0.99,0.02,0.0\n"
                "101,-0.98,0.03,0.0\n",
                encoding="utf-8",
            )
            with unittest.mock.patch(
                "life_report.video_products._probe_video_stream",
                return_value={"width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]},
            ):
                self.assertEqual(
                    _clip_orientation_filter(session, Path("/tmp/video.mp4"), {"start_utc_sec": 100, "end_utc_sec": 101}),
                    "",
                )

    def test_clip_orientation_filter_flips_upside_down_landscape_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            (session / "motion").mkdir()
            (session / "motion" / "device_motion.csv").write_text(
                "utc_sec,gravity_x,gravity_y,gravity_z\n"
                "100,0.99,-0.02,0.0\n"
                "101,0.98,-0.03,0.0\n",
                encoding="utf-8",
            )
            with unittest.mock.patch(
                "life_report.video_products._probe_video_stream",
                return_value={"width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]},
            ):
                self.assertEqual(
                    _clip_orientation_filter(session, Path("/tmp/video.mp4"), {"start_utc_sec": 100, "end_utc_sec": 101}),
                    "hflip,vflip",
                )

    def test_build_video_timeline_prefers_program_keyframes(self) -> None:
        timeline = build_video_timeline(
            [
                {
                    "clip_id": "1",
                    "understanding": {
                        "clip_summary": "我在走廊里行走。",
                        "keyframe_candidates": [{"relative_time_sec": 9.0, "keyframe_path": "/tmp/model.jpg"}],
                    },
                    "metadata": {
                        "original_video_path": "/tmp/clip.mp4",
                        "report_video_path": "/tmp/report.mp4",
                        "start_utc_sec": 1000.0,
                        "end_utc_sec": 1010.0,
                        "duration_sec": 10.0,
                        "report_keyframes": [
                            {
                                "keyframe_id": "fixed-1",
                                "relative_time_sec": 2.0,
                                "keyframe_path": "/tmp/fixed.jpg",
                                "quality_score": 0.8,
                                "accepted": True,
                            }
                        ],
                    },
                }
            ]
        )

        self.assertEqual(len(timeline["keyframes"]), 1)
        self.assertEqual(timeline["keyframes"][0]["keyframe_path"], "/tmp/fixed.jpg")
        self.assertEqual(timeline["keyframes"][0]["quality_score"], 0.8)

    def test_build_video_compact_raw_includes_report_media(self) -> None:
        timeline = {
            "time_range": {"start_local_time": "t1", "end_local_time": "t2"},
            "clips": [
                {
                    "clip_id": "1",
                    "start_local_time": "t1",
                    "end_local_time": "t2",
                    "clip_summary": "我走过走廊。",
                    "objects": ["钥匙"],
                    "activities": ["行走"],
                    "report_video_path": "/tmp/report.mp4",
                }
            ],
            "events": [],
            "keyframes": [{"clip_id": "1", "local_time": "t1", "caption": "画面", "keyframe_path": "/tmp/frame.jpg"}],
            "todo_candidates": [],
            "memory_candidates": [],
        }

        raw = build_video_compact_raw(timeline)

        self.assertIn("/tmp/report.mp4", raw)
        self.assertIn("/tmp/frame.jpg", raw)

    def test_story_media_manifest_selects_limited_keyframes(self) -> None:
        timeline = {
            "time_range": {"start_local_time": "t1", "end_local_time": "t2"},
            "clips": [
                {
                    "clip_id": "1",
                    "start_local_time": "t1",
                    "end_local_time": "t1b",
                    "report_video_path": "/tmp/report1.mp4",
                    "clip_summary": "我出门。",
                    "life_story_hint": "我准备出门。",
                },
                {
                    "clip_id": "2",
                    "start_local_time": "t2",
                    "end_local_time": "t2b",
                    "report_video_path": "/tmp/report2.mp4",
                    "clip_summary": "我在路上。",
                },
            ],
            "keyframes": [
                {
                    "keyframe_id": "kf1",
                    "clip_id": "1",
                    "absolute_utc_sec": 10.0,
                    "local_time": "t1",
                    "keyframe_path": "/tmp/kf1.jpg",
                    "caption": "门口画面",
                    "importance": "high",
                },
                {
                    "keyframe_id": "kf2",
                    "clip_id": "1",
                    "absolute_utc_sec": 11.0,
                    "local_time": "t1",
                    "keyframe_path": "/tmp/kf2.jpg",
                    "caption": "重复画面",
                    "accepted": False,
                    "reject_reasons": ["blurry"],
                },
                {
                    "keyframe_id": "kf3",
                    "clip_id": "2",
                    "absolute_utc_sec": 20.0,
                    "local_time": "t2",
                    "keyframe_path": "/tmp/kf3.jpg",
                    "reason": "高光路口",
                },
            ],
        }

        manifest = build_video_story_media_manifest(timeline, max_keyframes=2, max_keyframes_per_clip=1)

        self.assertEqual(manifest["all_keyframe_count"], 3)
        self.assertEqual(len(manifest["selected_keyframes"]), 2)
        self.assertEqual({frame["clip_id"] for frame in manifest["selected_keyframes"]}, {"1", "2"})
        self.assertEqual(len(manifest["selected_report_videos"]), 2)
        self.assertEqual(manifest["usable_keyframe_count"], 2)

    def test_story_media_manifest_distributes_keyframes_across_long_timeline(self) -> None:
        clips = []
        keyframes = []
        for index in range(1, 21):
            clip_id = str(index)
            clips.append(
                {
                    "clip_id": clip_id,
                    "start_local_time": f"t{index}",
                    "end_local_time": f"t{index}b",
                    "report_video_path": f"/tmp/report{index}.mp4",
                    "clip_summary": f"片段 {index}",
                }
            )
            keyframes.append(
                {
                    "keyframe_id": f"kf{index}",
                    "clip_id": clip_id,
                    "absolute_utc_sec": float(index),
                    "local_time": f"t{index}",
                    "keyframe_path": f"/tmp/kf{index}.jpg",
                    "quality_score": 0.9,
                    "accepted": True,
                }
            )
        manifest = build_video_story_media_manifest({"clips": clips, "keyframes": keyframes}, max_keyframes=5, max_keyframes_per_clip=1)

        selected_ids = [frame["clip_id"] for frame in manifest["selected_keyframes"]]
        self.assertEqual(len(selected_ids), 5)
        self.assertIn("1", selected_ids)
        self.assertIn("20", selected_ids)
        self.assertGreaterEqual(int(selected_ids[-2]), 14)
        self.assertEqual(len(manifest["selected_report_videos"]), 20)

    def test_build_video_products_writes_media_manifest(self) -> None:
        record = [
            {
                "clip_id": "1",
                "understanding": {
                    "clip_summary": "我在走廊里行走。",
                    "keyframe_candidates": [
                        {
                            "relative_time_sec": 3.0,
                            "caption": "走廊画面",
                            "keyframe_path": "/tmp/frame.jpg",
                        }
                    ],
                },
                "metadata": {
                    "original_video_path": "/tmp/clip.mp4",
                    "report_video_path": "/tmp/report.mp4",
                    "start_utc_sec": 1000.0,
                    "end_utc_sec": 1010.0,
                    "duration_sec": 10.0,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "video_understandings.json"
            input_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

            result = build_video_products(input_path, temp_path / "out", provider="none", max_story_keyframes=1)

            self.assertTrue(Path(result["video_story_media_manifest_path"]).exists())
            self.assertEqual(result["selected_story_keyframe_count"], 1)

    def test_build_video_products_injects_global_context(self) -> None:
        record = [
            {
                "clip_id": "1",
                "understanding": {"clip_summary": "我走过公园。"},
                "metadata": {
                    "original_video_path": "/tmp/clip.mp4",
                    "report_video_path": "/tmp/report.mp4",
                    "start_utc_sec": 1000.0,
                    "end_utc_sec": 1010.0,
                    "duration_sec": 10.0,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "video_understandings.json"
            location_compact = temp_path / "location_compact_raw.txt"
            motion_compact = temp_path / "motion_compact_raw.txt"
            input_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            location_compact.write_text("Overall route map: /tmp/route.png\n地点：树村公园", encoding="utf-8")
            motion_compact.write_text("运动：walking_like", encoding="utf-8")

            result = build_video_products(
                input_path,
                temp_path / "out",
                provider="none",
                location_compact_path=location_compact,
                motion_compact_path=motion_compact,
            )

            self.assertTrue(result["context_injected"])
            self.assertIn("树村公园", Path(result["video_story_input_path"]).read_text(encoding="utf-8"))

    def test_video_summary_prompt_requires_chinese_first_person(self) -> None:
        prompt = _video_story_input_system_prompt()

        self.assertIn("必须使用简体中文", prompt)
        self.assertIn("第一人称视角", prompt)
        self.assertIn("不要写成英文", prompt)
        self.assertIn("我看到", prompt)
        self.assertIn("不要写“我的女儿", prompt)

    def test_attach_video_context(self) -> None:
        clip = {"clip_id": "1"}
        contexts = {
            "location": {"video_clips": [{"clip_id": "1", "model_context_text": "在公园附近", "location_quality": "good"}]},
            "motion": {"video_clips": [{"clip_id": "1", "model_context_text": "手机晃动", "motion_state": "walking_like"}]},
        }

        enriched = _attach_video_context(clip, contexts)

        self.assertIn("在公园附近", enriched["context_text"])
        self.assertEqual(enriched["context_metadata"]["motion"]["motion_state"], "walking_like")

    def test_probe_video_model_isolates_per_clip_model_failure(self) -> None:
        class PartiallyFailingVideoModel:
            def understand_video(self, video, prompt):
                if video.clip_id == "2":
                    raise RuntimeError("model timeout")
                return {
                    "clip_id": video.clip_id,
                    "understanding": {"clip_summary": "ok"},
                    "metadata": {},
                }

        prepared = [
            {
                "clip_id": "1",
                "file_path": "/tmp/source1.mp4",
                "model_video_path": "/tmp/model1.mp4",
                "report_video_path": "/tmp/report1.mp4",
                "aligned_audio_path": "/tmp/audio1.m4a",
                "default_report_keyframes": [],
                "start_utc_sec": 100.0,
                "end_utc_sec": 110.0,
                "duration_sec": 10.0,
            },
            {
                "clip_id": "2",
                "file_path": "/tmp/source2.mp4",
                "model_video_path": "/tmp/model2.mp4",
                "report_video_path": "/tmp/report2.mp4",
                "aligned_audio_path": "/tmp/audio2.m4a",
                "default_report_keyframes": [],
                "start_utc_sec": 120.0,
                "end_utc_sec": 130.0,
                "duration_sec": 10.0,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            (session / "video").mkdir(parents=True)
            (session / "video" / "clip_index.csv").write_text(
                "clip_id,file_path,start_utc_sec,end_utc_sec,duration_sec,fps\n"
                "1,video/clip1.mp4,100,110,10,30\n"
                "2,video/clip2.mp4,120,130,10,30\n",
                encoding="utf-8",
            )
            with (
                unittest.mock.patch("life_report.video_products._prepare_clips", return_value=(prepared, [])),
                unittest.mock.patch("life_report.video_products.create_video_model", return_value=PartiallyFailingVideoModel()),
            ):
                result = probe_video_model(session, Path(temp_dir) / "out", provider="mock", understand_clips=2, concurrency=2)

            records = json.loads(Path(result.video_understandings_pretty_path).read_text(encoding="utf-8"))
            errors = json.loads(Path(result.video_model_errors_path).read_text(encoding="utf-8"))
            self.assertEqual(result.understood_clip_count, 1)
            self.assertEqual(result.model_error_count, 1)
            self.assertEqual(records[0]["clip_id"], "1")
            self.assertEqual(errors[0]["clip_id"], "2")
            self.assertEqual(errors[0]["stage"], "video_understanding")


if __name__ == "__main__":
    unittest.main()
