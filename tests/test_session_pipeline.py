from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from life_report.pipeline_defaults import DEFAULT_MAX_STORY_KEYFRAMES
from life_report.session_pipeline import (
    SessionPipelineConfig,
    _cached_location_valid,
    _cached_story_valid,
    run_session_pipeline,
)


class SessionPipelineTest(unittest.TestCase):
    def test_session_pipeline_config_uses_shared_keyframe_default(self) -> None:
        config = SessionPipelineConfig(session_path=Path("/tmp/session"))
        self.assertEqual(config.max_story_keyframes, DEFAULT_MAX_STORY_KEYFRAMES)

    def test_run_session_pipeline_mock_writes_story(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            session = base / "session_20260623_141001"
            _write_empty_session(session)
            output = base / "out"
            progress = []

            result = run_session_pipeline(
                SessionPipelineConfig(session_path=session, output_dir=output, provider="mock"),
                progress=progress.append,
            )

            self.assertEqual(result["output_dir"], str(output.resolve()))
            self.assertTrue((output / "location" / "location_timeline.json").exists())
            self.assertTrue((output / "motion" / "motion_timeline.json").exists())
            self.assertTrue((output / "audio" / "probe" / "audio_understandings.json").exists())
            self.assertTrue((output / "video" / "probe" / "video_understandings.json").exists())
            self.assertTrue((output / "story" / "life_story.html").exists())
            self.assertTrue(any(item.startswith("TASK|Final Story|done|") for item in progress))

    def test_run_session_pipeline_reuses_cached_major_steps(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            session = base / "session_20260623_141001"
            _write_empty_session(session)
            output = base / "out"

            run_session_pipeline(SessionPipelineConfig(session_path=session, output_dir=output, provider="mock"))
            progress = []
            result = run_session_pipeline(
                SessionPipelineConfig(session_path=session, output_dir=output, provider="mock"),
                progress=progress.append,
            )

            self.assertEqual(result["location"]["status"], "skipped")
            self.assertEqual(result["motion"]["status"], "skipped")
            self.assertEqual(result["audio_probe"]["status"], "skipped")
            self.assertEqual(result["video_probe"]["status"], "skipped")
            self.assertIn("TASK|Location|skipped|cached", progress)

    def test_run_session_pipeline_force_rebuild_ignores_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            session = base / "session_20260623_141001"
            _write_empty_session(session)
            output = base / "out"

            run_session_pipeline(SessionPipelineConfig(session_path=session, output_dir=output, provider="mock"))
            progress = []
            result = run_session_pipeline(
                SessionPipelineConfig(session_path=session, output_dir=output, provider="mock", force_rebuild=True),
                progress=progress.append,
            )

            self.assertNotEqual(result["location"]["status"], "skipped")
            self.assertTrue(any(item.startswith("TASK|Location|running|") for item in progress))

    def test_amap_location_cache_requires_overall_route_map(self) -> None:
        with TemporaryDirectory() as temp_dir:
            location = Path(temp_dir) / "location"
            location.mkdir()
            (location / "location_timeline.json").write_text('{"segments":[]}', encoding="utf-8")

            self.assertFalse(_cached_location_valid(location, use_amap=True))
            self.assertTrue(_cached_location_valid(location, use_amap=False))

    def test_story_cache_invalid_when_route_map_missing_from_evidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            location = base / "location"
            story = base / "story"
            maps = location / "maps"
            maps.mkdir(parents=True)
            story.mkdir()
            route_map = maps / "overall_route_map.png"
            route_map.write_bytes(b"png")
            (location / "location_timeline.json").write_text(
                '{"overall_map_image":"' + str(route_map) + '"}',
                encoding="utf-8",
            )
            (story / "story_evidence_pack.json").write_text('{"media":{}}', encoding="utf-8")

            self.assertFalse(_cached_story_valid(story, location))


def _write_empty_session(session: Path) -> None:
    for child in ["video", "audio", "location", "motion", "environment"]:
        (session / child).mkdir(parents=True)
    (session / "capture_policy.json").write_text("{}", encoding="utf-8")
    (session / "video" / "clip_index.csv").write_text(
        "clip_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec,fps\n",
        encoding="utf-8",
    )
    (session / "audio" / "audio_index.csv").write_text(
        "audio_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec\n",
        encoding="utf-8",
    )
    (session / "location" / "geo_location.csv").write_text(
        "sensor_sec,utc_sec,latitude,longitude,horizontal_accuracy,speed,course,altitude,vertical_accuracy\n",
        encoding="utf-8",
    )
    (session / "motion" / "device_motion.csv").write_text(
        "sensor_sec,utc_sec,user_acc_x,user_acc_y,user_acc_z,rot_x,rot_y,rot_z,roll,pitch,yaw\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
