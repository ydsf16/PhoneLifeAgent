from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from life_report.model_probe import probe_audio_model, read_audio_segments


class ModelProbeTest(unittest.TestCase):
    def test_read_audio_segments(self) -> None:
        with TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            (session / "audio").mkdir(parents=True)
            (session / "audio" / "audio_index.csv").write_text(
                "audio_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec\n"
                "1,audio/audio_000001.m4a,1,11,101,111,10\n",
                encoding="utf-8",
            )
            (session / "audio" / "audio_000001.m4a").write_bytes(b"fake")

            segments = read_audio_segments(session)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].audio_id, "1")
        self.assertEqual(segments[0].duration_sec, 10)

    def test_probe_audio_model_with_mock_provider(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            session = base / "session"
            output = base / "out"
            (session / "audio").mkdir(parents=True)
            (session / "audio" / "audio_index.csv").write_text(
                "audio_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec\n"
                "1,audio/audio_000001.m4a,1,11,101,111,10\n",
                encoding="utf-8",
            )
            (session / "audio" / "audio_000001.m4a").write_bytes(b"fake")

            result = probe_audio_model(session, output, provider="mock", limit_audio=1)
            rows = [
                json.loads(line)
                for line in result.audio_understandings_path.read_text(encoding="utf-8").splitlines()
            ]
            pretty_rows = json.loads(result.audio_understandings_pretty_path.read_text(encoding="utf-8"))

        self.assertEqual(result.processed_audio_count, 1)
        self.assertEqual(rows[0]["provider"], "mock")
        self.assertIn("environment_summary", rows[0])
        self.assertEqual(pretty_rows[0]["audio_id"], "1")
        self.assertIn("processing_duration_sec", rows[0])

    def test_probe_audio_model_injects_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            session = base / "session"
            output = base / "out"
            (session / "audio").mkdir(parents=True)
            (session / "audio" / "audio_index.csv").write_text(
                "audio_id,file_path,start_utc_sec,end_utc_sec,duration_sec\n"
                "1,audio/audio_000001.m4a,101,111,10\n",
                encoding="utf-8",
            )
            (session / "audio" / "audio_000001.m4a").write_bytes(b"fake")
            location_context = base / "clip_location_context.json"
            motion_context = base / "clip_motion_context.json"
            location_context.write_text(
                json.dumps({"audio_segments": [{"audio_id": "1", "model_context_text": "在公园附近", "location_quality": "good"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            motion_context.write_text(
                json.dumps({"audio_segments": [{"audio_id": "1", "model_context_text": "持续步行", "motion_state": "walking_like"}]}, ensure_ascii=False),
                encoding="utf-8",
            )

            result = probe_audio_model(
                session,
                output,
                provider="mock",
                limit_audio=1,
                location_context_path=location_context,
                motion_context_path=motion_context,
            )
            rows = json.loads(result.audio_understandings_pretty_path.read_text(encoding="utf-8"))

        self.assertIn("在公园附近", rows[0]["metadata"]["context_text"])
        self.assertEqual(rows[0]["metadata"]["context"]["motion"]["motion_state"], "walking_like")


if __name__ == "__main__":
    unittest.main()
