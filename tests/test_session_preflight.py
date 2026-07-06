from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from life_report.model_probe import read_audio_segments
from life_report.session_preflight import build_session_preflight
from life_report.video_products import read_video_clips


class SessionPreflightTest(unittest.TestCase):
    def test_preflight_quarantines_missing_and_invalid_media_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            session = base / "session"
            _write_session_with_bad_media(session)
            output = base / "preflight"

            result = build_session_preflight(session, output)

            self.assertTrue((output / "quarantine_manifest.json").exists())
            self.assertTrue((output / "session_health.json").exists())
            self.assertEqual(result["quarantined_media_rows"], 4)
            self.assertEqual(result["valid_audio_rows"], 0)
            self.assertEqual(result["valid_video_rows"], 0)

            audio_segments = read_audio_segments(session, quarantine_manifest_path=output / "quarantine_manifest.json")
            video_clips = read_video_clips(session, quarantine_manifest_path=output / "quarantine_manifest.json")
            self.assertEqual(audio_segments, [])
            self.assertEqual(video_clips, [])


def _write_session_with_bad_media(session: Path) -> None:
    for child in ["video", "audio", "location", "motion"]:
        (session / child).mkdir(parents=True)
    (session / "audio" / "tiny.m4a").write_bytes(b"x")
    (session / "video" / "tiny.mov").write_bytes(b"x")
    (session / "audio" / "audio_index.csv").write_text(
        "\n".join(
            [
                "audio_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec",
                "a_missing,audio/missing.m4a,0,10,100,110,10",
                "a_tiny,audio/tiny.m4a,0,10,100,90,-10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (session / "video" / "clip_index.csv").write_text(
        "\n".join(
            [
                "clip_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec,fps",
                "v_missing,video/missing.mov,0,10,100,110,10,30",
                "v_tiny,video/tiny.mov,0,10,100,90,-10,30",
            ]
        )
        + "\n",
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
