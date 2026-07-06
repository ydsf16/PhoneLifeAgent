from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from life_report.session_loader import load_sessions


class SessionLoaderTest(unittest.TestCase):
    def test_load_minimal_session(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "session_20260624_110456"
            for child in ["video", "audio", "location", "motion", "environment"]:
                (root / child).mkdir(parents=True)
            (root / "capture_policy.json").write_text('{"camera":{"enabled":true}}', encoding="utf-8")
            (root / "video" / "clip_index.csv").write_text(
                "clip_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec\n"
                "1,video/clip_000001.mp4,1,11,101,111,10\n",
                encoding="utf-8",
            )
            (root / "video" / "clip_000001.mp4").write_bytes(b"fake")
            (root / "audio" / "audio_index.csv").write_text(
                "audio_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec\n"
                "1,audio/audio_000001.m4a,2,12,102,112,10\n",
                encoding="utf-8",
            )
            (root / "audio" / "audio_000001.m4a").write_bytes(b"fake")
            (root / "location" / "geo_location.csv").write_text(
                "sensor_sec,utc_sec,latitude,longitude\n3,103,40.0,116.0\n",
                encoding="utf-8",
            )
            (root / "motion" / "device_motion.csv").write_text(
                "sensor_sec,utc_sec,qx,qy,qz,qw\n4,104,0,0,0,1\n",
                encoding="utf-8",
            )
            (root / "environment" / "barometer.csv").write_text(
                "sensor_sec,utc_sec,pressure_kpa\n5,105,100\n",
                encoding="utf-8",
            )
            (root / "environment" / "magnetometer.csv").write_text(
                "sensor_sec,utc_sec,mx_uT,my_uT,mz_uT\n6,106,1,2,3\n",
                encoding="utf-8",
            )

            bundle = load_sessions([root])
            overview = bundle.to_dict()["overview"]

        self.assertEqual(overview["session_count"], 1)
        self.assertEqual(overview["clip_count"], 1)
        self.assertEqual(overview["audio_count"], 1)
        self.assertEqual(overview["gps_points"], 1)
        self.assertEqual(overview["motion_points"], 1)
        self.assertEqual(overview["gap_count"], 0)


if __name__ == "__main__":
    unittest.main()
