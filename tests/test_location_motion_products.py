import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from life_report.location_products import AmapClient, build_location_products
from life_report.motion_products import build_motion_products, classify_motion


class LocationMotionProductsTest(unittest.TestCase):
    def test_build_location_products_writes_clip_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            for child in ["video", "audio", "location"]:
                (session / child).mkdir(parents=True)
            (session / "location" / "geo_location.csv").write_text(
                "sensor_sec,utc_sec,latitude,longitude,altitude,horizontal_accuracy,vertical_accuracy,speed,course\n"
                "1,100,40.0,116.0,50,5,3,1.0,90\n"
                "2,110,40.0001,116.0001,51,6,3,1.1,90\n",
                encoding="utf-8",
            )
            (session / "video" / "clip_index.csv").write_text(
                "clip_id,file_path,start_utc_sec,end_utc_sec\n"
                "1,video/clip.mp4,100,110\n",
                encoding="utf-8",
            )
            (session / "audio" / "audio_index.csv").write_text(
                "audio_id,file_path,start_utc_sec,end_utc_sec\n"
                "1,audio/audio.m4a,100,110\n",
                encoding="utf-8",
            )

            result = build_location_products(session, Path(temp_dir) / "out")

            self.assertEqual(result["point_count"], 2)
            self.assertEqual(result["video_context_count"], 1)
            self.assertTrue(Path(result["clip_location_context_path"]).exists())

    def test_build_motion_products_writes_clip_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            for child in ["video", "audio", "motion"]:
                (session / child).mkdir(parents=True)
            (session / "motion" / "device_motion.csv").write_text(
                "sensor_sec,utc_sec,user_acc_x,user_acc_y,user_acc_z,rot_x,rot_y,rot_z\n"
                "1,100,0.01,0.01,0.01,0.01,0.01,0.01\n"
                "2,101,0.02,0.01,0.01,0.01,0.01,0.02\n",
                encoding="utf-8",
            )
            (session / "video" / "clip_index.csv").write_text(
                "clip_id,file_path,start_utc_sec,end_utc_sec\n"
                "1,video/clip.mp4,100,101\n",
                encoding="utf-8",
            )
            (session / "audio" / "audio_index.csv").write_text(
                "audio_id,file_path,start_utc_sec,end_utc_sec\n"
                "1,audio/audio.m4a,100,101\n",
                encoding="utf-8",
            )

            result = build_motion_products(session, Path(temp_dir) / "out", window_sec=1)

            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["video_context_count"], 1)
            self.assertTrue(Path(result["clip_motion_context_path"]).exists())

    def test_location_products_exclude_unreasonable_gps_jump_from_route_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            for child in ["video", "audio", "location"]:
                (session / child).mkdir(parents=True)
            (session / "location" / "geo_location.csv").write_text(
                "sensor_sec,utc_sec,latitude,longitude,altitude,horizontal_accuracy,vertical_accuracy,speed,course\n"
                "1,100,40.0,116.0,50,5,3,1.0,90\n"
                "2,101,40.1,116.1,50,5,3,1.0,90\n"
                "3,110,40.0001,116.0001,51,6,3,1.1,90\n",
                encoding="utf-8",
            )
            (session / "video" / "clip_index.csv").write_text("clip_id,file_path,start_utc_sec,end_utc_sec\n", encoding="utf-8")
            (session / "audio" / "audio_index.csv").write_text("audio_id,file_path,start_utc_sec,end_utc_sec\n", encoding="utf-8")

            result = build_location_products(session, Path(temp_dir) / "out")
            timeline = json.loads(Path(result["location_timeline_path"]).read_text(encoding="utf-8"))
            points = json.loads(Path(result["location_points_path"]).read_text(encoding="utf-8"))["points"]

            self.assertEqual(timeline["summary"]["bad_points"], 1)
            self.assertLess(timeline["summary"]["distance_m"], 50)
            self.assertIn("derived_speed_outlier", points[1]["flags"])

    def test_location_products_builds_only_overall_static_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            for child in ["video", "audio", "location"]:
                (session / child).mkdir(parents=True)
            rows = [
                "sensor_sec,utc_sec,latitude,longitude,altitude,horizontal_accuracy,vertical_accuracy,speed,course",
                *[f"{index},{100 + index},40.{index:04d},116.{index:04d},50,5,3,1.0,90" for index in range(1, 5)],
            ]
            (session / "location" / "geo_location.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            (session / "video" / "clip_index.csv").write_text("clip_id,file_path,start_utc_sec,end_utc_sec\n", encoding="utf-8")
            (session / "audio" / "audio_index.csv").write_text("audio_id,file_path,start_utc_sec,end_utc_sec\n", encoding="utf-8")
            calls = []

            class FakeAmap:
                enabled = True

                def convert_wgs84_points(self, points):
                    return points

                def regeo(self, _lng, _lat):
                    return {}

                def nearby_pois(self, _lng, _lat):
                    return {}

                def static_map(self, _lng, _lat, output_path, path_points=None, zoom=16):
                    calls.append((Path(output_path).name, zoom))
                    _write_png(Path(output_path))
                    return True

            with patch("life_report.location_products.AmapClient", return_value=FakeAmap()):
                result = build_location_products(session, Path(temp_dir) / "out", use_amap=True)

            self.assertEqual(calls[0][0], "overall_route_map.png")
            self.assertTrue(Path(result["overall_map_image"]).exists())
            timeline = json.loads(Path(result["location_timeline_path"]).read_text(encoding="utf-8"))
            self.assertTrue(all(segment.get("map_image") is None for segment in timeline["segments"]))

    def test_location_products_adapts_overall_map_zoom_to_route_extent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            for child in ["video", "audio", "location"]:
                (session / child).mkdir(parents=True)
            rows = [
                "sensor_sec,utc_sec,latitude,longitude,altitude,horizontal_accuracy,vertical_accuracy,speed,course",
                "1,100,40.0000,116.0000,50,5,3,1.0,90",
                "2,2500,40.0300,116.0600,50,5,3,1.0,90",
            ]
            (session / "location" / "geo_location.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            (session / "video" / "clip_index.csv").write_text("clip_id,file_path,start_utc_sec,end_utc_sec\n", encoding="utf-8")
            (session / "audio" / "audio_index.csv").write_text("audio_id,file_path,start_utc_sec,end_utc_sec\n", encoding="utf-8")
            calls = []

            class FakeAmap:
                enabled = True

                def convert_wgs84_points(self, points):
                    return points

                def regeo(self, _lng, _lat):
                    return {}

                def nearby_pois(self, _lng, _lat):
                    return {}

                def static_map(self, lng, lat, output_path, path_points=None, zoom=16):
                    calls.append((lng, lat, zoom, len(path_points or [])))
                    _write_png(Path(output_path))
                    return True

            with patch("life_report.location_products.AmapClient", return_value=FakeAmap()):
                build_location_products(session, Path(temp_dir) / "out", use_amap=True)

            self.assertEqual(len(calls), 1)
            self.assertLess(calls[0][2], 14)

    def test_static_map_retries_after_invalid_cached_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "overall_route_map.png"
            output.write_text('{"status":"0","info":"CUQPS_HAS_EXCEEDED_THE_LIMIT"}', encoding="utf-8")

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def read(self):
                    png = Path(temp_dir) / "valid.png"
                    _write_png(png)
                    return png.read_bytes()

            with patch.dict("os.environ", {"AMAP_API_KEY": "key"}):
                client = AmapClient()
            with patch("urllib.request.urlopen", return_value=Response()):
                self.assertTrue(client.static_map(116.0, 40.0, output))

            self.assertGreater(output.stat().st_size, 100)

    def test_motion_context_is_stale_when_media_is_far_after_last_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            for child in ["video", "audio", "motion"]:
                (session / child).mkdir(parents=True)
            (session / "motion" / "device_motion.csv").write_text(
                "sensor_sec,utc_sec,user_acc_x,user_acc_y,user_acc_z,rot_x,rot_y,rot_z\n"
                "1,100,0.01,0.01,0.01,0.01,0.01,0.01\n",
                encoding="utf-8",
            )
            (session / "video" / "clip_index.csv").write_text(
                "clip_id,file_path,start_utc_sec,end_utc_sec\n"
                "1,video/clip.mp4,1000,1010\n",
                encoding="utf-8",
            )
            (session / "audio" / "audio_index.csv").write_text("audio_id,file_path,start_utc_sec,end_utc_sec\n", encoding="utf-8")

            result = build_motion_products(session, Path(temp_dir) / "out", window_sec=1)
            context = json.loads(Path(result["clip_motion_context_path"]).read_text(encoding="utf-8"))

            self.assertEqual(context["video_clips"][0]["context_status"], "stale")
            self.assertEqual(context["video_clips"][0]["motion_state"], "unknown")
            self.assertIn("没有可靠", context["video_clips"][0]["model_context_text"])

    def test_classify_motion_stationary(self) -> None:
        state, intensity, stability, confidence = classify_motion({"accel_rms": 0.02, "gyro_rms": 0.02, "jerk_rms": 0.01})

        self.assertEqual(state, "stationary")
        self.assertEqual(intensity, "low")
        self.assertEqual(stability, "stable")
        self.assertGreater(confidence, 0.8)

def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), "white").save(path, format="PNG")


if __name__ == "__main__":
    unittest.main()
