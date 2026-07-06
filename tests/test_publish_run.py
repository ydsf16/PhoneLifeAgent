import tempfile
import unittest
from pathlib import Path

from PIL import Image

from life_report.publish_run import publish_run


class PublishRunTest(unittest.TestCase):
    def test_publish_run_writes_docs_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "outputs" / "run_1"
            _write_fixture(run_dir)
            target = root / "docs" / "demo" / "run_1"

            result = publish_run(run_dir, target)

            self.assertTrue((target / "index.html").exists())
            self.assertTrue((target.parent / "index.html").exists())
            self.assertTrue((target / "story" / "index.html").exists())
            self.assertTrue((target / "comic" / "index.html").exists())
            self.assertTrue((target / "highlight_video" / "index.html").exists())
            self.assertTrue((target / "story" / "overall_route_map.png").exists())
            self.assertTrue((target / "story" / "keyframes" / "frame_1.jpg").exists())
            self.assertTrue((target / "comic" / "daily_comic_card.png").exists())
            self.assertTrue((target / "highlight_video" / "highlight_video.mp4").exists())
            self.assertIn("run_1", Path(result["run_index_path"]).read_text(encoding="utf-8"))


def _write_fixture(run_dir: Path) -> None:
    (run_dir / "story").mkdir(parents=True)
    (run_dir / "comic" / "refs").mkdir(parents=True)
    (run_dir / "highlight_video").mkdir(parents=True)
    route_map = run_dir / "story" / "overall_route_map.png"
    keyframe = run_dir / "story" / "frame_1.jpg"
    Image.new("RGB", (640, 480), (100, 140, 180)).save(route_map)
    Image.new("RGB", (640, 480), (120, 150, 190)).save(keyframe)
    (run_dir / "story" / "life_story.md").write_text("# Test Story\n\n## 一天总览\nA short day.\n", encoding="utf-8")
    (run_dir / "story" / "story_evidence_pack.json").write_text(
        "{\n"
        '  "time_range": {"start_local_time": "2026-07-05 12:00:00"},\n'
        f'  "media": {{"overall_route_map": "{route_map}", "selected_keyframes": [{{"keyframe_path": "{keyframe}", "caption": "frame one"}}]}}\n'
        "}\n",
        encoding="utf-8",
    )
    (run_dir / "comic" / "daily_comic.html").write_text('<img src="daily_comic_card.png">', encoding="utf-8")
    (run_dir / "comic" / "daily_comic_card.png").write_bytes(keyframe.read_bytes())
    (run_dir / "comic" / "daily_comic.png").write_bytes(keyframe.read_bytes())
    (run_dir / "comic" / "daily_comic_panel.png").write_bytes(keyframe.read_bytes())
    (run_dir / "comic" / "comic_storyline.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "comic" / "comic_storyboard.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "comic" / "comic_reference_plan.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "comic" / "refs" / "ref_01.jpg").write_bytes(keyframe.read_bytes())
    (run_dir / "highlight_video" / "highlight_video.mp4").write_bytes(b"mp4")
    (run_dir / "highlight_video" / "highlight_plan.json").write_text(
        '{"title":"Test Highlight","summary":"A short summary.","segments":[{"caption":"One","local_time_range":"12:00","reason":"test"}]}\n',
        encoding="utf-8",
    )
