import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from life_report.image_quality import analyze_image_quality


class ImageQualityTest(unittest.TestCase):
    def test_rejects_dark_solid_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dark.jpg"
            Image.new("RGB", (240, 240), (0, 0, 0)).save(path)

            quality = analyze_image_quality(path)

            self.assertFalse(quality["accepted"])
            self.assertIn("too_dark", quality["reject_reasons"])
            self.assertIn("low_contrast", quality["reject_reasons"])

    def test_accepts_sharp_varied_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sharp.jpg"
            image = Image.new("RGB", (240, 240), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            for x in range(0, 240, 16):
                color = (20, 20, 20) if (x // 16) % 2 == 0 else (30, 140, 220)
                draw.rectangle((x, 0, x + 8, 239), fill=color)
            for y in range(0, 240, 24):
                draw.line((0, y, 239, 239 - y), fill=(220, 60, 60), width=3)
            image.save(path)

            quality = analyze_image_quality(path)

            self.assertTrue(quality["accepted"], quality)
            self.assertGreater(quality["quality_score"], 0.3)


if __name__ == "__main__":
    unittest.main()
