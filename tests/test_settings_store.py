import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from life_report.settings_store import (
    ApiSettings,
    apply_api_settings,
    load_api_settings,
    missing_for_comic,
    missing_for_provider,
    save_api_settings,
)


class SettingsStoreTest(unittest.TestCase):
    def test_env_local_fallback_round_trip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            settings = ApiSettings(
                dashscope_api_key="dash-key",
                dashscope_openai_base_url="https://example.test/compatible-mode/v1",
                amap_api_key="amap-key",
                ark_api_key="ark-key",
            )

            with (
                patch("life_report.settings_store._keychain_available", return_value=False),
                patch("life_report.settings_store._read_keychain", return_value=""),
                patch.dict(os.environ, {}, clear=True),
            ):
                destination = save_api_settings(settings, repo)
                loaded = load_api_settings(repo)

            self.assertEqual(destination, str(repo / ".env.local"))
            self.assertEqual(loaded.dashscope_api_key, "dash-key")
            self.assertEqual(loaded.dashscope_openai_base_url, "https://example.test/compatible-mode/v1")
            self.assertEqual(loaded.amap_api_key, "amap-key")
            self.assertEqual(loaded.ark_api_key, "ark-key")

    def test_apply_api_settings_sets_environment(self) -> None:
        settings = ApiSettings("dash", "https://base", "amap", "ark")
        with patch.dict(os.environ, {}, clear=True):
            apply_api_settings(settings)
            self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "dash")
            self.assertEqual(os.environ["DASHSCOPE_OPENAI_BASE_URL"], "https://base")
            self.assertEqual(os.environ["AMAP_API_KEY"], "amap")
            self.assertEqual(os.environ["ARK_API_KEY"], "ark")
            self.assertEqual(os.environ["SEEDREAM_API_KEY"], "ark")

    def test_missing_for_provider(self) -> None:
        settings = ApiSettings(dashscope_api_key="dash")

        self.assertEqual(
            missing_for_provider(settings, provider="aliyun", use_amap=True),
            ["Aliyun DashScope OpenAI-compatible Base URL", "Amap/Gaode API Key"],
        )
        self.assertEqual(missing_for_provider(settings, provider="mock", use_amap=False), [])

    def test_missing_for_comic(self) -> None:
        settings = ApiSettings(dashscope_api_key="dash")

        self.assertEqual(
            missing_for_comic(settings, provider="aliyun", image_provider="ark"),
            ["Seedream/Volcengine Ark API Key"],
        )
        self.assertEqual(missing_for_comic(settings, provider="mock", image_provider="mock"), [])


if __name__ == "__main__":
    unittest.main()
