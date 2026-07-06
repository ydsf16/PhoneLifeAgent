from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


SERVICE_NAME = "PhoneLifeAgent"
ENV_LOCAL_PATH = Path(".env.local")
DEFAULT_DASHSCOPE_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class ApiSettings:
    dashscope_api_key: str = ""
    dashscope_openai_base_url: str = ""
    amap_api_key: str = ""
    ark_api_key: str = ""

    def masked_status(self) -> dict[str, bool]:
        return {
            "dashscope_api_key": bool(self.dashscope_api_key),
            "dashscope_openai_base_url": bool(self.dashscope_openai_base_url),
            "amap_api_key": bool(self.amap_api_key),
            "ark_api_key": bool(self.ark_api_key),
        }


KEYCHAIN_ACCOUNTS = {
    "dashscope_api_key": "DASHSCOPE_API_KEY",
    "dashscope_openai_base_url": "DASHSCOPE_OPENAI_BASE_URL",
    "amap_api_key": "AMAP_API_KEY",
    "ark_api_key": "ARK_API_KEY",
}


ENV_KEYS = {
    "dashscope_api_key": "DASHSCOPE_API_KEY",
    "dashscope_openai_base_url": "DASHSCOPE_OPENAI_BASE_URL",
    "amap_api_key": "AMAP_API_KEY",
    "ark_api_key": "ARK_API_KEY",
}


def load_api_settings(repo_root: Path | None = None) -> ApiSettings:
    repo_root = repo_root or Path.cwd()
    env_local = _read_env_local(repo_root / ENV_LOCAL_PATH)
    values = {}
    for field, env_key in ENV_KEYS.items():
        values[field] = os.environ.get(env_key) or _read_keychain(KEYCHAIN_ACCOUNTS[field]) or env_local.get(env_key, "")
    return ApiSettings(**values)


def save_api_settings(settings: ApiSettings, repo_root: Path | None = None) -> str:
    repo_root = repo_root or Path.cwd()
    if _keychain_available():
        for field, account in KEYCHAIN_ACCOUNTS.items():
            value = getattr(settings, field)
            if value:
                _write_keychain(account, value)
        return "macOS Keychain"
    _write_env_local(repo_root / ENV_LOCAL_PATH, settings)
    return str(repo_root / ENV_LOCAL_PATH)


def apply_api_settings(settings: ApiSettings) -> None:
    if settings.dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = settings.dashscope_api_key
    os.environ["DASHSCOPE_OPENAI_BASE_URL"] = settings.dashscope_openai_base_url or DEFAULT_DASHSCOPE_OPENAI_BASE_URL
    if settings.amap_api_key:
        os.environ["AMAP_API_KEY"] = settings.amap_api_key
    if settings.ark_api_key:
        os.environ["ARK_API_KEY"] = settings.ark_api_key
        os.environ["SEEDREAM_API_KEY"] = settings.ark_api_key


def missing_for_provider(settings: ApiSettings, provider: str, use_amap: bool) -> list[str]:
    missing = []
    if provider == "aliyun":
        if not settings.dashscope_api_key:
            missing.append("Aliyun DashScope API Key")
    if use_amap and not settings.amap_api_key:
        missing.append("Amap/Gaode API Key")
    return missing


def missing_for_comic(settings: ApiSettings, provider: str, image_provider: str) -> list[str]:
    missing = []
    if provider == "aliyun" and not settings.dashscope_api_key:
        missing.append("Aliyun DashScope API Key")
    if image_provider == "ark" and not settings.ark_api_key:
        missing.append("Seedream/Volcengine Ark API Key")
    return missing


def _keychain_available() -> bool:
    return _run_security(["list-keychains"]).returncode == 0


def _read_keychain(account: str) -> str:
    result = _run_security(["find-generic-password", "-s", SERVICE_NAME, "-a", account, "-w"])
    return result.stdout.strip() if result.returncode == 0 else ""


def _write_keychain(account: str, value: str) -> None:
    _run_security(["delete-generic-password", "-s", SERVICE_NAME, "-a", account])
    result = _run_security(["add-generic-password", "-s", SERVICE_NAME, "-a", account, "-w", value])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Failed to save {account} to Keychain")


def _run_security(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["security", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _read_env_local(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _write_env_local(path: Path, settings: ApiSettings) -> None:
    path.write_text(
        "\n".join(
            [
                "# PhoneLifeAgent local secrets. Do not commit.",
                f"DASHSCOPE_API_KEY={_quote_env(settings.dashscope_api_key)}",
                f"DASHSCOPE_OPENAI_BASE_URL={_quote_env(settings.dashscope_openai_base_url)}",
                f"AMAP_API_KEY={_quote_env(settings.amap_api_key)}",
                f"ARK_API_KEY={_quote_env(settings.ark_api_key)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _quote_env(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
