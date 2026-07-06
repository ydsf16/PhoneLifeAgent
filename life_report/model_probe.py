from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .context_injection import context_for_audio, context_metadata, format_model_context, load_clip_contexts
from .model_engine import AudioSegmentInput, create_speech_model
from .session_preflight import is_quarantined_audio, load_quarantine_manifest
from .session_loader import read_csv_rows


@dataclass(frozen=True)
class ProbeResult:
    output_dir: Path
    audio_understandings_path: Path
    audio_understandings_pretty_path: Path
    audio_probe_errors_path: Path
    processed_audio_count: int
    audio_error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["audio_understandings_path"] = str(self.audio_understandings_path)
        data["audio_understandings_pretty_path"] = str(self.audio_understandings_pretty_path)
        data["audio_probe_errors_path"] = str(self.audio_probe_errors_path)
        return data


def probe_audio_model(
    session_path: Path,
    output_dir: Path,
    provider: str = "mock",
    model: str | None = None,
    limit_audio: int = 1,
    audio_max_seconds: int | None = None,
    concurrency: int = 1,
    location_context_path: Path | None = None,
    motion_context_path: Path | None = None,
    quarantine_manifest_path: Path | None = None,
) -> ProbeResult:
    session = session_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    cache_dir = output / "model_response_cache"
    output.mkdir(parents=True, exist_ok=True)
    speech_model = create_speech_model(
        provider,
        model=model,
        cache_dir=cache_dir,
        max_audio_seconds=audio_max_seconds,
    )
    contexts = load_clip_contexts(location_context_path, motion_context_path)
    audio_segments = read_audio_segments(
        session,
        contexts=contexts,
        quarantine_manifest_path=quarantine_manifest_path,
    )
    if limit_audio > 0:
        audio_segments = audio_segments[:limit_audio]
    understandings_path = output / "audio_understandings.jsonl"
    pretty_path = output / "audio_understandings.json"
    errors_path = output / "audio_probe_errors.json"

    records = []
    errors = []
    if concurrency <= 1:
        for audio in audio_segments:
            try:
                records.append(_summarize_with_timing(speech_model, audio))
            except Exception as exc:
                errors.append(_audio_probe_error(audio, exc))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_summarize_with_timing, speech_model, audio): audio for audio in audio_segments}
            for future in as_completed(futures):
                audio = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    errors.append(_audio_probe_error(audio, exc))

    records.sort(key=lambda item: (_sort_time(item), item.get("audio_id", "")))
    errors.sort(key=lambda item: item.get("start_utc_sec") or 0)

    with understandings_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    pretty_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return ProbeResult(
        output_dir=output,
        audio_understandings_path=understandings_path,
        audio_understandings_pretty_path=pretty_path,
        audio_probe_errors_path=errors_path,
        processed_audio_count=len(audio_segments),
        audio_error_count=len(errors),
    )


def read_audio_segments(
    session_path: Path,
    contexts: dict[str, Any] | None = None,
    quarantine_manifest_path: Path | None = None,
) -> list[AudioSegmentInput]:
    rows = read_csv_rows(session_path / "audio" / "audio_index.csv")
    quarantine_manifest = load_quarantine_manifest(quarantine_manifest_path)
    segments = []
    for row in rows:
        if is_quarantined_audio(row, quarantine_manifest):
            continue
        rel_path = row.get("file_path")
        if not rel_path:
            continue
        audio_id = row.get("audio_id") or Path(rel_path).stem
        segment_context = context_for_audio(contexts or {}, str(audio_id))
        segments.append(
            AudioSegmentInput(
                audio_id=str(audio_id),
                file_path=session_path / rel_path,
                session_path=session_path,
                start_utc_sec=_float_or_none(row.get("start_utc_sec")),
                end_utc_sec=_float_or_none(row.get("end_utc_sec")),
                duration_sec=_float_or_none(row.get("duration_sec")),
                context_text=format_model_context(segment_context),
                context_metadata=context_metadata(segment_context),
            )
        )
    return segments


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_with_timing(speech_model: Any, audio: AudioSegmentInput) -> dict[str, Any]:
    started_at = time.time()
    record = speech_model.summarize_audio(audio).to_dict()
    finished_at = time.time()
    record["processing_started_at"] = started_at
    record["processing_finished_at"] = finished_at
    record["processing_duration_sec"] = round(finished_at - started_at, 3)
    return record


def _sort_time(record: dict[str, Any]) -> float:
    value = record.get("metadata", {}).get("start_utc_sec")
    return value if isinstance(value, (int, float)) else 0.0


def _audio_probe_error(audio: AudioSegmentInput, exc: Exception) -> dict[str, Any]:
    return {
        "audio_id": audio.audio_id,
        "file_path": str(audio.file_path),
        "start_utc_sec": audio.start_utc_sec,
        "end_utc_sec": audio.end_utc_sec,
        "duration_sec": audio.duration_sec,
        "reason": type(exc).__name__,
        "error": _one_line(str(exc), 1000),
    }


def _one_line(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
