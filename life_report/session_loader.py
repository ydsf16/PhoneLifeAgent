from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


STREAM_FILES = {
    "video": Path("video/clip_index.csv"),
    "audio": Path("audio/audio_index.csv"),
    "location": Path("location/geo_location.csv"),
    "device_motion": Path("motion/device_motion.csv"),
    "barometer": Path("environment/barometer.csv"),
    "magnetometer": Path("environment/magnetometer.csv"),
}


@dataclass(frozen=True)
class StreamSummary:
    name: str
    path: str
    exists: bool
    rows: int
    start_sensor_sec: float | None = None
    end_sensor_sec: float | None = None
    start_utc_sec: float | None = None
    end_utc_sec: float | None = None
    duration_sec: float | None = None
    total_recorded_duration_sec: float | None = None
    max_gap_sec: float | None = None
    file_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionSummary:
    name: str
    path: str
    exists: bool
    total_size_bytes: int
    start_utc_sec: float | None
    end_utc_sec: float | None
    duration_sec: float | None
    streams: dict[str, StreamSummary]
    warnings: list[str] = field(default_factory=list)
    capture_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["streams"] = {name: stream.to_dict() for name, stream in self.streams.items()}
        return data


@dataclass(frozen=True)
class SessionBundle:
    sessions: list[SessionSummary]
    start_utc_sec: float | None
    end_utc_sec: float | None
    duration_sec: float | None
    recorded_duration_sec: float
    gap_count: int
    gaps: list[dict[str, Any]]
    total_size_bytes: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions": [session.to_dict() for session in self.sessions],
            "start_utc_sec": self.start_utc_sec,
            "end_utc_sec": self.end_utc_sec,
            "duration_sec": self.duration_sec,
            "recorded_duration_sec": self.recorded_duration_sec,
            "gap_count": self.gap_count,
            "gaps": self.gaps,
            "total_size_bytes": self.total_size_bytes,
            "warnings": self.warnings,
            "overview": bundle_overview(self),
        }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(line for line in file if not line.startswith("#"))
        return list(reader)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def span(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def max_gap(values: list[float]) -> float | None:
    values = sorted(values)
    if len(values) < 2:
        return None
    return max(b - a for a, b in zip(values[:-1], values[1:]))


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def indexed_media_size(session_path: Path, rows: list[dict[str, str]]) -> int:
    total = 0
    for row in rows:
        rel_path = row.get("file_path")
        if rel_path:
            media_path = session_path / rel_path
            if media_path.exists():
                total += media_path.stat().st_size
    return total


def summarize_media_stream(name: str, session_path: Path, rel_path: Path) -> StreamSummary:
    index_path = session_path / rel_path
    rows = read_csv_rows(index_path)
    starts = [value for value in (number(row.get("start_sensor_sec")) for row in rows) if value is not None]
    ends = [value for value in (number(row.get("end_sensor_sec")) for row in rows) if value is not None]
    utc_starts = [value for value in (number(row.get("start_utc_sec")) for row in rows) if value is not None]
    utc_ends = [value for value in (number(row.get("end_utc_sec")) for row in rows) if value is not None]
    durations = [value for value in (number(row.get("duration_sec")) for row in rows) if value is not None]
    return StreamSummary(
        name=name,
        path=str(rel_path),
        exists=index_path.exists(),
        rows=len(rows),
        start_sensor_sec=min(starts) if starts else None,
        end_sensor_sec=max(ends) if ends else None,
        start_utc_sec=min(utc_starts) if utc_starts else None,
        end_utc_sec=max(utc_ends) if utc_ends else None,
        duration_sec=span(min(starts), max(ends)) if starts and ends else None,
        total_recorded_duration_sec=sum(durations),
        max_gap_sec=max_gap(starts),
        file_size_bytes=indexed_media_size(session_path, rows),
    )


def summarize_sensor_stream(name: str, session_path: Path, rel_path: Path) -> StreamSummary:
    csv_path = session_path / rel_path
    rows = read_csv_rows(csv_path)
    sensor_times = [value for value in (number(row.get("sensor_sec")) for row in rows) if value is not None]
    utc_times = [value for value in (number(row.get("utc_sec")) for row in rows) if value is not None]
    return StreamSummary(
        name=name,
        path=str(rel_path),
        exists=csv_path.exists(),
        rows=len(rows),
        start_sensor_sec=min(sensor_times) if sensor_times else None,
        end_sensor_sec=max(sensor_times) if sensor_times else None,
        start_utc_sec=min(utc_times) if utc_times else None,
        end_utc_sec=max(utc_times) if utc_times else None,
        duration_sec=span(min(sensor_times), max(sensor_times)) if sensor_times else None,
        max_gap_sec=max_gap(sensor_times),
        file_size_bytes=csv_path.stat().st_size if csv_path.exists() else 0,
    )


def summarize_session(path: Path) -> SessionSummary:
    session_path = path.expanduser().resolve()
    warnings = []
    if not session_path.exists():
        return SessionSummary(
            name=session_path.name,
            path=str(session_path),
            exists=False,
            total_size_bytes=0,
            start_utc_sec=None,
            end_utc_sec=None,
            duration_sec=None,
            streams={},
            warnings=[f"Session path does not exist: {session_path}"],
        )
    if not session_path.is_dir():
        return SessionSummary(
            name=session_path.name,
            path=str(session_path),
            exists=False,
            total_size_bytes=0,
            start_utc_sec=None,
            end_utc_sec=None,
            duration_sec=None,
            streams={},
            warnings=[f"Session path is not a directory: {session_path}"],
        )

    capture_policy_path = session_path / "capture_policy.json"
    capture_policy = read_json(capture_policy_path)
    if not capture_policy_path.exists():
        warnings.append("Missing capture_policy.json")

    streams = {
        "video": summarize_media_stream("video", session_path, STREAM_FILES["video"]),
        "audio": summarize_media_stream("audio", session_path, STREAM_FILES["audio"]),
        "location": summarize_sensor_stream("location", session_path, STREAM_FILES["location"]),
        "device_motion": summarize_sensor_stream("device_motion", session_path, STREAM_FILES["device_motion"]),
        "barometer": summarize_sensor_stream("barometer", session_path, STREAM_FILES["barometer"]),
        "magnetometer": summarize_sensor_stream("magnetometer", session_path, STREAM_FILES["magnetometer"]),
    }
    for stream in streams.values():
        if not stream.exists:
            warnings.append(f"Missing {stream.path}")

    non_empty = [stream for stream in streams.values() if stream.rows > 0]
    starts = [stream.start_utc_sec for stream in non_empty if stream.start_utc_sec is not None]
    ends = [stream.end_utc_sec for stream in non_empty if stream.end_utc_sec is not None]

    return SessionSummary(
        name=session_path.name,
        path=str(session_path),
        exists=True,
        total_size_bytes=directory_size(session_path),
        start_utc_sec=min(starts) if starts else None,
        end_utc_sec=max(ends) if ends else None,
        duration_sec=span(min(starts), max(ends)) if starts and ends else None,
        streams=streams,
        warnings=warnings,
        capture_policy=capture_policy,
    )


def load_sessions(paths: list[Path]) -> SessionBundle:
    sessions = [summarize_session(path) for path in paths]
    valid_sessions = [session for session in sessions if session.exists]
    starts = [session.start_utc_sec for session in valid_sessions if session.start_utc_sec is not None]
    ends = [session.end_utc_sec for session in valid_sessions if session.end_utc_sec is not None]
    intervals = sorted(
        (session.start_utc_sec, session.end_utc_sec, session.name)
        for session in valid_sessions
        if session.start_utc_sec is not None and session.end_utc_sec is not None
    )
    gaps = []
    for previous, current in zip(intervals[:-1], intervals[1:]):
        previous_end = previous[1]
        current_start = current[0]
        if current_start > previous_end:
            gaps.append(
                {
                    "start_utc_sec": previous_end,
                    "end_utc_sec": current_start,
                    "duration_sec": current_start - previous_end,
                    "from_session": previous[2],
                    "to_session": current[2],
                }
            )
    warnings = [warning for session in sessions for warning in session.warnings]
    return SessionBundle(
        sessions=sessions,
        start_utc_sec=min(starts) if starts else None,
        end_utc_sec=max(ends) if ends else None,
        duration_sec=span(min(starts), max(ends)) if starts and ends else None,
        recorded_duration_sec=sum(session.duration_sec or 0 for session in valid_sessions),
        gap_count=len(gaps),
        gaps=gaps,
        total_size_bytes=sum(session.total_size_bytes for session in valid_sessions),
        warnings=warnings,
    )


def format_datetime(utc_sec: float | None) -> str:
    if utc_sec is None:
        return "-"
    return datetime.fromtimestamp(utc_sec).strftime("%Y-%m-%d %H:%M:%S")


def format_time_range(start_utc_sec: float | None, end_utc_sec: float | None) -> str:
    if start_utc_sec is None or end_utc_sec is None:
        return "-"
    return f"{datetime.fromtimestamp(start_utc_sec).strftime('%H:%M')} - {datetime.fromtimestamp(end_utc_sec).strftime('%H:%M')}"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def format_size(bytes_count: int) -> str:
    if bytes_count >= 1024**3:
        return f"{bytes_count / 1024**3:.2f} GB"
    if bytes_count >= 1024**2:
        return f"{bytes_count / 1024**2:.1f} MB"
    if bytes_count >= 1024:
        return f"{bytes_count / 1024:.1f} KB"
    return f"{bytes_count} B"


def stream_rows(session: SessionSummary, stream_name: str) -> int:
    stream = session.streams.get(stream_name)
    return stream.rows if stream else 0


def bundle_overview(bundle: SessionBundle) -> dict[str, Any]:
    return {
        "session_count": len([session for session in bundle.sessions if session.exists]),
        "time_range": format_time_range(bundle.start_utc_sec, bundle.end_utc_sec),
        "start_time": format_datetime(bundle.start_utc_sec),
        "end_time": format_datetime(bundle.end_utc_sec),
        "duration": format_duration(bundle.duration_sec),
        "recorded_duration": format_duration(bundle.recorded_duration_sec),
        "clip_count": sum(stream_rows(session, "video") for session in bundle.sessions),
        "audio_count": sum(stream_rows(session, "audio") for session in bundle.sessions),
        "gps_points": sum(stream_rows(session, "location") for session in bundle.sessions),
        "motion_points": sum(stream_rows(session, "device_motion") for session in bundle.sessions),
        "barometer_points": sum(stream_rows(session, "barometer") for session in bundle.sessions),
        "magnetometer_points": sum(stream_rows(session, "magnetometer") for session in bundle.sessions),
        "total_size": format_size(bundle.total_size_bytes),
        "gap_count": bundle.gap_count,
        "warnings": bundle.warnings,
    }
