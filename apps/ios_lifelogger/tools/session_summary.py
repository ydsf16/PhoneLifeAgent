#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


def read_rows(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            rows.append(row)
    return rows


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def duration(start, end):
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def average(values):
    return sum(values) / len(values) if values else None


def max_gap(values):
    values = sorted(values)
    if len(values) < 2:
        return None
    return max(b - a for a, b in zip(values[:-1], values[1:]))


def file_size(path: Path):
    return path.stat().st_size if path.exists() else 0


def indexed_media_size(session: Path, rows):
    total = 0
    for row in rows:
        rel = row.get("file_path")
        if rel:
            total += file_size(session / rel)
    return total


def probe_duration(path: Path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path.exists():
        return None
    try:
        output = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
            timeout=10,
        ).strip()
        return float(output)
    except (subprocess.SubprocessError, ValueError):
        return None


def media_duration_check(session: Path, rows):
    errors = []
    probed = []
    for row in rows:
        rel = row.get("file_path")
        expected = number(row.get("duration_sec"))
        actual = probe_duration(session / rel) if rel else None
        if actual is None:
            continue
        probed.append(actual)
        if expected is not None:
            errors.append(abs(expected - actual))
    return {
        "probed_file_count": len(probed),
        "total_file_duration_sec": sum(probed),
        "max_duration_error_sec": max(errors) if errors else None,
        "duration_mismatch_count": sum(1 for e in errors if e > 1.0),
    }


def media_summary(session: Path, index_path: Path):
    rows = read_rows(index_path)
    starts = [x for x in (number(r.get("start_sensor_sec")) for r in rows) if x is not None]
    ends = [x for x in (number(r.get("end_sensor_sec")) for r in rows) if x is not None]
    utc_starts = [x for x in (number(r.get("start_utc_sec")) for r in rows) if x is not None]
    utc_ends = [x for x in (number(r.get("end_utc_sec")) for r in rows) if x is not None]
    durations = [x for x in (number(r.get("duration_sec")) for r in rows) if x is not None]
    summary = {
        "rows": len(rows),
        "start_sensor_sec": min(starts) if starts else None,
        "end_sensor_sec": max(ends) if ends else None,
        "duration_sensor_sec": duration(min(starts), max(ends)) if starts and ends else None,
        "start_utc_sec": min(utc_starts) if utc_starts else None,
        "end_utc_sec": max(utc_ends) if utc_ends else None,
        "duration_utc_sec": duration(min(utc_starts), max(utc_ends)) if utc_starts and utc_ends else None,
        "total_recorded_duration_sec": sum(durations),
        "average_recorded_duration_sec": average(durations),
        "max_start_gap_sec": max_gap(starts),
        "file_size_bytes": indexed_media_size(session, rows),
    }
    summary.update(media_duration_check(session, rows))
    return summary


def sensor_summary(path: Path):
    rows = read_rows(path)
    sensor = [x for x in (number(r.get("sensor_sec")) for r in rows) if x is not None]
    utc = [x for x in (number(r.get("utc_sec")) for r in rows) if x is not None]
    sensor_span = duration(min(sensor), max(sensor)) if sensor else None
    return {
        "rows": len(rows),
        "start_sensor_sec": min(sensor) if sensor else None,
        "end_sensor_sec": max(sensor) if sensor else None,
        "duration_sensor_sec": sensor_span,
        "start_utc_sec": min(utc) if utc else None,
        "end_utc_sec": max(utc) if utc else None,
        "duration_utc_sec": duration(min(utc), max(utc)) if utc else None,
        "average_hz": ((len(rows) - 1) / sensor_span) if sensor_span and sensor_span > 0 else None,
        "max_sample_gap_sec": max_gap(sensor),
        "file_size_bytes": file_size(path),
    }


def directory_size(path: Path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def build_summary(session: Path):
    streams = {
        "video": media_summary(session, session / "video" / "clip_index.csv"),
        "audio": media_summary(session, session / "audio" / "audio_index.csv"),
        "location": sensor_summary(session / "location" / "geo_location.csv"),
        "device_motion": sensor_summary(session / "motion" / "device_motion.csv"),
        "barometer": sensor_summary(session / "environment" / "barometer.csv"),
        "magnetometer": sensor_summary(session / "environment" / "magnetometer.csv"),
    }
    non_empty = [s for s in streams.values() if s["rows"] > 0]
    sensor_starts = [s["start_sensor_sec"] for s in non_empty if s["start_sensor_sec"] is not None]
    sensor_ends = [s["end_sensor_sec"] for s in non_empty if s["end_sensor_sec"] is not None]
    utc_starts = [s["start_utc_sec"] for s in non_empty if s["start_utc_sec"] is not None]
    utc_ends = [s["end_utc_sec"] for s in non_empty if s["end_utc_sec"] is not None]
    return {
        "generated_utc_sec": time.time(),
        "session_name": session.name,
        "session_path": str(session),
        "total_size_bytes": directory_size(session),
        "start_sensor_sec": min(sensor_starts) if sensor_starts else None,
        "end_sensor_sec": max(sensor_ends) if sensor_ends else None,
        "duration_sensor_sec": duration(min(sensor_starts), max(sensor_ends)) if sensor_starts and sensor_ends else None,
        "start_utc_sec": min(utc_starts) if utc_starts else None,
        "end_utc_sec": max(utc_ends) if utc_ends else None,
        "duration_utc_sec": duration(min(utc_starts), max(utc_ends)) if utc_starts and utc_ends else None,
        "streams": streams,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    summary = build_summary(args.session)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if not args.no_write:
        (args.session / "session_summary.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
