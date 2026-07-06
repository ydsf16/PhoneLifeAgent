from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .session_loader import read_csv_rows


MAX_REASONABLE_DERIVED_SPEED_MPS = 35.0


def build_location_products(session_path: Path, output_dir: Path, use_amap: bool = False) -> dict[str, Any]:
    session = session_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    maps_dir = output / "maps"
    points = read_location_points(session)
    amap = AmapClient() if use_amap else None
    if amap and amap.enabled:
        apply_amap_coordinate_conversion(points, amap)
    timeline = build_location_timeline(points)
    overall_map_path = build_overall_route_map(points, amap, maps_dir)
    if overall_map_path:
        timeline["overall_map_image"] = overall_map_path
    enrich_location_timeline(timeline, amap, maps_dir)
    clip_context = build_clip_location_context(session, points, timeline, amap, maps_dir)
    compact_raw = build_location_compact_raw(timeline, clip_context)
    route_geojson = build_route_geojson(points, timeline)

    points_path = output / "location_points_clean.json"
    timeline_path = output / "location_timeline.json"
    context_path = output / "clip_location_context.json"
    compact_path = output / "location_compact_raw.txt"
    geojson_path = output / "route.geojson"
    points_path.write_text(json.dumps({"schema_version": "location_points_clean.v1", "source": "location/geo_location.csv", "points": points}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context_path.write_text(json.dumps(clip_context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact_path.write_text(compact_raw, encoding="utf-8")
    geojson_path.write_text(json.dumps(route_geojson, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "location_points_path": str(points_path),
        "location_timeline_path": str(timeline_path),
        "clip_location_context_path": str(context_path),
        "location_compact_raw_path": str(compact_path),
        "route_geojson_path": str(geojson_path),
        "overall_map_image": overall_map_path,
        "point_count": len(points),
        "segment_count": len(timeline["segments"]),
        "video_context_count": len(clip_context["video_clips"]),
        "audio_context_count": len(clip_context["audio_segments"]),
        "amap_enabled": bool(amap and amap.enabled),
    }


def read_location_points(session_path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(session_path / "location" / "geo_location.csv")
    points = []
    previous_route_point: dict[str, Any] | None = None
    for index, row in enumerate(rows, start=1):
        utc = _float_or_none(row.get("utc_sec"))
        lat = _float_or_none(row.get("latitude"))
        lng = _float_or_none(row.get("longitude"))
        if utc is None or lat is None or lng is None:
            continue
        accuracy = _float_or_none(row.get("horizontal_accuracy"))
        speed = _float_or_none(row.get("speed"))
        course = _float_or_none(row.get("course"))
        gcj_lat, gcj_lng = wgs84_to_gcj02(lat, lng)
        flags = []
        if speed is None or speed < 0:
            flags.append("invalid_speed")
            speed = None
        if accuracy is None:
            flags.append("missing_accuracy")
        elif accuracy > 80:
            flags.append("bad_accuracy")
        elif accuracy > 20:
            flags.append("low_accuracy")
        derived_speed = None
        if previous_route_point:
            dt = max(0.0, utc - previous_route_point["utc_sec"])
            if dt > 0:
                derived_speed = haversine_m(lat, lng, previous_route_point["wgs84"]["lat"], previous_route_point["wgs84"]["lng"]) / dt
                if derived_speed > MAX_REASONABLE_DERIVED_SPEED_MPS:
                    flags.append("derived_speed_outlier")
        quality = _point_quality(accuracy)
        movement = _point_movement(speed if speed is not None else derived_speed)
        if "derived_speed_outlier" in flags:
            quality = "bad"
            movement = "unknown"
        point = {
            "point_id": f"loc_{index:06d}",
            "utc_sec": utc,
            "local_time": _format_local_time(utc),
            "wgs84": {"lat": lat, "lng": lng},
            "gcj02": {"lat": gcj_lat, "lng": gcj_lng},
            "altitude_m": _float_or_none(row.get("altitude")),
            "horizontal_accuracy_m": accuracy,
            "vertical_accuracy_m": _float_or_none(row.get("vertical_accuracy")),
            "speed_mps": speed,
            "derived_speed_mps": derived_speed,
            "course_deg": course if course is not None and course >= 0 else None,
            "quality": quality,
            "movement": movement,
            "flags": flags,
        }
        points.append(point)
        if point["quality"] != "bad":
            previous_route_point = point
    return points


def apply_amap_coordinate_conversion(points: list[dict[str, Any]], amap: "AmapClient") -> None:
    for chunk in _chunks(points, 40):
        converted = amap.convert_wgs84_points([(point["wgs84"]["lng"], point["wgs84"]["lat"]) for point in chunk])
        if len(converted) != len(chunk):
            continue
        for point, (lng, lat) in zip(chunk, converted):
            point["gcj02"] = {"lat": lat, "lng": lng}
            point["flags"] = [flag for flag in point.get("flags", []) if flag != "local_gcj02_fallback"]


def build_location_timeline(points: list[dict[str, Any]]) -> dict[str, Any]:
    segments = []
    if points:
        windows = _location_windows(points, window_sec=30.0)
        current_points, current_state = windows[0]
        for window_points, state in windows[1:]:
            gap = window_points[0]["utc_sec"] - current_points[-1]["utc_sec"]
            if gap > 90 or state != current_state:
                segments.append(_make_location_segment(len(segments) + 1, current_points, current_state))
                current_points, current_state = window_points, state
            else:
                current_points.extend(window_points)
        segments.append(_make_location_segment(len(segments) + 1, current_points, current_state))
    segments = _merge_short_location_segments(segments)
    start = points[0]["utc_sec"] if points else None
    end = points[-1]["utc_sec"] if points else None
    return {
        "schema_version": "location_timeline.v1",
        "time_range": {
            "start_utc_sec": start,
            "end_utc_sec": end,
            "start_local_time": _format_local_time(start),
            "end_local_time": _format_local_time(end),
        },
        "segments": segments,
        "summary": {
            "point_count": len(points),
            "good_points": sum(1 for point in points if point["quality"] == "good"),
            "rough_points": sum(1 for point in points if point["quality"] == "rough"),
            "bad_points": sum(1 for point in points if point["quality"] == "bad"),
            "distance_m": round(sum(segment.get("distance_m") or 0 for segment in segments), 1),
        },
    }


def enrich_location_timeline(timeline: dict[str, Any], amap: "AmapClient | None", maps_dir: Path) -> None:
    if not amap or not amap.enabled:
        return
    maps_dir.mkdir(parents=True, exist_ok=True)
    for segment in timeline.get("segments", []):
        rep = segment.get("representative_point", {})
        point = rep.get("gcj02") or {}
        lat, lng = point.get("lat"), point.get("lng")
        if lat is None or lng is None:
            continue
        regeo = amap.regeo(lng, lat)
        pois = amap.nearby_pois(lng, lat)
        segment["amap"] = _amap_facts(regeo, pois)
        segment["map_image"] = None


def build_overall_route_map(points: list[dict[str, Any]], amap: "AmapClient | None", maps_dir: Path) -> str | None:
    if not amap or not amap.enabled or not points:
        return None
    usable = [point for point in points if point.get("quality") != "bad"]
    if not usable:
        return None
    center_lng, center_lat, zoom = _overall_route_viewport(usable)
    map_path = maps_dir / "overall_route_map.png"
    for count in (80, 40, 20, 0):
        path_points = [(point["gcj02"]["lng"], point["gcj02"]["lat"]) for point in _downsample(usable, count)] if count else None
        if amap.static_map(center_lng, center_lat, map_path, path_points=path_points, zoom=zoom):
            return str(map_path)
    return None


def build_clip_location_context(session_path: Path, points: list[dict[str, Any]], timeline: dict[str, Any], amap: "AmapClient | None", maps_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "clip_location_context.v1",
        "video_clips": [
            _location_context_for_media(row, "clip_id", points, timeline, amap, maps_dir, "video_clip")
            for row in read_csv_rows(session_path / "video" / "clip_index.csv")
        ],
        "audio_segments": [
            _location_context_for_media(row, "audio_id", points, timeline, amap, maps_dir, "audio")
            for row in read_csv_rows(session_path / "audio" / "audio_index.csv")
        ],
    }


def build_location_compact_raw(timeline: dict[str, Any], context: dict[str, Any]) -> str:
    lines = [
        "PhoneLifeAgent Location Compact Raw",
        f"Time range: {timeline.get('time_range', {}).get('start_local_time')} -> {timeline.get('time_range', {}).get('end_local_time')}",
        f"Summary: {json.dumps(timeline.get('summary', {}), ensure_ascii=False)}",
        f"Overall route map: {timeline.get('overall_map_image')}",
        "",
    ]
    for segment in timeline.get("segments", []):
        lines.extend(
            [
                f"[{segment['segment_id']}] {segment['start_local_time']} -> {segment['end_local_time']}",
                f"Type: {segment['type']} | movement={segment['movement']} | quality={segment['quality']}",
                f"Distance: {segment['distance_m']}m | avg_speed={segment['avg_speed_mps']}m/s | points={segment['point_count']}",
                f"Center: {segment.get('center', {})}",
                f"Geo facts: {json.dumps(segment.get('amap', {}), ensure_ascii=False)}",
                f"Map: {segment.get('map_image')}",
                "",
            ]
        )
    lines.append("Video Clip Context:")
    for item in context.get("video_clips", []):
        lines.append(f"- clip {item.get('clip_id')} | {item.get('local_time_range')} | {item.get('model_context_text')}")
    lines.append("")
    lines.append("Audio Segment Context:")
    for item in context.get("audio_segments", []):
        lines.append(f"- audio {item.get('audio_id')} | {item.get('local_time_range')} | {item.get('model_context_text')}")
    return "\n".join(lines).strip() + "\n"


def build_route_geojson(points: list[dict[str, Any]], timeline: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "clean_route_wgs84"},
                "geometry": {"type": "LineString", "coordinates": [[p["wgs84"]["lng"], p["wgs84"]["lat"]] for p in points if p["quality"] != "bad"]},
            },
            *[
                {
                    "type": "Feature",
                    "properties": {"segment_id": segment["segment_id"], "type": segment["type"], "movement": segment["movement"]},
                    "geometry": {"type": "Point", "coordinates": [segment["center"]["wgs84"]["lng"], segment["center"]["wgs84"]["lat"]]},
                }
                for segment in timeline.get("segments", [])
                if segment.get("center")
            ],
        ],
    }


class AmapClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("AMAP_API_KEY") or os.environ.get("GAODE_API_KEY")
        self.enabled = bool(self.api_key)

    def regeo(self, lng: float, lat: float) -> dict[str, Any]:
        return self._get_json("reverse_geocode", "https://restapi.amap.com/v3/geocode/regeo", {"location": f"{lng:.6f},{lat:.6f}", "radius": "200", "extensions": "all"})

    def nearby_pois(self, lng: float, lat: float) -> dict[str, Any]:
        return self._get_json("nearby_search", "https://restapi.amap.com/v3/place/around", {"location": f"{lng:.6f},{lat:.6f}", "radius": "200", "extensions": "all", "offset": "10", "page": "1"})

    def convert_wgs84_points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not self.enabled or not points:
            return []
        locations = "|".join(f"{lng:.6f},{lat:.6f}" for lng, lat in points)
        data = self._get_json(
            "coordinate_convert",
            "https://restapi.amap.com/v3/assistant/coordinate/convert",
            {"locations": locations, "coordsys": "gps", "output": "JSON"},
        )
        if data.get("status") != "1" or not data.get("locations"):
            return []
        converted = []
        for item in str(data["locations"]).split(";"):
            try:
                lng_text, lat_text = item.split(",", 1)
                converted.append((float(lng_text), float(lat_text)))
            except ValueError:
                return []
        return converted

    def static_map(self, lng: float, lat: float, output_path: Path, path_points: list[tuple[float, float]] | None = None, zoom: int = 16) -> bool:
        if not self.enabled:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "location": f"{lng:.6f},{lat:.6f}",
            "zoom": str(zoom),
            "size": "600*400",
            "scale": "2",
            "markers": f"mid,,A:{lng:.6f},{lat:.6f}",
            "key": self.api_key,
        }
        if path_points and len(path_points) >= 2:
            coords = ";".join(f"{point_lng:.6f},{point_lat:.6f}" for point_lng, point_lat in path_points)
            params["paths"] = f"8,0x3366FF,0.9,,:{coords}"
        url = "https://restapi.amap.com/v3/staticmap?" + urllib.parse.urlencode(params)
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    body = response.read()
                output_path.write_bytes(body)
                if _normalize_static_map_image(output_path):
                    return True
                error_info = _amap_static_map_error_info(body)
                output_path.unlink(missing_ok=True)
                if _should_retry_amap_static_map(error_info, attempt):
                    time.sleep(1.0 + attempt)
                    continue
                return False
            except Exception:
                if output_path.exists() and not _normalize_static_map_image(output_path):
                    output_path.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                return False
        return False

    def _get_json(self, namespace: str, url: str, params: dict[str, str]) -> dict[str, Any]:
        if not self.enabled:
            return {}
        params = {**params, "key": self.api_key or ""}
        request_url = url + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(request_url, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {"status": "0", "info": str(exc), "namespace": namespace}


def _location_context_for_media(
    row: dict[str, str],
    id_key: str,
    points: list[dict[str, Any]],
    timeline: dict[str, Any],
    amap: AmapClient | None,
    maps_dir: Path,
    map_prefix: str,
) -> dict[str, Any]:
    start = _float_or_none(row.get("start_utc_sec"))
    end = _float_or_none(row.get("end_utc_sec"))
    media_id = str(row.get(id_key) or "")
    local_range = f"{_format_local_time(start)} - {_format_local_time(end)}"
    matched_points = [p for p in points if start is not None and end is not None and start <= p["utc_sec"] <= end]
    if not matched_points and start is not None and end is not None:
        mid = (start + end) / 2
        matched_points = sorted(points, key=lambda p: abs(p["utc_sec"] - mid))[:1]
    matched_segments = _overlap_segments(timeline, start, end)
    rep = _representative_point(matched_points)
    facts = _geo_facts_for_point(rep, amap)
    map_image = None
    movement = _majority([segment.get("movement") for segment in matched_segments]) or _majority([p.get("movement") for p in matched_points]) or "unknown"
    quality = _overall_quality([p.get("quality") for p in matched_points])
    route_context = _route_context(matched_points)
    return {
        id_key: media_id,
        "start_utc_sec": start,
        "end_utc_sec": end,
        "local_time_range": local_range,
        "location_quality": quality,
        "movement": movement,
        "matched_segments": [segment["segment_id"] for segment in matched_segments],
        "point_count": len(matched_points),
        "representative_point": _public_point(rep),
        "geo_facts": facts,
        "route_context": route_context,
        "map_image": map_image,
        "model_context_text": _location_model_context_text(local_range, quality, movement, facts, route_context),
    }


def _make_location_segment(index: int, points: list[dict[str, Any]], state: str) -> dict[str, Any]:
    start, end = points[0]["utc_sec"], points[-1]["utc_sec"]
    distance = _polyline_distance(points)
    duration = max(0.0, end - start)
    movement = _majority([p["movement"] for p in points]) or "unknown"
    quality = _overall_quality([p["quality"] for p in points])
    if state == "indoor_low_gps":
        segment_type = "indoor_low_gps"
        movement = "stationary" if movement == "unknown" else movement
    elif movement == "stationary":
        segment_type = "stay"
    else:
        segment_type = "moving"
    return {
        "segment_id": f"loc_seg_{index:04d}",
        "type": segment_type,
        "movement": movement,
        "start_utc_sec": start,
        "end_utc_sec": end,
        "start_local_time": _format_local_time(start),
        "end_local_time": _format_local_time(end),
        "duration_sec": round(duration, 3),
        "quality": quality,
        "distance_m": round(distance, 1),
        "avg_speed_mps": round(distance / duration, 3) if duration else 0.0,
        "point_count": len(points),
        "center": {"wgs84": _center(points, "wgs84"), "gcj02": _center(points, "gcj02")},
        "representative_point": _public_point(_representative_point(points)),
        "polyline": {
            "wgs84": [[p["wgs84"]["lat"], p["wgs84"]["lng"]] for p in _downsample(points, 80)],
            "gcj02": [[p["gcj02"]["lat"], p["gcj02"]["lng"]] for p in _downsample(points, 80)],
        },
        "amap": {},
        "map_image": None,
        "confidence_notes": _location_confidence_notes(quality, segment_type),
        "_points": points,
    }


def _merge_short_location_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment["duration_sec"] < 20 and previous["type"] == segment["type"] and previous["movement"] == segment["movement"]:
            points = previous.pop("_points", []) + segment.pop("_points", [])
            merged[-1] = _make_location_segment(len(merged), points, _segment_state_for_points(points))
        else:
            merged.append(segment)
    for index, segment in enumerate(merged, start=1):
        segment["segment_id"] = f"loc_seg_{index:04d}"
        segment.pop("_points", None)
    return merged


def _segment_state_for_point(point: dict[str, Any]) -> str:
    if point["quality"] != "good" and point.get("movement") in {"stationary", "unknown"}:
        return "indoor_low_gps"
    return point.get("movement") or "unknown"


def _segment_state_for_points(points: list[dict[str, Any]]) -> str:
    quality = _overall_quality([point.get("quality") for point in points])
    movement = _majority([point.get("movement") for point in points]) or "unknown"
    if quality != "good" and movement in {"stationary", "unknown"}:
        return "indoor_low_gps"
    return movement


def _location_windows(points: list[dict[str, Any]], window_sec: float) -> list[tuple[list[dict[str, Any]], str]]:
    if not points:
        return []
    windows: list[tuple[list[dict[str, Any]], str]] = []
    start = points[0]["utc_sec"]
    current: list[dict[str, Any]] = []
    current_bucket = 0
    for point in points:
        bucket = int((point["utc_sec"] - start) // window_sec)
        if current and bucket != current_bucket:
            windows.append((current, _segment_state_for_points(current)))
            current = []
        current_bucket = bucket
        current.append(point)
    if current:
        windows.append((current, _segment_state_for_points(current)))
    return windows


def _point_quality(accuracy: float | None) -> str:
    if accuracy is None:
        return "rough"
    if accuracy <= 20:
        return "good"
    if accuracy <= 80:
        return "rough"
    return "bad"


def _point_movement(speed: float | None) -> str:
    if speed is None:
        return "unknown"
    if speed < 0.35:
        return "stationary"
    if speed < 2.2:
        return "walking"
    if speed < 4.5:
        return "running"
    if speed < 10:
        return "bike_or_scooter"
    return "vehicle"


def _route_context(points: list[dict[str, Any]]) -> dict[str, Any]:
    usable = _usable_route_points(points)
    distance = _polyline_distance(usable)
    duration = max(0.0, usable[-1]["utc_sec"] - usable[0]["utc_sec"]) if len(usable) >= 2 else 0.0
    avg_speed = distance / duration if duration else 0.0
    shape = "single_point" if len(usable) <= 1 else "short_local_motion" if distance < 50 else "moving_route"
    return {"distance_m": round(distance, 1), "avg_speed_mps": round(avg_speed, 3), "shape": shape}


def _location_model_context_text(local_range: str, quality: str, movement: str, facts: dict[str, Any], route: dict[str, Any]) -> str:
    places = []
    if facts.get("address"):
        places.append(str(facts["address"]))
    if facts.get("roads"):
        places.append("道路: " + "、".join(facts["roads"][:3]))
    if facts.get("pois"):
        names = [poi.get("name") if isinstance(poi, dict) else str(poi) for poi in facts["pois"][:5]]
        places.append("附近POI: " + "、".join(names))
    place_text = "；".join(places) if places else "暂无高德地点语义"
    caution = "GPS精度较好，可辅助地点判断。" if quality == "good" else "GPS精度有限，请把地点作为弱证据。"
    return f"{local_range} 定位上下文：movement={movement}，quality={quality}，路线约 {route.get('distance_m')}m；{place_text}；{caution}"


def _geo_facts_for_point(point: dict[str, Any] | None, amap: AmapClient | None) -> dict[str, Any]:
    if not point or not amap or not amap.enabled:
        return {"address": None, "roads": [], "pois": [], "aoi": []}
    gcj = point["gcj02"]
    return _amap_facts(amap.regeo(gcj["lng"], gcj["lat"]), amap.nearby_pois(gcj["lng"], gcj["lat"]))


def _amap_facts(regeo: dict[str, Any], nearby: dict[str, Any]) -> dict[str, Any]:
    regeocode = regeo.get("regeocode") if isinstance(regeo, dict) else {}
    address = regeocode.get("formatted_address") if isinstance(regeocode, dict) else None
    roads = []
    pois = []
    aois = []
    if isinstance(regeocode, dict):
        roads = [item.get("name") for item in _as_list(regeocode.get("roads")) if item.get("name")]
        aois = [item.get("name") for item in _as_list(regeocode.get("aois")) if item.get("name")]
    for item in _as_list(nearby.get("pois") if isinstance(nearby, dict) else []):
        if item.get("name"):
            pois.append({"name": item.get("name"), "type": item.get("type"), "distance_m": _float_or_none(item.get("distance"))})
    return {"address": address, "roads": roads[:5], "pois": pois[:10], "aoi": aois[:5]}


def _overlap_segments(timeline: dict[str, Any], start: float | None, end: float | None) -> list[dict[str, Any]]:
    if start is None or end is None:
        return []
    return [segment for segment in timeline.get("segments", []) if segment["end_utc_sec"] >= start and segment["start_utc_sec"] <= end]


def _representative_point(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not points:
        return None
    return sorted(points, key=lambda p: (_quality_rank(p.get("quality")), p.get("horizontal_accuracy_m") or 9999))[0]


def _public_point(point: dict[str, Any] | None) -> dict[str, Any] | None:
    if not point:
        return None
    return {
        "wgs84": point.get("wgs84"),
        "gcj02": point.get("gcj02"),
        "accuracy_m": point.get("horizontal_accuracy_m"),
        "utc_sec": point.get("utc_sec"),
        "local_time": point.get("local_time") or _format_local_time(point.get("utc_sec")),
    }


def _normalize_static_map_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format not in {"PNG", "JPEG"}:
                return False
            if image.width < 16 or image.height < 16:
                return False
            rgb = image.convert("RGB")
            rgb.save(path, format="PNG")
        return True
    except Exception:
        return False


def _overall_route_viewport(points: list[dict[str, Any]]) -> tuple[float, float, int]:
    coords = [
        (float(point["gcj02"]["lng"]), float(point["gcj02"]["lat"]))
        for point in points
        if point.get("gcj02", {}).get("lng") is not None and point.get("gcj02", {}).get("lat") is not None
    ]
    if not coords:
        return 116.397, 39.909, 14
    lngs = [item[0] for item in coords]
    lats = [item[1] for item in coords]
    min_lng, max_lng = min(lngs), max(lngs)
    min_lat, max_lat = min(lats), max(lats)
    center_lng = (min_lng + max_lng) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    zoom = _fit_route_zoom(min_lng, max_lng, min_lat, max_lat)
    return center_lng, center_lat, zoom


def _fit_route_zoom(min_lng: float, max_lng: float, min_lat: float, max_lat: float) -> int:
    width_px = 1200 * 0.78
    height_px = 800 * 0.78
    lng_span = max(1e-6, max_lng - min_lng)
    merc_min = _mercator_lat(min_lat)
    merc_max = _mercator_lat(max_lat)
    lat_span = max(1e-6, abs(merc_max - merc_min))
    zoom_x = math.log2(width_px * 360.0 / (lng_span * 256.0))
    zoom_y = math.log2(height_px / (lat_span * 256.0))
    zoom = int(math.floor(min(zoom_x, zoom_y)))
    return max(3, min(17, zoom))


def _mercator_lat(lat: float) -> float:
    clamped = max(-85.0, min(85.0, lat))
    radians = math.radians(clamped)
    return math.log(math.tan(math.pi / 4.0 + radians / 2.0))


def _amap_static_map_error_info(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("info") or data.get("infocode") or "")


def _should_retry_amap_static_map(error_info: str, attempt: int) -> bool:
    if attempt >= 2:
        return False
    upper = error_info.upper()
    return "QPS" in upper or "LIMIT" in upper or "TIMEOUT" in upper


def _overall_quality(values: list[str | None]) -> str:
    counts = Counter(value for value in values if value)
    if not counts:
        return "unknown"
    if counts["good"] >= max(counts["rough"], counts["bad"]):
        return "good"
    if counts["rough"] >= counts["bad"]:
        return "rough"
    return "bad"


def _quality_rank(value: str | None) -> int:
    return {"good": 0, "rough": 1, "bad": 2}.get(value or "", 3)


def _majority(values: list[Any]) -> Any:
    clean = [value for value in values if value is not None]
    return Counter(clean).most_common(1)[0][0] if clean else None


def _center(points: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {"lat": sum(p[key]["lat"] for p in points) / len(points), "lng": sum(p[key]["lng"] for p in points) / len(points)}


def _polyline_distance(points: list[dict[str, Any]]) -> float:
    usable = _usable_route_points(points)
    return sum(haversine_m(a["wgs84"]["lat"], a["wgs84"]["lng"], b["wgs84"]["lat"], b["wgs84"]["lng"]) for a, b in zip(usable[:-1], usable[1:]))


def _usable_route_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [point for point in points if point.get("quality") != "bad" and "derived_speed_outlier" not in point.get("flags", [])]


def _downsample(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    step = (len(items) - 1) / (limit - 1)
    return [items[round(i * step)] for i in range(limit)]


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _location_confidence_notes(quality: str, segment_type: str) -> list[str]:
    notes = []
    if quality == "good":
        notes.append("主要由 good GPS 点构成，可用于路径判断。")
    elif quality == "rough":
        notes.append("GPS 精度有限，适合判断大致区域或附近 POI。")
    else:
        notes.append("GPS 精度差，仅作为弱证据。")
    if segment_type == "indoor_low_gps":
        notes.append("可能是室内或低精度停留，不应把漂移当成真实移动。")
    return notes


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def wgs84_to_gcj02_dict(lat: float, lng: float) -> dict[str, float]:
    gcj_lat, gcj_lng = wgs84_to_gcj02(lat, lng)
    return {"lat": gcj_lat, "lng": gcj_lng}


def wgs84_to_gcj02(lat: float, lng: float) -> tuple[float, float]:
    if not _in_china(lat, lng):
        return lat, lng
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - 0.00669342162296594323 * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((6335552.717000426 * magic) / (sqrt_magic * magic) * math.pi)
    dlng = (dlng * 180.0) / (6378245.0 / sqrt_magic * math.cos(radlat) * math.pi)
    return lat + dlat, lng + dlng


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def _in_china(lat: float, lng: float) -> bool:
    return 3.86 <= lng <= 135.05 and 0.83 <= lat <= 53.55


def _as_list(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _format_local_time(utc_sec: float | None) -> str:
    if utc_sec is None:
        return "-"
    return datetime.fromtimestamp(utc_sec).strftime("%Y-%m-%d %H:%M:%S")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
