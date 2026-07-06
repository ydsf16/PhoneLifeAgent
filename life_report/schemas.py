from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MediaRefs:
    clips: list[str] = field(default_factory=list)
    keyframes: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)
    gps_ranges: list[str] = field(default_factory=list)
    motion_ranges: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LifeEvent:
    event_id: str
    start_time: str
    end_time: str
    place: str
    geo_summary: str
    motion: str
    visual_summary: str
    audio_summary: str
    event_summary: str
    first_person_narrative: str
    importance_score: float
    media_refs: MediaRefs

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["media_refs"] = self.media_refs.to_dict()
        return data


@dataclass(frozen=True)
class ReportOverview:
    title: str
    date: str
    time_range: str
    recording_duration: str
    distance: str
    sessions_count: int
    clip_count: int
    audio_count: int
    gps_points: int
    motion_points: int
    staypoint_count: int
    gap_count: int
    first_person_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComicPanel:
    panel_id: int
    event_id: str
    time: str
    place: str
    caption: str
    image_prompt: str
    reference_frames: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VlogSegment:
    order: int
    event_id: str
    clip_id: str
    source_range: str
    reason: str
    subtitle: str
    voiceover: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportState:
    report_id: str
    overview: ReportOverview
    events: list[LifeEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "overview": self.overview.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }
