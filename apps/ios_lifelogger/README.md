# LifeLogger

LifeLogger is an iOS life-logging capture app that uses an iPhone as an always-on multimodal logger.

[中文 README](README.zh-CN.md)

## Goal

- Show a full-screen camera preview before recording so the user can adjust the wearing angle.
- Enter a low-power black recording screen after Start.
- Record continuous audio and intermittent video clips.
- Record Location, Device Motion, Magnetometer, and Barometer streams.
- Keep all capture data local on device.
- Write aligned session files for later AI analysis.

## Current MVP

- App display name: `LifeLogger`
- Target/module: `LifeLogger`
- Bundle ID: `com.grape.LifeLogger`
- Minimum iOS version: `15.4`
- Camera selection: prefer Ultra Wide, fall back to Wide.
- Default video policy: one 10-second clip every 2 minutes.
- Default sensor policy: 1Hz for Location, Barometer, Magnetometer, and Device Motion.
- Audio policy: continuous M4A recording, segmented every 5 minutes by default.
- Stop protection: long press anywhere on the recording screen for 3 seconds to stop.

## Features

- Full-screen preview home screen.
- Bottom overlay icon buttons for Settings, Start, and Files.
- Settings are saved immediately and applied to the preview.
- Settings are persisted at:

```text
Application Support/LifeLogger/lifelogger_settings.json
```

- Files opens the app Documents directory in the system Files app.
- Recording mode can dim the screen and disable auto-lock.
- Video clip index records camera intrinsics when iOS provides them:
  - `fx_px`
  - `fy_px`
  - `cx_px`
  - `cy_px`

## Session Layout

```text
session_YYYYMMDD_HHMMSS/
  capture_policy.json
  video/
    clip_000001_YYYYMMDD_HHMMSS.mp4
    clip_index.csv
  audio/
    audio_000001_YYYYMMDD_HHMMSS.m4a
    audio_index.csv
  location/
    geo_location.csv
  motion/
    device_motion.csv
  environment/
    barometer.csv
    magnetometer.csv
```

## Time Model

Aligned streams use two timestamps:

- `sensor_sec`: monotonic sensor time for offline multimodal alignment.
- `utc_sec`: Unix UTC time for maps, calendar correlation, and human-readable timelines.

Use `sensor_sec` from CSV indexes as the primary alignment key in post-processing.

## Build

```bash
open LifeLogger.xcodeproj
```

Simulator build:

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -project LifeLogger.xcodeproj \
  -scheme LifeLogger \
  -sdk iphonesimulator \
  -configuration Debug \
  CODE_SIGNING_ALLOWED=NO \
  build
```

Device builds require an Apple Team and signing configured in Xcode.

