# LifeLogger

LifeLogger 是一个 iOS 生活记录采集 App，用 iPhone 模拟 Always-on 多模态记录设备。

[English README](README.md)

## 项目目标

- 打开 App 后显示全屏相机 Preview，方便调整佩戴角度。
- 点击 Start 后进入黑屏低功耗录制界面。
- 录制期间连续记录音频，间歇记录视频 clip。
- 同步记录 Location、Device Motion、Magnetometer、Barometer。
- 所有数据保存在本地，不上传网络。
- 每个 session 都写入统一的文件结构和时间戳，方便后续 AI 分析。

## 当前 MVP

- App 名称：`LifeLogger`
- Target/module：`LifeLogger`
- Bundle ID：`com.grape.LifeLogger`
- 最低 iOS 版本：`15.4`
- 相机策略：优先使用 Ultra Wide，没有则 fallback 到 Wide。
- 默认视频策略：每 2 分钟录制 10 秒。
- 默认传感器频率：Location、Barometer、Magnetometer、Device Motion 都是 1Hz。
- 音频策略：连续录制，默认每 5 分钟切一个 M4A 分片。
- Stop 防误触：录制中长按屏幕任意位置 3 秒停止。

## 主要功能

- 全屏 Preview 主界面。
- Settings / Start / Files 三个底部图标按钮。
- Settings 修改后立即保存并刷新 Preview。
- 设置文件保存到：

```text
Application Support/LifeLogger/lifelogger_settings.json
```

- Files 按钮会打开系统 Files 中的 App Documents 目录。
- 录制时可降低屏幕亮度并禁用自动锁屏。
- 视频 clip index 会记录相机内参：
  - `fx_px`
  - `fy_px`
  - `cx_px`
  - `cy_px`

## Session 文件结构

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

## 时间戳模型

所有可对齐的数据都使用两种时间：

- `sensor_sec`：单调递增时间，适合多传感器离线对齐。
- `utc_sec`：Unix UTC 时间，适合地图、日历、人类可读时间线。

后处理时以 CSV index 里的 `sensor_sec` 作为主对齐键。

## 构建

```bash
open LifeLogger.xcodeproj
```

命令行模拟器构建：

```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild \
  -project LifeLogger.xcodeproj \
  -scheme LifeLogger \
  -sdk iphonesimulator \
  -configuration Debug \
  CODE_SIGNING_ALLOWED=NO \
  build
```

真机运行需要在 Xcode 中配置 Apple Team 和签名。

