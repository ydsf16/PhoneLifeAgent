import Foundation
import QuartzCore

struct RecordingSession {
    let rootURL: URL
    let videoURL: URL
    let audioURL: URL
    let locationURL: URL
    let motionURL: URL
    let environmentURL: URL
}

enum SessionFileStore {
    static var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static func createSession(settings: LoggerSettings, cameraType: String) throws -> RecordingSession {
        let name = "session_\(Self.timestampForFolder())"
        let root = documentsURL.appendingPathComponent(name, isDirectory: true)
        let video = root.appendingPathComponent("video", isDirectory: true)
        let audio = root.appendingPathComponent("audio", isDirectory: true)
        let location = root.appendingPathComponent("location", isDirectory: true)
        let motion = root.appendingPathComponent("motion", isDirectory: true)
        let environment = root.appendingPathComponent("environment", isDirectory: true)

        for url in [video, audio, location, motion, environment] {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }

        let session = RecordingSession(
            rootURL: root,
            videoURL: video,
            audioURL: audio,
            locationURL: location,
            motionURL: motion,
            environmentURL: environment
        )
        try writeCapturePolicy(settings: settings, cameraType: cameraType, to: root)
        return session
    }

    static func listSessions() -> [URL] {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: documentsURL,
            includingPropertiesForKeys: [.isDirectoryKey, .creationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return urls
            .filter { $0.lastPathComponent.hasPrefix("session_") }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    static func availableDiskBytes() -> Int64? {
        guard let value = try? documentsURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let bytes = value.volumeAvailableCapacityForImportantUsage else {
            return nil
        }
        return bytes
    }

    static func allocatedSize(of url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.totalFileAllocatedSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var total = Int64(0)
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize) ?? 0
            total += Int64(size)
        }
        return total
    }

    static func timestampForFile(date: Date = Date()) -> String {
        fileFormatter.string(from: date)
    }

    private static func timestampForFolder(date: Date = Date()) -> String {
        folderFormatter.string(from: date)
    }

    private static func writeCapturePolicy(settings: LoggerSettings, cameraType: String, to directory: URL) throws {
        let payload: [String: Any] = [
            "created_utc_sec": Date().timeIntervalSince1970,
            "time_model": [
                "sensor_sec": "Monotonic seconds from CACurrentMediaTime/CoreMotion timestamps.",
                "utc_sec": "Unix UTC seconds.",
                "alignment": "Use index CSV sensor_sec fields as the primary alignment key."
            ],
            "camera": [
                "enabled": settings.camera.enabled,
                "selection": "prefer_ultrawide_fallback_wide",
                "selected_camera": cameraType,
                "clip_interval_sec": settings.camera.clipIntervalSec,
                "clip_duration_sec": settings.camera.clipDurationSec,
                "resolution": settings.camera.resolution,
                "frame_rate": settings.camera.frameRate,
                "max_exposure_duration_ms": settings.camera.maxExposureDurationMS,
                "auto_focus": settings.camera.autoFocus,
                "stabilization_mode": settings.camera.stabilizationMode,
                "stabilization_mode_display": CameraFormatCatalog.stabilizationDisplayName(
                    for: settings.camera.stabilizationMode
                )
            ],
            "audio": [
                "enabled": settings.audio.enabled,
                "segment_duration_sec": settings.audio.segmentDurationSec,
                "format": "aac_m4a"
            ],
            "location": ["enabled": settings.location.enabled, "hz": settings.location.hz],
            "barometer": ["enabled": settings.barometer.enabled, "hz": settings.barometer.hz],
            "magnetometer": ["enabled": settings.magnetometer.enabled, "hz": settings.magnetometer.hz],
            "device_motion": ["enabled": settings.deviceMotion.enabled, "hz": settings.deviceMotion.hz]
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("capture_policy.json"), options: .atomic)
    }

    private static let folderFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()

    private static let fileFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()
}

final class CSVWriter {
    private var handle: FileHandle?
    private var pendingWrites = 0

    init(url: URL, header: String) {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try? FileHandle(forWritingTo: url)
        writeLine(header)
    }

    func writeLine(_ line: String) {
        guard let data = (line + "\n").data(using: .utf8) else { return }
        handle?.write(data)
        pendingWrites += 1
        if pendingWrites >= 30 {
            handle?.synchronizeFile()
            pendingWrites = 0
        }
    }

    func close() {
        handle?.synchronizeFile()
        handle?.closeFile()
        handle = nil
    }
}

enum Timebase {
    static var sensorSec: Double {
        CACurrentMediaTime()
    }

    static var utcSec: Double {
        Date().timeIntervalSince1970
    }

    static var utcMinusSensorOffsetSec: Double {
        utcSec - sensorSec
    }
}

enum SessionSummaryWriter {
    static func writeSummary(for session: RecordingSession) {
        let streams = [
            "video": mediaSummary(
                url: session.videoURL.appendingPathComponent("clip_index.csv"),
                startKey: "start_sensor_sec",
                endKey: "end_sensor_sec",
                utcStartKey: "start_utc_sec",
                utcEndKey: "end_utc_sec",
                durationKey: "duration_sec"
            ),
            "audio": mediaSummary(
                url: session.audioURL.appendingPathComponent("audio_index.csv"),
                startKey: "start_sensor_sec",
                endKey: "end_sensor_sec",
                utcStartKey: "start_utc_sec",
                utcEndKey: "end_utc_sec",
                durationKey: "duration_sec"
            ),
            "location": sensorSummary(url: session.locationURL.appendingPathComponent("geo_location.csv")),
            "device_motion": sensorSummary(url: session.motionURL.appendingPathComponent("device_motion.csv")),
            "barometer": sensorSummary(url: session.environmentURL.appendingPathComponent("barometer.csv")),
            "magnetometer": sensorSummary(url: session.environmentURL.appendingPathComponent("magnetometer.csv"))
        ]

        let nonEmpty = streams.values.filter { ($0["rows"] as? Int ?? 0) > 0 }
        let sensorStarts = nonEmpty.compactMap { $0["start_sensor_sec"] as? Double }
        let sensorEnds = nonEmpty.compactMap { $0["end_sensor_sec"] as? Double }
        let utcStarts = nonEmpty.compactMap { $0["start_utc_sec"] as? Double }
        let utcEnds = nonEmpty.compactMap { $0["end_utc_sec"] as? Double }

        let payload: [String: Any] = [
            "generated_utc_sec": Date().timeIntervalSince1970,
            "session_name": session.rootURL.lastPathComponent,
            "session_path": session.rootURL.path,
            "total_size_bytes": SessionFileStore.allocatedSize(of: session.rootURL),
            "start_sensor_sec": jsonValue(sensorStarts.min()),
            "end_sensor_sec": jsonValue(sensorEnds.max()),
            "duration_sensor_sec": jsonValue(duration(start: sensorStarts.min(), end: sensorEnds.max())),
            "start_utc_sec": jsonValue(utcStarts.min()),
            "end_utc_sec": jsonValue(utcEnds.max()),
            "duration_utc_sec": jsonValue(duration(start: utcStarts.min(), end: utcEnds.max())),
            "streams": streams
        ]

        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]) else {
            return
        }
        try? data.write(to: session.rootURL.appendingPathComponent("session_summary.json"), options: .atomic)
    }

    private static func mediaSummary(
        url: URL,
        startKey: String,
        endKey: String,
        utcStartKey: String,
        utcEndKey: String,
        durationKey: String
    ) -> [String: Any] {
        let rows = readCSV(url: url)
        let starts = rows.compactMap { double($0[startKey]) }
        let ends = rows.compactMap { double($0[endKey]) }
        let utcStarts = rows.compactMap { double($0[utcStartKey]) }
        let utcEnds = rows.compactMap { double($0[utcEndKey]) }
        let durations = rows.compactMap { double($0[durationKey]) }
        return [
            "rows": rows.count,
            "start_sensor_sec": jsonValue(starts.min()),
            "end_sensor_sec": jsonValue(ends.max()),
            "duration_sensor_sec": jsonValue(duration(start: starts.min(), end: ends.max())),
            "start_utc_sec": jsonValue(utcStarts.min()),
            "end_utc_sec": jsonValue(utcEnds.max()),
            "duration_utc_sec": jsonValue(duration(start: utcStarts.min(), end: utcEnds.max())),
            "total_recorded_duration_sec": durations.reduce(0, +),
            "average_recorded_duration_sec": jsonValue(average(durations)),
            "max_start_gap_sec": jsonValue(maxGap(starts.sorted())),
            "file_size_bytes": fileSizeForCSVIndexedMedia(indexURL: url, rows: rows)
        ]
    }

    private static func sensorSummary(url: URL) -> [String: Any] {
        let rows = readCSV(url: url)
        let sensorTimes = rows.compactMap { double($0["sensor_sec"]) }
        let utcTimes = rows.compactMap { double($0["utc_sec"]) }
        let sensorDuration = duration(start: sensorTimes.min(), end: sensorTimes.max())
        let averageHz = sensorDuration.map { $0 > 0 ? Double(max(rows.count - 1, 0)) / $0 : 0 }
        return [
            "rows": rows.count,
            "start_sensor_sec": jsonValue(sensorTimes.min()),
            "end_sensor_sec": jsonValue(sensorTimes.max()),
            "duration_sensor_sec": jsonValue(sensorDuration),
            "start_utc_sec": jsonValue(utcTimes.min()),
            "end_utc_sec": jsonValue(utcTimes.max()),
            "duration_utc_sec": jsonValue(duration(start: utcTimes.min(), end: utcTimes.max())),
            "average_hz": jsonValue(averageHz),
            "max_sample_gap_sec": jsonValue(maxGap(sensorTimes.sorted())),
            "file_size_bytes": fileSize(url)
        ]
    }

    private static func readCSV(url: URL) -> [[String: String]] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let lines = text.split(whereSeparator: \.isNewline).map(String.init)
        guard let headerLine = lines.first else { return [] }
        let headers = splitCSVLine(headerLine)
        return lines.dropFirst().compactMap { line in
            guard !line.hasPrefix("#") else { return nil }
            let values = splitCSVLine(line)
            guard values.count >= headers.count else { return nil }
            return Dictionary(uniqueKeysWithValues: zip(headers, values))
        }
    }

    private static func splitCSVLine(_ line: String) -> [String] {
        line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
    }

    private static func double(_ value: String?) -> Double? {
        guard let value, !value.isEmpty else { return nil }
        return Double(value)
    }

    private static func duration(start: Double?, end: Double?) -> Double? {
        guard let start, let end else { return nil }
        return max(0, end - start)
    }

    private static func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func maxGap(_ values: [Double]) -> Double? {
        guard values.count > 1 else { return nil }
        return zip(values.dropLast(), values.dropFirst()).map { $1 - $0 }.max()
    }

    private static func fileSizeForCSVIndexedMedia(indexURL: URL, rows: [[String: String]]) -> Int64 {
        let root = indexURL.deletingLastPathComponent().deletingLastPathComponent()
        return rows.reduce(Int64(0)) { total, row in
            guard let path = row["file_path"] else { return total }
            return total + fileSize(root.appendingPathComponent(path))
        }
    }

    private static func fileSize(_ url: URL) -> Int64 {
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return Int64(size)
    }

    private static func jsonValue(_ value: Double?) -> Any {
        value ?? NSNull()
    }
}
