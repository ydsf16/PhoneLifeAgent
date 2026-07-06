import Foundation

struct LoggerSettings: Codable, Equatable {
    var camera = Camera()
    var audio = Audio()
    var location = Sensor(enabled: true, hz: 1.0)
    var barometer = Sensor(enabled: true, hz: 1.0)
    var magnetometer = Sensor(enabled: true, hz: 1.0)
    var deviceMotion = Sensor(enabled: true, hz: 1.0)
    var power = Power()

    struct Camera: Codable, Equatable {
        var enabled = true
        var clipIntervalSec: Double = 120
        var clipDurationSec: Double = 10
        var resolution = "1920x1080"
        var frameRate = 30.0
        var maxExposureDurationMS = 10.0
        var autoFocus = true
        var previewMatchesRecording = true
        var stabilizationMode = "best"

        init() {}

        private enum CodingKeys: String, CodingKey {
            case enabled
            case clipIntervalSec
            case clipDurationSec
            case resolution
            case frameRate
            case maxExposureDurationMS
            case autoFocus
            case previewMatchesRecording
            case stabilizationMode
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            enabled = try values.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
            clipIntervalSec = try values.decodeIfPresent(Double.self, forKey: .clipIntervalSec) ?? 120
            clipDurationSec = try values.decodeIfPresent(Double.self, forKey: .clipDurationSec) ?? 10
            resolution = try values.decodeIfPresent(String.self, forKey: .resolution) ?? "1920x1080"
            frameRate = try values.decodeIfPresent(Double.self, forKey: .frameRate) ?? 30
            maxExposureDurationMS = try values.decodeIfPresent(Double.self, forKey: .maxExposureDurationMS) ?? 10
            autoFocus = try values.decodeIfPresent(Bool.self, forKey: .autoFocus) ?? true
            previewMatchesRecording = try values.decodeIfPresent(Bool.self, forKey: .previewMatchesRecording) ?? true
            stabilizationMode = try values.decodeIfPresent(String.self, forKey: .stabilizationMode) ?? "best"
        }
    }

    struct Audio: Codable, Equatable {
        var enabled = true
        var segmentDurationSec: Double = 300
    }

    struct Sensor: Codable, Equatable {
        var enabled: Bool
        var hz: Double
    }

    struct Power: Codable, Equatable {
        var dimScreen = true
        var disableIdleTimer = true
        var showRecordingHUD = true
    }

    private static let storageKey = "lifelogger.settings.v1"
    private static let legacyStorageKey = "lifelog_recorder.settings.v1"
    static let configFileName = "lifelogger_settings.json"
    private static let legacyConfigFileName = "recorder_settings.json"

    static func load() -> LoggerSettings {
        for url in [configFileURL, legacyConfigFileURL] {
            if let data = try? Data(contentsOf: url),
               let settings = try? JSONDecoder().decode(LoggerSettings.self, from: data) {
                var sanitized = settings
                sanitized.sanitize()
                sanitized.save()
                return sanitized
            }
        }

        for key in [storageKey, legacyStorageKey] {
            if let data = UserDefaults.standard.data(forKey: key),
               let settings = try? JSONDecoder().decode(LoggerSettings.self, from: data) {
                var sanitized = settings
                sanitized.sanitize()
                sanitized.save()
                return sanitized
            }
        }

        return LoggerSettings()
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let prettyData = try? encoder.encode(self) else { return }
        try? FileManager.default.createDirectory(
            at: Self.configDirectoryURL,
            withIntermediateDirectories: true
        )
        try? prettyData.write(to: Self.configFileURL, options: .atomic)
    }

    mutating func sanitize() {
        camera.clipIntervalSec = max(camera.clipIntervalSec, 10)
        camera.clipDurationSec = max(1, min(camera.clipDurationSec, camera.clipIntervalSec - 1))
        camera.frameRate = min(max(camera.frameRate, 1), 120)
        camera.maxExposureDurationMS = min(max(camera.maxExposureDurationMS, 0.1), 1000)
        if !Self.validStabilizationModes.contains(camera.stabilizationMode) {
            camera.stabilizationMode = "best"
        }
        audio.segmentDurationSec = max(audio.segmentDurationSec, 60)
        location.hz = max(0.05, location.hz)
        barometer.hz = max(0.05, barometer.hz)
        magnetometer.hz = max(0.05, magnetometer.hz)
        deviceMotion.hz = max(0.05, deviceMotion.hz)
    }

    private static var configDirectoryURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LifeLogger", isDirectory: true)
    }

    static var configFileURL: URL {
        configDirectoryURL.appendingPathComponent(configFileName)
    }

    private static var legacyConfigFileURL: URL {
        configDirectoryURL.appendingPathComponent(legacyConfigFileName)
    }

    private static let validStabilizationModes: Set<String> = [
        "best",
        "auto",
        "off",
        "standard",
        "cinematic",
        "cinematic_extended",
        "cinematic_extended_enhanced",
        "preview_optimized",
        "low_latency"
    ]
}
