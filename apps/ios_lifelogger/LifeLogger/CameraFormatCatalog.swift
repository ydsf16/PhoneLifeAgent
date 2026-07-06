import AVFoundation
import Foundation

enum CameraFormatCatalog {
    struct StabilizationOption: Identifiable, Equatable {
        let id: String
        let title: String
        let isSupported: Bool
    }

    static let exposureOptionsMS: [Double] = [1, 5, 10, 20, 30, 50, 100]

    static func resolutionOptions(for device: AVCaptureDevice? = CameraDeviceSelector.preferredCamera()) -> [String] {
        guard let device else { return ["1920x1080"] }
        let values = Set(device.formats.compactMap { format -> String? in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard isRecordable(width: Int(dimensions.width), height: Int(dimensions.height)) else { return nil }
            return "\(dimensions.width)x\(dimensions.height)"
        })
        return values.sorted { lhs, rhs in
            area(lhs) > area(rhs)
        }
    }

    static func frameRateOptions(
        for resolution: String,
        device: AVCaptureDevice? = CameraDeviceSelector.preferredCamera()
    ) -> [Double] {
        guard let device else { return [24, 30] }
        let fpsValues = Set(device.formats.flatMap { format -> [Double] in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard "\(dimensions.width)x\(dimensions.height)" == resolution else { return [] }
            return format.videoSupportedFrameRateRanges.flatMap { range in
                [24, 30, 60, 120].filter { value in
                    value >= range.minFrameRate && value <= range.maxFrameRate
                }
            }
        })
        let sorted = fpsValues.sorted()
        return sorted.isEmpty ? [30] : sorted
    }

    static func configure(device: AVCaptureDevice, settings: LoggerSettings.Camera) {
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }

            if let format = preferredFormat(for: device, settings: settings) {
                device.activeFormat = format
            }

            let fps = supportedFrameRate(for: device.activeFormat, requested: settings.frameRate)
            let duration = frameDuration(for: fps)
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration

            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            } else if device.isExposureModeSupported(.autoExpose) {
                device.exposureMode = .autoExpose
            }
            let requestedExposure = CMTimeMakeWithSeconds(
                settings.maxExposureDurationMS / 1000.0,
                preferredTimescale: 1_000_000_000
            )
            device.activeMaxExposureDuration = clampedExposureDuration(requestedExposure, for: device.activeFormat)

            if settings.autoFocus {
                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                } else if device.isFocusModeSupported(.autoFocus) {
                    device.focusMode = .autoFocus
                }
            } else if device.isFocusModeSupported(.locked) {
                device.focusMode = .locked
            }
        } catch {
            return
        }
    }

    static func preferredFormat(
        for device: AVCaptureDevice,
        settings: LoggerSettings.Camera
    ) -> AVCaptureDevice.Format? {
        let target = settings.resolution
        return device.formats.filter { format in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return "\(dimensions.width)x\(dimensions.height)" == target &&
                isRecordable(width: Int(dimensions.width), height: Int(dimensions.height))
        }.sorted { lhs, rhs in
            frameRateDistance(lhs, requested: settings.frameRate) < frameRateDistance(rhs, requested: settings.frameRate)
        }.first
    }

    static func supportedFrameRate(for format: AVCaptureDevice.Format, requested: Double) -> Double {
        guard let range = format.videoSupportedFrameRateRanges.min(by: { lhs, rhs in
            let lhsClamped = min(max(requested, lhs.minFrameRate), lhs.maxFrameRate)
            let rhsClamped = min(max(requested, rhs.minFrameRate), rhs.maxFrameRate)
            return abs(lhsClamped - requested) < abs(rhsClamped - requested)
        }) else {
            return requested
        }
        return min(max(requested, range.minFrameRate), range.maxFrameRate)
    }

    static func stabilizationOptions(
        for settings: LoggerSettings.Camera,
        device: AVCaptureDevice? = CameraDeviceSelector.preferredCamera()
    ) -> [StabilizationOption] {
        let format = device.flatMap { preferredFormat(for: $0, settings: settings) ?? $0.activeFormat }
        return stabilizationDescriptors.map { descriptor in
            StabilizationOption(
                id: descriptor.id,
                title: descriptor.title,
                isSupported: isStabilizationSupported(descriptor.id, format: format)
            )
        }
    }

    static func stabilizationDisplayName(for id: String) -> String {
        stabilizationDescriptors.first { $0.id == id }?.title ?? id
    }

    static func stabilizationMode(for id: String) -> AVCaptureVideoStabilizationMode? {
        stabilizationDescriptors.first { $0.id == id }?.mode
    }

    static func stabilizationModeName(_ mode: AVCaptureVideoStabilizationMode) -> String {
        switch mode {
        case .off:
            return "off"
        case .standard:
            return "standard"
        case .cinematic:
            return "cinematic"
        case .cinematicExtended:
            return "cinematic_extended"
        case .cinematicExtendedEnhanced:
            return "cinematic_extended_enhanced"
        case .previewOptimized:
            return "preview_optimized"
        case .lowLatency:
            return "low_latency"
        case .auto:
            return "auto"
        @unknown default:
            return "unknown_\(mode.rawValue)"
        }
    }

    static func bestAvailableStabilizationMode(
        for format: AVCaptureDevice.Format
    ) -> AVCaptureVideoStabilizationMode {
        var candidates: [AVCaptureVideoStabilizationMode] = []
        if #available(iOS 18.0, *) {
            candidates.append(.cinematicExtendedEnhanced)
            candidates.append(.previewOptimized)
        }
        candidates += [
            .cinematicExtended,
            .cinematic,
            .standard,
            .auto
        ]
        return candidates.first {
            format.isVideoStabilizationModeSupported($0)
        } ?? .auto
    }

    private static func frameRateDistance(_ format: AVCaptureDevice.Format, requested: Double) -> Double {
        abs(supportedFrameRate(for: format, requested: requested) - requested)
    }

    private static var stabilizationDescriptors: [(id: String, title: String, mode: AVCaptureVideoStabilizationMode?)] {
        var descriptors: [(String, String, AVCaptureVideoStabilizationMode?)] = [
            ("best", "Strongest Available", nil),
            ("auto", "Auto", .auto),
            ("off", "Off", .off),
            ("standard", "Standard", .standard),
            ("cinematic", "Cinematic", .cinematic),
            ("cinematic_extended", "Cinematic Extended", .cinematicExtended)
        ]
        if #available(iOS 18.0, *) {
            descriptors.append(("cinematic_extended_enhanced", "Cinematic Extended Enhanced", .cinematicExtendedEnhanced))
            descriptors.append(("preview_optimized", "Preview Optimized", .previewOptimized))
        }
        if #available(iOS 26.0, *) {
            descriptors.append(("low_latency", "Low Latency", .lowLatency))
        }
        return descriptors
    }

    private static func isStabilizationSupported(
        _ id: String,
        format: AVCaptureDevice.Format?
    ) -> Bool {
        guard let format else { return id == "best" || id == "off" }
        if id == "best" || id == "off" || id == "auto" {
            return true
        }
        guard let mode = stabilizationMode(for: id) else { return false }
        return format.isVideoStabilizationModeSupported(mode)
    }

    private static func frameDuration(for fps: Double) -> CMTime {
        let scale: Int32 = 600
        let value = max(Int64(round(Double(scale) / max(fps, 1))), 1)
        return CMTime(value: value, timescale: scale)
    }

    private static func clampedExposureDuration(_ duration: CMTime, for format: AVCaptureDevice.Format) -> CMTime {
        if CMTimeCompare(duration, format.minExposureDuration) < 0 {
            return format.minExposureDuration
        }
        if CMTimeCompare(duration, format.maxExposureDuration) > 0 {
            return format.maxExposureDuration
        }
        return duration
    }

    private static func isRecordable(width: Int, height: Int) -> Bool {
        guard width > 0, height > 0 else { return false }
        let pixels = width * height
        let longEdge = max(width, height)
        let shortEdge = min(width, height)
        let aspect = Double(longEdge) / Double(shortEdge)
        let isFourByThree = abs(aspect - 4.0 / 3.0) < 0.03
        let isSixteenByNine = abs(aspect - 16.0 / 9.0) < 0.03
        if isFourByThree {
            return longEdge <= 1920 && shortEdge <= 1440
        }
        if isSixteenByNine {
            return longEdge <= 3840 && shortEdge <= 2160 && pixels <= 3840 * 2160
        }
        return false
    }

    private static func area(_ resolution: String) -> Int {
        let parts = resolution.split(separator: "x")
        guard parts.count == 2,
              let width = Int(parts[0]),
              let height = Int(parts[1]) else {
            return 0
        }
        return width * height
    }
}
