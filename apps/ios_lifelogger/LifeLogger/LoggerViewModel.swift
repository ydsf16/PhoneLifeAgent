import AVFoundation
import CoreLocation
import SwiftUI

@MainActor
final class LoggerViewModel: ObservableObject {
    @Published var settings = LoggerSettings.load()
    @Published var isRecording = false
    @Published var statusText = "Preview ready"
    @Published var elapsedText = "00:00:00"
    @Published var freeSpaceText = "--"
    @Published var remainingText = "--"
    @Published var sessionName = "--"
    @Published var showAlert = false
    @Published var alertMessage = ""

    private let previewController = PreviewCameraController()
    private let locationLogger = LocationLogger()
    private var motionLogger: MotionSensorLogger?
    private var audioLogger: AudioSegmentLogger?
    private var cameraLogger: CameraClipLogger?
    private var session: RecordingSession?
    private var startDate: Date?
    private var statusTimer: Timer?
    private var previousBrightness = UIScreen.main.brightness
    private var previousIdleDisabled = UIApplication.shared.isIdleTimerDisabled

    var previewSession: AVCaptureSession {
        previewController.session
    }

    var cameraLabel: String {
        "Camera: \(previewController.cameraType)"
    }

    var idleStatusRows: [String] {
        [
            "Camera \(settings.camera.enabled ? "on" : "off") \(Int(settings.camera.clipIntervalSec / 60))m/\(Int(settings.camera.clipDurationSec))s",
            "\(settings.camera.resolution) \(Int(settings.camera.frameRate))fps exp<=\(Int(settings.camera.maxExposureDurationMS))ms AF \(settings.camera.autoFocus ? "on" : "off")",
            "Stabilization \(CameraFormatCatalog.stabilizationDisplayName(for: settings.camera.stabilizationMode))",
            "Audio \(settings.audio.enabled ? "on" : "off") segment \(Int(settings.audio.segmentDurationSec / 60))m",
            "Location \(settings.location.enabled ? hzText(settings.location.hz) : "off")  Motion \(settings.deviceMotion.enabled ? hzText(settings.deviceMotion.hz) : "off")",
            "Free \(freeSpaceText)"
        ]
    }

    var recordingStatusRows: [String] {
        [
            "REC \(elapsedText)",
            "SESSION \(sessionName)",
            "STAB \(CameraFormatCatalog.stabilizationDisplayName(for: settings.camera.stabilizationMode))",
            "FREE \(freeSpaceText)",
            "REMAIN \(remainingText)"
        ]
    }

    func preparePreview() async {
        updateDiskText()
        await requestInitialPermissions()
        previewController.configureAndStart(settings: settings.camera)
    }

    func restartPreview() {
        guard !isRecording else { return }
        previewController.configureAndStart(settings: settings.camera)
    }

    func applySettings(_ updated: LoggerSettings) {
        var sanitized = updated
        sanitized.sanitize()
        settings = sanitized
        settings.save()
        restartPreview()
    }

    func startRecording() async {
        guard !isRecording else { return }
        settings.sanitize()
        settings.save()

        await requestInitialPermissions()
        previewController.stop()

        do {
            let selectedCamera = CameraDeviceSelector.cameraType(for: CameraDeviceSelector.preferredCamera())
            let newSession = try SessionFileStore.createSession(settings: settings, cameraType: selectedCamera)
            session = newSession
            sessionName = newSession.rootURL.lastPathComponent
            startDate = Date()
            previousBrightness = UIScreen.main.brightness
            previousIdleDisabled = UIApplication.shared.isIdleTimerDisabled

            if settings.power.disableIdleTimer {
                UIApplication.shared.isIdleTimerDisabled = true
            }
            if settings.power.dimScreen {
                UIScreen.main.brightness = 0.01
            }

            if settings.audio.enabled {
                let audio = AudioSegmentLogger(
                    directory: newSession.audioURL,
                    segmentDurationSec: settings.audio.segmentDurationSec
                )
                audio.start()
                audioLogger = audio
            }

            if settings.camera.enabled {
                let camera = CameraClipLogger(
                    directory: newSession.videoURL,
                    intervalSec: settings.camera.clipIntervalSec,
                    durationSec: settings.camera.clipDurationSec,
                    cameraSettings: settings.camera
                )
                camera.start()
                cameraLogger = camera
            }

            let motion = MotionSensorLogger()
            motion.start(session: newSession, settings: settings)
            motionLogger = motion

            if settings.location.enabled {
                locationLogger.start(session: newSession, hz: settings.location.hz)
            }

            isRecording = true
            statusText = "Recording"
            startStatusTimer()
            updateStatus()
        } catch {
            previewController.configureAndStart(settings: settings.camera)
            show(message: "Failed to start recording: \(error.localizedDescription)")
        }
    }

    func stopRecording() async {
        guard isRecording else { return }
        let camera = cameraLogger
        let audio = audioLogger
        let motion = motionLogger
        let finishedSession = session
        let stopGroup = DispatchGroup()

        isRecording = false
        statusText = "Stopping"
        statusTimer?.invalidate()
        statusTimer = nil

        cameraLogger = nil
        audioLogger = nil
        motionLogger = nil
        locationLogger.stop()

        UIApplication.shared.isIdleTimerDisabled = previousIdleDisabled
        UIScreen.main.brightness = previousBrightness

        statusText = "Preview ready"
        session = nil
        startDate = nil
        remainingText = "--"
        updateDiskText()

        if let camera {
            stopGroup.enter()
            camera.stop { [weak self] in
                stopGroup.leave()
                Task { @MainActor in
                    guard let self else { return }
                    self.previewController.configureAndStart(settings: self.settings.camera)
                }
            }
        } else {
            previewController.configureAndStart(settings: settings.camera)
        }

        if let audio {
            stopGroup.enter()
            audio.stop {
                stopGroup.leave()
            }
        }

        if let motion {
            stopGroup.enter()
            motion.stop {
                stopGroup.leave()
            }
        }

        stopGroup.notify(queue: .global(qos: .utility)) {
            guard let finishedSession else { return }
            SessionSummaryWriter.writeSummary(for: finishedSession)
        }
    }

    private func requestInitialPermissions() async {
        await requestAVPermission(.video)
        if settings.audio.enabled {
            await requestAVPermission(.audio)
        }
        if settings.location.enabled {
            locationLogger.requestPermission()
        }
    }

    private func requestAVPermission(_ type: AVMediaType) async {
        let status = AVCaptureDevice.authorizationStatus(for: type)
        guard status == .notDetermined else { return }
        _ = await AVCaptureDevice.requestAccess(for: type)
    }

    private func startStatusTimer() {
        statusTimer?.invalidate()
        statusTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.updateStatus()
            }
        }
    }

    private func updateStatus() {
        guard let startDate else {
            updateDiskText()
            return
        }
        let elapsed = Date().timeIntervalSince(startDate)
        elapsedText = durationText(elapsed)
        updateDiskText()

        guard let session else {
            remainingText = "--"
            return
        }
        let bytes = SessionFileStore.allocatedSize(of: session.rootURL)
        guard elapsed > 5, bytes > 0, let free = SessionFileStore.availableDiskBytes() else {
            remainingText = "--"
            return
        }
        let bytesPerSecond = Double(bytes) / elapsed
        remainingText = durationText(Double(free) / max(bytesPerSecond, 1))
    }

    private func updateDiskText() {
        if let free = SessionFileStore.availableDiskBytes() {
            freeSpaceText = ByteCountFormatter.string(fromByteCount: free, countStyle: .file)
        } else {
            freeSpaceText = "--"
        }
    }

    private func show(message: String) {
        alertMessage = message
        showAlert = true
    }

    private func hzText(_ hz: Double) -> String {
        hz < 1 ? String(format: "%.1fHz", hz) : String(format: "%.0fHz", hz)
    }

    private func durationText(_ seconds: TimeInterval) -> String {
        let total = max(Int(seconds.rounded()), 0)
        return String(format: "%02d:%02d:%02d", total / 3600, (total % 3600) / 60, total % 60)
    }
}
