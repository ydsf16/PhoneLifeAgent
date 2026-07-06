import AVFoundation
import Foundation
import simd

final class CameraClipLogger: NSObject, AVCaptureFileOutputRecordingDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let directory: URL
    private let intervalSec: Double
    private let durationSec: Double
    private let cameraSettings: LoggerSettings.Camera
    private let indexWriter: CSVWriter
    private let queue = DispatchQueue(label: "com.grape.lifelog.camera")
    private var timer: DispatchSourceTimer?
    private var session: AVCaptureSession?
    private var movieOutput: AVCaptureMovieFileOutput?
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private var clipID = 0
    private var currentStartSensorSec = 0.0
    private var currentStartUTCSec = 0.0
    private var currentPath = ""
    private var currentCameraType = "unavailable"
    private var currentIntrinsics: CameraIntrinsics?
    private var currentPreferredStabilizationMode = "unavailable"
    private var currentActiveStabilizationMode = "unavailable"
    private var isStopping = false
    private var stopCompletion: (() -> Void)?

    private struct CameraIntrinsics {
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float
    }

    init(
        directory: URL,
        intervalSec: Double,
        durationSec: Double,
        cameraSettings: LoggerSettings.Camera
    ) {
        self.directory = directory
        self.intervalSec = intervalSec
        self.durationSec = durationSec
        self.cameraSettings = cameraSettings
        self.indexWriter = CSVWriter(
            url: directory.appendingPathComponent("clip_index.csv"),
            header: "clip_id,file_path,camera_type,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec,width,height,fps,fx_px,fy_px,cx_px,cy_px,preferred_stabilization_mode,active_stabilization_mode"
        )
    }

    func start() {
        queue.async {
            self.startClip()
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(deadline: .now() + self.intervalSec, repeating: self.intervalSec)
            timer.setEventHandler { [weak self] in
                self?.startClip()
            }
            self.timer = timer
            timer.resume()
        }
    }

    func stop(completion: (() -> Void)? = nil) {
        queue.async {
            self.isStopping = true
            self.stopCompletion = completion
            self.timer?.cancel()
            self.timer = nil
            self.stopClip()
            if self.movieOutput?.isRecording != true {
                self.indexWriter.close()
                self.stopCompletion?()
                self.stopCompletion = nil
            }
        }
    }

    private func startClip() {
        guard session == nil else { return }
        guard let device = CameraDeviceSelector.preferredCamera(),
              let input = try? AVCaptureDeviceInput(device: device) else {
            return
        }
        CameraFormatCatalog.configure(device: device, settings: cameraSettings)

        let session = AVCaptureSession()
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureMovieFileOutput()
        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        let videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            if let connection = videoDataOutput.connection(with: .video),
               connection.isCameraIntrinsicMatrixDeliverySupported {
                connection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }
        session.commitConfiguration()

        guard !session.inputs.isEmpty, !session.outputs.isEmpty else { return }

        let stabilization = configureSelectedStabilization(for: output, device: device)

        self.session = session
        self.movieOutput = output
        self.videoDataOutput = videoDataOutput
        self.currentCameraType = CameraDeviceSelector.cameraType(for: device)
        self.currentIntrinsics = nil
        self.currentPreferredStabilizationMode = stabilization.preferred
        self.currentActiveStabilizationMode = stabilization.active
        self.clipID += 1

        let stamp = SessionFileStore.timestampForFile()
        let filename = String(format: "clip_%06d_%@.mp4", clipID, stamp)
        let url = directory.appendingPathComponent(filename)
        currentPath = "video/\(filename)"
        currentStartSensorSec = Timebase.sensorSec
        currentStartUTCSec = Timebase.utcSec

        session.startRunning()
        output.startRecording(to: url, recordingDelegate: self)
        currentActiveStabilizationMode = CameraFormatCatalog.stabilizationModeName(
            output.connection(with: .video)?.activeVideoStabilizationMode ?? .off
        )

        queue.asyncAfter(deadline: .now() + durationSec) { [weak self] in
            self?.stopClip()
        }
    }

    private func stopClip() {
        if movieOutput?.isRecording == true {
            movieOutput?.stopRecording()
        } else {
            finishSession()
        }
    }

    private func finishSession() {
        session?.stopRunning()
        session = nil
        movieOutput = nil
        videoDataOutput = nil
    }

    func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        queue.async {
            let endSensorSec = Timebase.sensorSec
            let endUTCSec = Timebase.utcSec
            let duration = max(0, endSensorSec - self.currentStartSensorSec)
            let dimensions = self.videoDimensions(url: outputFileURL)
            let fps = self.videoFPS(url: outputFileURL)
            let intrinsics = self.currentIntrinsics
            let activeStabilizationMode = connections.first
                .map { CameraFormatCatalog.stabilizationModeName($0.activeVideoStabilizationMode) } ??
                self.currentActiveStabilizationMode
            self.indexWriter.writeLine(String(
                format: "%d,%@,%@,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.3f,%.9f,%.9f,%.9f,%.9f,%@,%@",
                self.clipID,
                self.currentPath,
                self.currentCameraType,
                self.currentStartSensorSec,
                endSensorSec,
                self.currentStartUTCSec,
                endUTCSec,
                duration,
                dimensions.width,
                dimensions.height,
                fps,
                intrinsics?.fx ?? .nan,
                intrinsics?.fy ?? .nan,
                intrinsics?.cx ?? .nan,
                intrinsics?.cy ?? .nan,
                self.currentPreferredStabilizationMode,
                activeStabilizationMode
            ))
            self.finishSession()
            if self.isStopping {
                self.indexWriter.close()
                self.stopCompletion?()
                self.stopCompletion = nil
            }
        }
    }

    private func configureSelectedStabilization(
        for output: AVCaptureMovieFileOutput,
        device: AVCaptureDevice
    ) -> (preferred: String, active: String) {
        guard let connection = output.connection(with: .video),
              connection.isVideoStabilizationSupported else {
            return ("unsupported", "unsupported")
        }

        let best = CameraFormatCatalog.bestAvailableStabilizationMode(for: device.activeFormat)
        let requested = CameraFormatCatalog.stabilizationMode(for: cameraSettings.stabilizationMode)
        let preferred: AVCaptureVideoStabilizationMode
        if cameraSettings.stabilizationMode == "best" {
            preferred = best
        } else if let requested,
                  requested == .off ||
                    requested == .auto ||
                    device.activeFormat.isVideoStabilizationModeSupported(requested) {
            preferred = requested
        } else {
            preferred = best
        }

        connection.preferredVideoStabilizationMode = preferred
        return (
            CameraFormatCatalog.stabilizationModeName(preferred),
            CameraFormatCatalog.stabilizationModeName(connection.activeVideoStabilizationMode)
        )
    }

    private func videoDimensions(url: URL) -> (width: Int, height: Int) {
        let asset = AVAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else { return (0, 0) }
        let size = track.naturalSize.applying(track.preferredTransform)
        return (Int(abs(size.width)), Int(abs(size.height)))
    }

    private func videoFPS(url: URL) -> Double {
        let asset = AVAsset(url: url)
        return Double(asset.tracks(withMediaType: .video).first?.nominalFrameRate ?? 0)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard currentIntrinsics == nil,
              let intrinsics = Self.cameraIntrinsics(from: sampleBuffer) else {
            return
        }
        currentIntrinsics = intrinsics
    }

    private static func cameraIntrinsics(from sampleBuffer: CMSampleBuffer) -> CameraIntrinsics? {
        guard let attachment = CMGetAttachment(
            sampleBuffer,
            key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
            attachmentModeOut: nil
        ) else {
            return nil
        }
        let data = attachment as! CFData
        guard CFDataGetLength(data) >= MemoryLayout<matrix_float3x3>.size else {
            return nil
        }
        let matrix = CFDataGetBytePtr(data).withMemoryRebound(to: matrix_float3x3.self, capacity: 1) { $0.pointee }
        return CameraIntrinsics(
            fx: matrix.columns.0.x,
            fy: matrix.columns.1.y,
            cx: matrix.columns.2.x,
            cy: matrix.columns.2.y
        )
    }
}
