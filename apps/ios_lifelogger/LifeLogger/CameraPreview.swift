import AVFoundation
import SwiftUI
import UIKit

final class PreviewCameraController {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.grape.lifelog.preview")
    private(set) var cameraType = "unavailable"

    func configureAndStart(settings: LoggerSettings.Camera) {
        queue.async {
            self.restartLocked(settings: settings, retriesRemaining: 2)
        }
    }

    func stop() {
        queue.async {
            guard self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    private func restartLocked(settings: LoggerSettings.Camera, retriesRemaining: Int) {
        if session.isRunning {
            session.stopRunning()
        }

        session.beginConfiguration()
        session.sessionPreset = .high
        session.inputs.forEach { session.removeInput($0) }

        if let device = CameraDeviceSelector.preferredCamera(),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            CameraFormatCatalog.configure(device: device, settings: settings)
            session.addInput(input)
            cameraType = CameraDeviceSelector.cameraType(for: device)
        } else {
            cameraType = "unavailable"
        }

        session.commitConfiguration()

        if !session.inputs.isEmpty {
            session.startRunning()
        }

        guard retriesRemaining > 0 else { return }
        queue.asyncAfter(deadline: .now() + 0.8) {
            if !self.session.isRunning || self.session.inputs.isEmpty {
                self.restartLocked(settings: settings, retriesRemaining: retriesRemaining - 1)
            }
        }
    }
}

enum CameraDeviceSelector {
    static func preferredCamera() -> AVCaptureDevice? {
        if let ultraWide = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back) {
            return ultraWide
        }
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
    }

    static func cameraType(for device: AVCaptureDevice?) -> String {
        guard let device else { return "unavailable" }
        switch device.deviceType {
        case .builtInUltraWideCamera:
            return "ultrawide"
        case .builtInWideAngleCamera:
            return "wide"
        default:
            return device.localizedName
        }
    }

}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let settings: LoggerSettings.Camera

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        configurePreviewStabilization(for: view.previewLayer)
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
        configurePreviewStabilization(for: uiView.previewLayer)
    }

    private func configurePreviewStabilization(for layer: AVCaptureVideoPreviewLayer) {
        guard let connection = layer.connection,
              connection.isVideoStabilizationSupported else {
            return
        }

        guard settings.previewMatchesRecording else {
            connection.preferredVideoStabilizationMode = .auto
            return
        }

        let mode: AVCaptureVideoStabilizationMode
        if settings.stabilizationMode == "best" {
            if let device = CameraDeviceSelector.preferredCamera() {
                mode = CameraFormatCatalog.bestAvailableStabilizationMode(for: device.activeFormat)
            } else {
                mode = .auto
            }
        } else {
            mode = CameraFormatCatalog.stabilizationMode(for: settings.stabilizationMode) ?? .auto
        }
        connection.preferredVideoStabilizationMode = mode
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}
