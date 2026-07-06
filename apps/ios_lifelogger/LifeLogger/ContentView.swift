import AVFoundation
import SwiftUI

struct ContentView: View {
    @StateObject private var logger = LoggerViewModel()
    @State private var showSettings = false
    @State private var showFiles = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if logger.isRecording {
                RecordingView(logger: logger)
            } else {
                MainView(
                    logger: logger,
                    showSettings: $showSettings,
                    showFiles: $showFiles
                )
            }
        }
        .task {
            await logger.preparePreview()
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(logger: logger)
        }
        .sheet(isPresented: $showFiles) {
            FilesView()
        }
        .alert("LifeLogger", isPresented: $logger.showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(logger.alertMessage)
        }
    }
}

private struct MainView: View {
    @ObservedObject var logger: LoggerViewModel
    @Binding var showSettings: Bool
    @Binding var showFiles: Bool

    var body: some View {
        ZStack {
            CameraPreview(session: logger.previewSession, settings: logger.settings.camera)
                .ignoresSafeArea()
                .background(Color.black)

            LinearGradient(
                colors: [.black.opacity(0.72), .clear, .black.opacity(0.78)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack {
                VStack(spacing: 6) {
                    Text("LifeLogger")
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(.white)
                    Text("\(logger.statusText) · \(logger.cameraLabel)")
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.82))
                }
                .padding(.top, 18)

                StatusPanel(rows: logger.idleStatusRows)
                    .padding(.top, 10)
                    .padding(.horizontal, 16)
                    .opacity(0.92)

                Spacer()

                HStack(alignment: .center, spacing: 34) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .font(.title2.weight(.semibold))
                            .frame(width: 54, height: 54)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Settings")

                    Button {
                        Task { await logger.startRecording() }
                    } label: {
                        Image(systemName: "record.circle.fill")
                            .font(.system(size: 42, weight: .semibold))
                            .frame(width: 62, height: 62)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .accessibilityLabel("Start")

                    Button {
                        ActivityPresenter.openDocuments()
                    } label: {
                        Image(systemName: "folder")
                            .font(.title2.weight(.semibold))
                            .frame(width: 54, height: 54)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Files")
                }
                .controlSize(.large)
                .padding(.bottom, 26)
            }
        }
        .onAppear {
            logger.restartPreview()
        }
    }
}

private struct RecordingView: View {
    @ObservedObject var logger: LoggerViewModel

    var body: some View {
        VStack {
            Spacer()

            StatusPanel(rows: logger.recordingStatusRows)
                .opacity(logger.settings.power.showRecordingHUD ? 1 : 0.08)

            Spacer()

            Text("Long press anywhere for 3 seconds to stop")
                .font(.system(.footnote, design: .monospaced))
                .foregroundStyle(.white.opacity(logger.settings.power.showRecordingHUD ? 0.62 : 0.12))
                .multilineTextAlignment(.center)
                .padding(.bottom, 24)
        }
        .padding(24)
        .contentShape(Rectangle())
        .onLongPressGesture(minimumDuration: 3.0) {
            Task { await logger.stopRecording() }
        }
    }
}

private struct StatusPanel: View {
    let rows: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(rows, id: \.self) { row in
                Text(row)
                    .font(.system(.footnote, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.75))
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct SettingsView: View {
    @ObservedObject var logger: LoggerViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var draft: LoggerSettings

    init(logger: LoggerViewModel) {
        self.logger = logger
        _draft = State(initialValue: logger.settings)
    }

    var body: some View {
        NavigationView {
            Form {
                Section("Camera") {
                    Toggle("Enable video clips", isOn: binding(\.camera.enabled))
                    Picker("Resolution", selection: binding(\.camera.resolution)) {
                        ForEach(CameraFormatCatalog.resolutionOptions(), id: \.self) { resolution in
                            Text(resolution).tag(resolution)
                        }
                    }
                    Picker("Frame rate", selection: binding(\.camera.frameRate)) {
                        ForEach(CameraFormatCatalog.frameRateOptions(for: draft.camera.resolution), id: \.self) { fps in
                            Text("\(Int(fps)) fps").tag(fps)
                        }
                    }
                    Picker("Stabilization", selection: binding(\.camera.stabilizationMode)) {
                        ForEach(CameraFormatCatalog.stabilizationOptions(for: draft.camera)) { option in
                            Text(option.title)
                                .foregroundStyle(option.isSupported ? .primary : .secondary)
                                .tag(option.id)
                                .disabled(!option.isSupported)
                        }
                    }
                    Toggle("Preview matches recording", isOn: binding(\.camera.previewMatchesRecording))
                    Picker("Max exposure", selection: binding(\.camera.maxExposureDurationMS)) {
                        ForEach(CameraFormatCatalog.exposureOptionsMS, id: \.self) { value in
                            Text("\(Int(value)) ms").tag(value)
                        }
                    }
                    Toggle("Auto Focus", isOn: binding(\.camera.autoFocus))
                    Stepper(
                        "Interval \(Int(draft.camera.clipIntervalSec / 60)) min",
                        value: binding(\.camera.clipIntervalSec),
                        in: 60...3600,
                        step: 60
                    )
                    Stepper(
                        "Clip duration \(Int(draft.camera.clipDurationSec)) sec",
                        value: binding(\.camera.clipDurationSec),
                        in: 2...120,
                        step: 1
                    )
                    Text("Clip duration is clamped below interval when saving.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("For minimum shake, use Strongest Available stabilization.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("When enabled, preview requests the same stabilization mode as recording.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Audio") {
                    Toggle("Enable continuous audio", isOn: binding(\.audio.enabled))
                    Stepper(
                        "Segment \(Int(draft.audio.segmentDurationSec / 60)) min",
                        value: binding(\.audio.segmentDurationSec),
                        in: 60...1800,
                        step: 60
                    )
                }

                SensorSection(title: "Location", enabled: binding(\.location.enabled), hz: binding(\.location.hz), range: 0.1...1.0, step: 0.1)
                SensorSection(title: "Barometer", enabled: binding(\.barometer.enabled), hz: binding(\.barometer.hz), range: 0.2...5.0, step: 0.2)
                SensorSection(title: "Magnetometer", enabled: binding(\.magnetometer.enabled), hz: binding(\.magnetometer.hz), range: 1.0...10.0, step: 1.0)
                SensorSection(title: "Device Motion", enabled: binding(\.deviceMotion.enabled), hz: binding(\.deviceMotion.hz), range: 1.0...20.0, step: 1.0)

                Section("Power") {
                    Toggle("Dim screen while recording", isOn: binding(\.power.dimScreen))
                    Toggle("Disable auto-lock while recording", isOn: binding(\.power.disableIdleTimer))
                    Toggle("Show recording HUD", isOn: binding(\.power.showRecordingHUD))
                }
            }
            .navigationTitle("Settings")
            .onAppear {
                ensureSupportedStabilization()
            }
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        commit()
                        dismiss()
                    }
                }
            }
            .onDisappear {
                commit()
            }
            .onChange(of: draft.camera.resolution) { resolution in
                let options = CameraFormatCatalog.frameRateOptions(for: resolution)
                if !options.contains(draft.camera.frameRate) {
                    draft.camera.frameRate = options.first ?? 30
                    commit()
                }
                ensureSupportedStabilization()
            }
            .onChange(of: draft.camera.frameRate) { _ in
                ensureSupportedStabilization()
            }
        }
    }

    private func binding<Value>(_ keyPath: WritableKeyPath<LoggerSettings, Value>) -> Binding<Value> {
        Binding(
            get: { draft[keyPath: keyPath] },
            set: { value in
                draft[keyPath: keyPath] = value
                commit()
            }
        )
    }

    private func commit() {
        logger.applySettings(draft)
        draft = logger.settings
    }

    private func ensureSupportedStabilization() {
        let options = CameraFormatCatalog.stabilizationOptions(for: draft.camera)
        let selectedIsSupported = options.contains { option in
            option.id == draft.camera.stabilizationMode && option.isSupported
        }
        if !selectedIsSupported {
            draft.camera.stabilizationMode = "best"
            commit()
        }
    }
}

private struct SensorSection: View {
    let title: String
    @Binding var enabled: Bool
    @Binding var hz: Double
    let range: ClosedRange<Double>
    let step: Double

    var body: some View {
        Section(title) {
            Toggle("Enable", isOn: $enabled)
            Stepper(
                "\(hzText) Hz",
                value: $hz,
                in: range,
                step: step
            )
        }
    }

    private var hzText: String {
        hz < 1 ? String(format: "%.1f", hz) : String(format: "%.0f", hz)
    }
}

private struct FilesView: View {
    @State private var sessions = SessionFileStore.listSessions()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            List {
                if sessions.isEmpty {
                    Text("No sessions yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(sessions, id: \.self) { url in
                        Button {
                            ActivityPresenter.present(url: url)
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(url.lastPathComponent)
                                    .font(.headline)
                                Text(url.path)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .navigationTitle("Files")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        sessions = SessionFileStore.listSessions()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
    }
}

enum ActivityPresenter {
    static func openDocuments() {
        var components = URLComponents(url: SessionFileStore.documentsURL, resolvingAgainstBaseURL: false)
        components?.scheme = "shareddocuments"
        if let url = components?.url {
            UIApplication.shared.open(url)
        }
    }

    static func present(url: URL) {
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let root = scene.windows.first(where: { $0.isKeyWindow })?.rootViewController else {
            return
        }
        let controller = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        root.present(controller, animated: true)
    }
}

#Preview {
    ContentView()
}
