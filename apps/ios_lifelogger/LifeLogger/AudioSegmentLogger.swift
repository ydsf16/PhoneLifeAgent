import AVFoundation
import Foundation

final class AudioSegmentLogger: NSObject, AVAudioRecorderDelegate {
    private let directory: URL
    private let segmentDurationSec: Double
    private let indexWriter: CSVWriter
    private var logger: AVAudioRecorder?
    private var timer: DispatchSourceTimer?
    private var segmentID = 0
    private var currentStartSensorSec = 0.0
    private var currentStartUTCSec = 0.0
    private var currentPath = ""
    private var isStopping = false
    private let queue = DispatchQueue(label: "com.grape.lifelog.audio")

    init(directory: URL, segmentDurationSec: Double) {
        self.directory = directory
        self.segmentDurationSec = segmentDurationSec
        self.indexWriter = CSVWriter(
            url: directory.appendingPathComponent("audio_index.csv"),
            header: "audio_id,file_path,start_sensor_sec,end_sensor_sec,start_utc_sec,end_utc_sec,duration_sec"
        )
    }

    func start() {
        queue.async {
            self.isStopping = false
            self.configureSession()
            self.startNewSegment()
            self.startTimer()
        }
    }

    func stop(completion: (() -> Void)? = nil) {
        queue.async {
            self.isStopping = true
            self.timer?.cancel()
            self.timer = nil
            self.finishCurrentSegment(stopRecorder: true)
            self.indexWriter.close()
            completion?()
        }
    }

    private func configureSession() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.record, mode: .measurement, options: [])
        try? session.setPreferredSampleRate(24_000)
        try? session.setPreferredInputNumberOfChannels(1)
        try? session.setActive(true)
    }

    private func startTimer() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + segmentDurationSec, repeating: segmentDurationSec)
        timer.setEventHandler { [weak self] in
            self?.rotateSegment()
        }
        self.timer = timer
        timer.resume()
    }

    private func rotateSegment() {
        finishCurrentSegment(stopRecorder: true)
        startNewSegment()
    }

    private func startNewSegment() {
        segmentID += 1
        let stamp = SessionFileStore.timestampForFile()
        let filename = String(format: "audio_%06d_%@.m4a", segmentID, stamp)
        let url = directory.appendingPathComponent(filename)
        currentPath = "audio/\(filename)"
        currentStartSensorSec = Timebase.sensorSec
        currentStartUTCSec = Timebase.utcSec

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 24_000,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 48_000
        ]
        logger = try? AVAudioRecorder(url: url, settings: settings)
        logger?.delegate = self
        logger?.isMeteringEnabled = false
        logger?.record()
    }

    private func finishCurrentSegment(stopRecorder: Bool) {
        guard let logger else { return }

        let recordedDuration = max(0, logger.currentTime)
        if stopRecorder {
            logger.stop()
        }
        self.logger = nil

        let fallbackEndSensorSec = Timebase.sensorSec
        let fallbackEndUTCSec = Timebase.utcSec
        let duration: Double
        let endSensorSec: Double
        let endUTCSec: Double

        if recordedDuration > 0 {
            duration = recordedDuration
            endSensorSec = currentStartSensorSec + recordedDuration
            endUTCSec = currentStartUTCSec + recordedDuration
        } else {
            endSensorSec = fallbackEndSensorSec
            endUTCSec = fallbackEndUTCSec
            duration = max(0, endSensorSec - currentStartSensorSec)
        }

        indexWriter.writeLine(String(
            format: "%d,%@,%.6f,%.6f,%.6f,%.6f,%.6f",
            segmentID,
            currentPath,
            currentStartSensorSec,
            endSensorSec,
            currentStartUTCSec,
            endUTCSec,
            duration
        ))
    }

    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        queue.async {
            guard self.logger === recorder else { return }
            self.finishCurrentSegment(stopRecorder: false)
            guard !self.isStopping else { return }
            self.timer?.cancel()
            self.timer = nil
            self.startNewSegment()
            self.startTimer()
        }
    }
}
