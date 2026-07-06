import CoreLocation
import CoreMotion
import Foundation

final class MotionSensorLogger {
    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let queue = OperationQueue()
    private var deviceMotionWriter: CSVWriter?
    private var magnetometerWriter: CSVWriter?
    private var barometerWriter: CSVWriter?
    private let utcOffset = Timebase.utcMinusSensorOffsetSec
    private var minBarometerInterval = 1.0
    private var lastBarometerSensorSec = 0.0

    init() {
        queue.name = "com.grape.lifelog.motion"
        queue.maxConcurrentOperationCount = 1
    }

    func start(session: RecordingSession, settings: LoggerSettings) {
        if settings.deviceMotion.enabled {
            deviceMotionWriter = CSVWriter(
                url: session.motionURL.appendingPathComponent("device_motion.csv"),
                header: "sensor_sec,utc_sec,qx,qy,qz,qw,roll,pitch,yaw,gravity_x,gravity_y,gravity_z,user_acc_x,user_acc_y,user_acc_z,rot_x,rot_y,rot_z"
            )
            startDeviceMotion(hz: settings.deviceMotion.hz)
        }

        if settings.magnetometer.enabled {
            magnetometerWriter = CSVWriter(
                url: session.environmentURL.appendingPathComponent("magnetometer.csv"),
                header: "sensor_sec,utc_sec,mx_uT,my_uT,mz_uT"
            )
            startMagnetometer(hz: settings.magnetometer.hz)
        }

        if settings.barometer.enabled {
            barometerWriter = CSVWriter(
                url: session.environmentURL.appendingPathComponent("barometer.csv"),
                header: "sensor_sec,utc_sec,pressure_kpa,relative_altitude_m"
            )
            startBarometer(hz: settings.barometer.hz)
        }
    }

    func stop(completion: (() -> Void)? = nil) {
        motionManager.stopDeviceMotionUpdates()
        motionManager.stopMagnetometerUpdates()
        altimeter.stopRelativeAltitudeUpdates()
        queue.addOperation {
            self.deviceMotionWriter?.close()
            self.magnetometerWriter?.close()
            self.barometerWriter?.close()
            self.deviceMotionWriter = nil
            self.magnetometerWriter = nil
            self.barometerWriter = nil
            completion?()
        }
    }

    private func startDeviceMotion(hz: Double) {
        guard motionManager.isDeviceMotionAvailable else {
            deviceMotionWriter?.writeLine("# unavailable")
            return
        }
        motionManager.deviceMotionUpdateInterval = 1.0 / hz
        motionManager.startDeviceMotionUpdates(using: .xArbitraryCorrectedZVertical, to: queue) { [weak self] data, error in
            guard let self, let data else {
                if let error { self?.deviceMotionWriter?.writeLine("# error,\(error.localizedDescription)") }
                return
            }
            let sensorSec = data.timestamp
            let utcSec = sensorSec + self.utcOffset
            let q = data.attitude.quaternion
            self.deviceMotionWriter?.writeLine(String(
                format: "%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f",
                sensorSec,
                utcSec,
                q.x,
                q.y,
                q.z,
                q.w,
                data.attitude.roll,
                data.attitude.pitch,
                data.attitude.yaw,
                data.gravity.x,
                data.gravity.y,
                data.gravity.z,
                data.userAcceleration.x,
                data.userAcceleration.y,
                data.userAcceleration.z,
                data.rotationRate.x,
                data.rotationRate.y,
                data.rotationRate.z
            ))
        }
    }

    private func startMagnetometer(hz: Double) {
        guard motionManager.isMagnetometerAvailable else {
            magnetometerWriter?.writeLine("# unavailable")
            return
        }
        motionManager.magnetometerUpdateInterval = 1.0 / hz
        motionManager.startMagnetometerUpdates(to: queue) { [weak self] data, error in
            guard let self, let data else {
                if let error { self?.magnetometerWriter?.writeLine("# error,\(error.localizedDescription)") }
                return
            }
            let sensorSec = data.timestamp
            let utcSec = sensorSec + self.utcOffset
            self.magnetometerWriter?.writeLine(String(
                format: "%.6f,%.6f,%.9f,%.9f,%.9f",
                sensorSec,
                utcSec,
                data.magneticField.x,
                data.magneticField.y,
                data.magneticField.z
            ))
        }
    }

    private func startBarometer(hz: Double) {
        guard CMAltimeter.isRelativeAltitudeAvailable() else {
            barometerWriter?.writeLine("# unavailable")
            return
        }
        minBarometerInterval = 1.0 / max(0.05, hz)
        lastBarometerSensorSec = 0
        altimeter.startRelativeAltitudeUpdates(to: queue) { [weak self] data, error in
            guard let self, let data else {
                if let error { self?.barometerWriter?.writeLine("# error,\(error.localizedDescription)") }
                return
            }
            let sensorSec = data.timestamp
            guard sensorSec - self.lastBarometerSensorSec >= self.minBarometerInterval else { return }
            self.lastBarometerSensorSec = sensorSec
            let utcSec = sensorSec + self.utcOffset
            self.barometerWriter?.writeLine(String(
                format: "%.6f,%.6f,%.9f,%.9f",
                sensorSec,
                utcSec,
                data.pressure.doubleValue,
                data.relativeAltitude.doubleValue
            ))
        }
    }
}

final class LocationLogger: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private var writer: CSVWriter?
    private var minInterval = 5.0
    private var lastWriteUTCSec = 0.0
    private let utcOffset = Timebase.utcMinusSensorOffsetSec

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.distanceFilter = kCLDistanceFilterNone
        manager.pausesLocationUpdatesAutomatically = false
    }

    func requestPermission() {
        if manager.authorizationStatus == .notDetermined {
            manager.requestWhenInUseAuthorization()
        }
    }

    func start(session: RecordingSession, hz: Double) {
        minInterval = 1.0 / max(0.05, hz)
        lastWriteUTCSec = 0
        writer = CSVWriter(
            url: session.locationURL.appendingPathComponent("geo_location.csv"),
            header: "sensor_sec,utc_sec,latitude,longitude,altitude,horizontal_accuracy,vertical_accuracy,speed,course"
        )
        manager.startUpdatingLocation()
    }

    func stop() {
        manager.stopUpdatingLocation()
        writer?.close()
        writer = nil
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        let utcSec = location.timestamp.timeIntervalSince1970
        guard utcSec - lastWriteUTCSec >= minInterval else { return }
        lastWriteUTCSec = utcSec
        let sensorSec = utcSec - utcOffset
        writer?.writeLine(String(
            format: "%.6f,%.6f,%.9f,%.9f,%.9f,%.3f,%.3f,%.6f,%.6f",
            sensorSec,
            utcSec,
            location.coordinate.latitude,
            location.coordinate.longitude,
            location.altitude,
            location.horizontalAccuracy,
            location.verticalAccuracy,
            location.speed,
            location.course
        ))
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        writer?.writeLine("# error,\(error.localizedDescription)")
    }
}
