@preconcurrency import AVFoundation
import Foundation

/// Records the microphone for a session.
public protocol AudioRecorder: AnyObject, Sendable {
    /// Starts recording; onLevel gets the input's level (0...1) about
    /// twenty times a second, on any thread. Opening the audio session
    /// can block for long (Bluetooth inputs), so it is async.
    func start(onLevel: @escaping @Sendable (Float) -> Void) async throws
    /// Stops and returns what was recorded; the audio closes after.
    func stop() -> Recording
    /// Stops and drops what was recorded.
    func cancel()
}

public enum RecorderError: Error, Equatable {
    case microphoneDenied
    case noInput
    case converter
}

/// The microphone through AVAudioEngine, converted to 16 kHz mono as it
/// comes in. The engine and the audio session are only touched on a
/// serial queue, off the main thread (iOS warns that activating a
/// session there can freeze the UI); the samples are under the lock.
public final class MicRecorder: AudioRecorder, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let queue = DispatchQueue(label: "voxt.recorder", qos: .userInitiated)
    private let lock = NSLock()
    private var samples: [Float] = []
    private var running = false
    /// On the queue: the tap is installed and the session active.
    private var isOpen = false

    public init() {}

    /// Asks for the microphone; true when allowed.
    public static func requestPermission() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    public func start(onLevel: @escaping @Sendable (Float) -> Void) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                continuation.resume(with: Result { try self.open(onLevel: onLevel) })
            }
        }
    }

    /// On the queue: opens the session and the tap; a failure part way
    /// closes what was opened.
    private func open(onLevel: @escaping @Sendable (Float) -> Void) throws {
        close()
        isOpen = true
        do {
            try begin(onLevel: onLevel)
        } catch {
            close()
            throw error
        }
    }

    private func begin(onLevel: @escaping @Sendable (Float) -> Void) throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        guard AVAudioApplication.shared.recordPermission == .granted else {
            throw RecorderError.microphoneDenied
        }
        #endif
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw RecorderError.noInput
        }
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(Recording.sampleRate), channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: format, to: target) else {
            throw RecorderError.converter
        }
        lock.lock()
        samples = []
        samples.reserveCapacity(Recording.sampleRate * 60)
        running = true
        lock.unlock()
        let ratio = target.sampleRate / format.sampleRate
        var out: AVAudioPCMBuffer? // reused while it is large enough (the tap's buffers are alike)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
            if out.map({ $0.frameCapacity < capacity }) ?? true {
                out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity)
            }
            guard let out else { return }
            out.frameLength = 0
            var fed = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if fed {
                    status.pointee = .noDataNow
                    return nil
                }
                fed = true
                status.pointee = .haveData
                return buffer
            }
            guard error == nil, let channel = out.floatChannelData?[0] else { return }
            let chunk = UnsafeBufferPointer(start: channel, count: Int(out.frameLength))
            self.append(chunk)
            onLevel(Self.level(chunk))
        }
        engine.prepare()
        try engine.start()
    }

    public func stop() -> Recording {
        lock.lock()
        running = false
        let recording = Recording(samples: samples)
        samples = []
        lock.unlock()
        queue.async { self.close() }
        return recording
    }

    public func cancel() {
        _ = stop()
    }

    private func append(_ chunk: UnsafeBufferPointer<Float>) {
        lock.lock()
        if running {
            samples.append(contentsOf: chunk)
        }
        lock.unlock()
    }

    /// On the queue: removes the tap, stops the engine and gives the audio
    /// back to other apps.
    private func close() {
        lock.lock()
        running = false
        lock.unlock()
        guard isOpen else { return }
        isOpen = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }

    /// A chunk's level for the meter: its RMS on a -50...0 dB scale.
    static func level<C: Collection<Float>>(_ chunk: C) -> Float {
        guard !chunk.isEmpty else { return 0 }
        var sum: Float = 0
        for s in chunk {
            sum += s * s
        }
        let rms = (sum / Float(chunk.count)).squareRoot()
        let db = 20 * log10(max(rms, 1e-6))
        return max(0, min(1, (db + 50) / 50))
    }
}
