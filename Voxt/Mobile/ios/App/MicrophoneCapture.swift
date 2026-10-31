import AVFoundation
import VoxtMobileKit

/// Keeps one AVAudioEngine input tap running for the whole session. iOS lets a background
/// app keep using an audio session that was started in the foreground, which is what lets
/// the keyboard start dictations without bringing this app forward each time.
@MainActor
final class MicrophoneCapture: AudioCapturing {
    enum CaptureError: LocalizedError {
        case microphoneDenied
        case noInput

        var errorDescription: String? {
            switch self {
            case .microphoneDenied: "Microphone access is off. Turn it on in Settings > Voxt."
            case .noInput: "No microphone is available."
            }
        }
    }

    let engine = AVAudioEngine()
    private let lock = NSLock()
    private var observers: [NSObjectProtocol] = []
    /// True between startSession and endSession.
    var isSessionOpen = false
    /// Rebuilds the input tap and starts the engine again; replaceable in tests.
    lazy var restartEngine: () throws -> Void = { [unowned self] in try self.startEngine() }
    private nonisolated(unsafe) var isCapturing = false
    private nonisolated(unsafe) var samples: [Float] = []
    private nonisolated(unsafe) var downsampler: PCMDownsampler?
    private nonisolated(unsafe) var measuredLevel: Float = 0
    var level: Float { lock.withLock { measuredLevel } }

    init(notificationCenter: NotificationCenter = .default) {
        // A route or category change (for example the floating window's session) stops the
        // engine without an error, and the tap then delivers nothing.
        observers.append(notificationCenter.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: nil, queue: nil
        ) { [weak self] note in
            let fromOurEngine = note.object as AnyObject? === self?.engine
            guard fromOurEngine else { return }
            Self.onMain { self?.restartIfOpen() }
        })
        observers.append(notificationCenter.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: nil
        ) { [weak self] note in
            let type = (note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt).flatMap(AVAudioSession.InterruptionType.init)
            guard type == .ended else { return }
            Self.onMain { self?.restartIfOpen() }
        })
    }

    /// The engine posts its change notification on an internal queue.
    private nonisolated static func onMain(_ work: @escaping @MainActor @Sendable () -> Void) {
        if Thread.isMainThread { MainActor.assumeIsolated(work) } else { Task { @MainActor in work() } }
    }

    private func restartIfOpen() {
        guard isSessionOpen else { return }
        try? restartEngine()
    }

    func startSession() throws {
        guard AVAudioApplication.shared.recordPermission == .granted else { throw CaptureError.microphoneDenied }
        try Self.configure(AVAudioSession.sharedInstance())
        try startEngine()
        isSessionOpen = true
    }

    private func startEngine() throws {
        engine.stop()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw CaptureError.noInput }
        let downsampler = try PCMDownsampler(inputFormat: format)
        lock.withLock { self.downsampler = downsampler }

        input.removeTap(onBus: 0)
        installInputTap(on: input, format: format)
        engine.prepare()
        try engine.start()
    }

    static func configure(_ session: AudioSessionConfiguring) throws {
        try session.setCategory(.playAndRecord, mode: .default, options: [.mixWithOthers, .defaultToSpeaker, .allowBluetoothHFP])
        // Without this, an active recording session mutes system haptics device-wide,
        // including every keyboard's key feedback (dictus, MIT, UnifiedAudioEngine.swift:156-166).
        try session.setAllowHapticsAndSystemSoundsDuringRecording(true)
        try session.activate()
    }

    // iOS invokes this on its audio queue, not MainActor. Keep the closure Sendable.
    func installInputTap(on input: AVAudioNode, format: AVAudioFormat) {
        input.installTap(onBus: 0, bufferSize: 4_096, format: format) { @Sendable [weak self] buffer, _ in
            self?.append(buffer)
        }
    }

    func endSession() {
        isSessionOpen = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        lock.withLock {
            isCapturing = false
            measuredLevel = 0
            samples = []
        }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    func beginCapture() {
        lock.withLock {
            samples = []
            samples.reserveCapacity(16_000 * 60)
            isCapturing = true
        }
    }

    func endCapture() -> [Float] {
        lock.withLock {
            isCapturing = false
            measuredLevel = 0
            let tail = (try? downsampler?.flush()) ?? []
            let captured = samples + tail
            samples = []
            return captured
        }
    }

    func capturedSamples(from offset: Int) -> [Float] {
        lock.withLock { offset < samples.count ? Array(samples[offset...]) : [] }
    }

    static func requestPermission() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    private nonisolated func append(_ buffer: AVAudioPCMBuffer) {
        lock.withLock {
            guard isCapturing, let downsampler, let converted = try? downsampler.convert(buffer) else { return }
            measuredLevel = AudioLevel.rms(converted)
            samples.append(contentsOf: converted)
        }
    }
}

protocol AudioSessionConfiguring {
    func setCategory(_ category: AVAudioSession.Category, mode: AVAudioSession.Mode, options: AVAudioSession.CategoryOptions) throws
    func setAllowHapticsAndSystemSoundsDuringRecording(_ allow: Bool) throws
    func activate() throws
}

extension AVAudioSession: AudioSessionConfiguring {
    func activate() throws { try setActive(true, options: []) }
}
