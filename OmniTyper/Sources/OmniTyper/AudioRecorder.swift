// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import AudioToolbox
import Combine
import CoreAudio

struct MicrophoneDevice: Identifiable, Hashable {
    let id: String
    let name: String
}

// Note (Codex): The audio callback and teardown share a lock so conversion cannot race file closure.
final class AudioCaptureSink: @unchecked Sendable {
    private let lock = NSLock()
    private let converter: AVAudioConverter
    private let format: AVAudioFormat
    private let onPCM: (@Sendable (Data) -> Void)?
    private var file: AVAudioFile?
    private var failure: Error?
    private var meter = 0.0
    private var framesWritten: AVAudioFrameCount = 0
    private let maximumFrames: AVAudioFrameCount = 300 * 16_000

    init(input: AVAudioFormat, url: URL, onPCM: (@Sendable (Data) -> Void)? = nil) throws {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: 16_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: input, to: format) else {
            throw Failure("sys.audioFormat")
        }
        self.format = format
        self.converter = converter
        self.onPCM = onPCM
        file = try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
    }

    func consume(_ input: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard let file, failure == nil, framesWritten < maximumFrames else { return }
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * format.sampleRate / input.format.sampleRate)) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
        var supplied = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return input
        }
        if let conversionError {
            failure = conversionError
            return
        }
        output.frameLength = min(output.frameLength, maximumFrames - framesWritten)
        guard output.frameLength > 0 else { return }
        if let samples = output.floatChannelData?[0] {
            var squares: Double = 0
            for index in 0..<Int(output.frameLength) {
                squares += Double(samples[index]) * Double(samples[index])
            }
            meter = min(1, sqrt(squares / Double(output.frameLength)) * 5)
        }
        do {
            try file.write(from: output)
            framesWritten += output.frameLength
            if let onPCM, let samples = output.floatChannelData?[0] {
                let pcm = (0..<Int(output.frameLength)).map { index -> Int16 in
                    let sample = samples[index].isFinite ? samples[index] : 0
                    return Int16(min(32767, max(-32768, sample * 32768))).littleEndian
                }
                pcm.withUnsafeBytes { onPCM(Data($0)) }
            }
        }
        catch { failure = error }
    }

    func level() -> Double {
        lock.lock()
        defer { lock.unlock() }
        return meter
    }

    func fail(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if failure == nil { failure = error }
    }

    func close() throws {
        lock.lock()
        defer { lock.unlock() }
        file = nil
        if let failure { throw failure }
    }
}

@MainActor
final class AudioRecorder: ObservableObject {
    @Published private(set) var level = 0.0
    @Published private(set) var elapsed = 0.0

    private var engine: AVAudioEngine?
    private var sink: AudioCaptureSink?
    private var outputURL: URL?
    private var meterTask: Task<Void, Never>?
    private var configurationObserver: NSObjectProtocol?
    private var generation = UUID()
    private var isStarting = false

    deinit {
        meterTask?.cancel()
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        try? sink?.close()
        if let outputURL { try? FileManager.default.removeItem(at: outputURL) }
    }

    static func devices() -> [MicrophoneDevice] {
        inputDevices().compactMap { device in
            guard let uid = stringProperty(device, selector: kAudioDevicePropertyDeviceUID),
                  let name = stringProperty(device, selector: kAudioObjectPropertyName) else { return nil }
            return MicrophoneDevice(id: uid, name: name)
        }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    func start(deviceUID: String, onPCM: (@Sendable (Data) -> Void)? = nil) async throws {
        guard engine == nil, !isStarting else {
            throw Failure("sys.recording")
        }
        isStarting = true
        let currentGeneration = UUID()
        generation = currentGeneration
        defer { isStarting = false }
        let authorized: Bool
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: authorized = true
        case .notDetermined: authorized = await AVCaptureDevice.requestAccess(for: .audio)
        default: authorized = false
        }
        try Task.checkCancellation()
        guard generation == currentGeneration else { throw CancellationError() }
        guard authorized else {
            throw Failure("sys.micPermission")
        }

        let engine = AVAudioEngine()
        let input = engine.inputNode
        if !deviceUID.isEmpty {
            guard var device = Self.inputDevices().first(where: {
                Self.stringProperty($0, selector: kAudioDevicePropertyDeviceUID) == deviceUID
            }), let unit = input.audioUnit else {
                throw Failure("sys.micGone")
            }
            let result = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice,
                                             kAudioUnitScope_Global, 0, &device,
                                             UInt32(MemoryLayout<AudioDeviceID>.size))
            guard result == noErr else {
                throw Failure("sys.micAudioError", String(result))
            }
        }
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw Failure("sys.micNoInput")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("OmniTyper-\(UUID().uuidString).wav")
        let sink = try AudioCaptureSink(input: format, url: url, onPCM: onPCM)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
            sink.consume(buffer)
        }
        do {
            // Note (Yifei Leng): Core Audio needs about 100 ms to start. On the main actor that froze the popup
            // in the very frames it appears, so only the bookkeeping stays here.
            try await Task.detached(priority: .userInitiated) {
                engine.prepare()
                try engine.start()
            }.value
            guard generation == currentGeneration else { throw CancellationError() }
        } catch {
            input.removeTap(onBus: 0)
            engine.stop()
            try? sink.close()
            try? FileManager.default.removeItem(at: url)
            throw error
        }
        self.engine = engine
        self.sink = sink
        outputURL = url
        level = 0
        elapsed = 0
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { _ in
            sink.fail(Failure("sys.micChanged"))
        }
        let started = ProcessInfo.processInfo.systemUptime
        meterTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 50_000_000)
                guard !Task.isCancelled, let self else { break }
                self.level = sink.level()
                self.elapsed = ProcessInfo.processInfo.systemUptime - started
            }
        }
    }

    func stop() async throws -> URL {
        guard let url = outputURL else {
            throw Failure("sys.noRecording")
        }
        let sink = self.sink
        let engine = self.engine
        // Note (Yifei Leng): Stopping Core Audio and closing the file take about 60 ms, so they leave the
        // main actor too. Clearing the engine first keeps releaseAudio() from stopping it here.
        self.engine = nil
        releaseAudio()
        do {
            try await Task.detached(priority: .userInitiated) {
                engine?.inputNode.removeTap(onBus: 0)
                engine?.stop()
                try sink?.close()
            }.value
        } catch {
            try? FileManager.default.removeItem(at: url)
            throw error
        }
        return url
    }

    func cancel() {
        generation = UUID()
        let url = outputURL
        let sink = self.sink
        releaseAudio()
        try? sink?.close()
        if let url { try? FileManager.default.removeItem(at: url) }
        elapsed = 0
    }

    private func releaseAudio() {
        meterTask?.cancel()
        meterTask = nil
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
        configurationObserver = nil
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        sink = nil
        outputURL = nil
        level = 0
    }

    private static func inputDevices() -> [AudioDeviceID] {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                                mScope: kAudioObjectPropertyScopeGlobal,
                                                mElement: kAudioObjectPropertyElementMain)
        var bytes: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &bytes) == noErr else { return [] }
        var devices = [AudioDeviceID](repeating: 0, count: Int(bytes) / MemoryLayout<AudioDeviceID>.size)
        guard !devices.isEmpty,
              AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &bytes, &devices) == noErr else { return [] }
        return devices.filter { device in
            var inputAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams,
                                                         mScope: kAudioDevicePropertyScopeInput,
                                                         mElement: kAudioObjectPropertyElementMain)
            var size: UInt32 = 0
            return AudioObjectGetPropertyDataSize(device, &inputAddress, 0, nil, &size) == noErr && size > 0
        }
    }

    private static func stringProperty(_ device: AudioDeviceID, selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &value) == noErr else { return nil }
        return value?.takeRetainedValue() as String?
    }
}
