import VoxtNewHelperKit
@preconcurrency import AVFoundation
import CoreAudio
import Foundation

// capture.start's echoCancel: Apple's voice processing on the microphone,
// whose reference is the output device's own stream (every app's sound),
// while the Mac plays and records through its built-in speakers and
// microphone. Elsewhere (headphones, Bluetooth, other devices) there is
// no echo, or no telling: the microphone records as without it.
// It runs in a child process, voxtnew-helper --vp-capture: in the helper
// it silences capture.system's tap. It records the default input (told to
// use another device, it starts but delivers nothing), and ends when the
// default input or output changes, for the helper to start another. The
// child writes to stdout the sample rate (Float64), then its first
// channel as Float32 (voice processing's output; the others are the raw
// microphones), until stdin closes.

/// voxtnew-helper --vp-capture: the default input through voice
/// processing, to stdout.
func runVoiceProcessingChild() -> Never {
    // A write to a pipe the helper closed fails (and ends the child)
    // instead of a signal. Exits are _exit: Core Audio's threads still run.
    signal(SIGPIPE, SIG_IGN)
    // The helper ignores these for its own handlers (Output.swift), and a
    // child inherits that: without the defaults back, its terminate() does
    // nothing to a child stuck starting voice processing.
    for sig in [SIGTERM, SIGINT, SIGHUP] {
        signal(sig, SIG_DFL)
    }
    // The helper closes stdin (or dies): end, also while voice processing starts.
    Thread {
        while !FileHandle.standardInput.availableData.isEmpty {}
        _exit(0)
    }.start()
    let engine = AVAudioEngine()
    let err = FileHandle.standardError
    let input = engine.inputNode
    do {
        try input.setVoiceProcessingEnabled(true)
        // Other apps ducked as little as macOS allows: the meeting stays
        // audible. The engine's output stays out of the graph: a main
        // mixer connected to it fails the start (-10875).
        input.voiceProcessingOtherAudioDuckingConfiguration =
            AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
        engine.prepare()
        try engine.start()
    } catch {
        err.write(Data("[vp-capture] voice processing did not start: \(error)\n".utf8))
        _exit(1)
    }
    let format = input.outputFormat(forBus: 0)
    guard format.sampleRate > 0, format.channelCount > 0 else {
        err.write(Data("[vp-capture] the input has no usable format\n".utf8))
        _exit(1)
    }
    var rate = format.sampleRate
    let out = FileHandle.standardOutput
    let send = { (chunk: Data) in
        do { try out.write(contentsOf: chunk) } catch { _exit(0) } // the helper stopped reading
    }
    send(Data(bytes: &rate, count: MemoryLayout<Double>.size))
    input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
        guard let data = buffer.floatChannelData else { return }
        let n = Int(buffer.frameLength)
        if buffer.format.isInterleaved {
            let stride = Int(buffer.format.channelCount)
            let first = (0 ..< n).map { data[0][$0 * stride] }
            first.withUnsafeBufferPointer { send(Data(buffer: $0)) }
        } else {
            send(Data(bytes: data[0], count: n * MemoryLayout<Float>.size))
        }
    }
    // A new default input or output (headphones plugged in), or an engine
    // macOS stopped: end, and the helper starts a child on the new ones
    // (CaptureRun's restart). A change that leaves both is the engine's own,
    // unless the output is no longer the speakers: on some Macs the headphone
    // jack is the same device, another data source.
    let devices = (input: inputDeviceIDs().defaultID, output: defaultOutput()?.id)
    NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { n in
        let running = (n.object as? AVAudioEngine)?.isRunning ?? false
        if running && inputDeviceIDs().defaultID == devices.input && defaultOutput()?.id == devices.output && onBuiltInSpeakersAndMic() { return }
        FileHandle.standardError.write(Data("[vp-capture] the audio devices changed: ending\n".utf8))
        _exit(0)
    }
    err.write(Data("[vp-capture] running, \(Int(format.sampleRate)) Hz, \(format.channelCount) channels\n".utf8))
    RunLoop.main.run()
    _exit(0)
}

/// The default output is the built-in speakers and the default input the
/// built-in microphone (data sources ispk and imic: a built-in headphone
/// output is hdpn).
func onBuiltInSpeakersAndMic() -> Bool {
    guard let output = defaultOutput()?.id else { return false }
    return builtIn(output, kAudioDevicePropertyScopeOutput, source: 0x6973_706B) // 'ispk'
        && builtIn(inputDeviceIDs().defaultID, kAudioDevicePropertyScopeInput, source: 0x696D_6963) // 'imic'
}

private func builtIn(_ device: AudioObjectID, _ scope: AudioObjectPropertyScope, source: UInt32) -> Bool {
    var transport: UInt32 = 0, dataSource: UInt32 = 0
    return device != 0 && property(device, kAudioDevicePropertyTransportType, &transport) && transport == kAudioDeviceTransportTypeBuiltIn
        && property(device, kAudioDevicePropertyDataSource, scope: scope, &dataSource) && dataSource == source
}

/// echoCancel's source for a CaptureRun: the default input it records,
/// the running child, and its resampler to rate.
func startVoiceProcessing(rate: Int) throws -> (device: AudioObjectID, child: VPChild, resampler: MonoResampler) {
    let device = inputDeviceIDs().defaultID
    guard device != 0 else { throw RPCError(Code.deviceUnavailable, "no default input") }
    let child: VPChild
    do {
        child = try VPChild()
    } catch {
        throw RPCError(Code.deviceUnavailable, "\(deviceName(device)): voice processing did not start: \(errorMessage(error))")
    }
    guard let res = MonoResampler(inputRate: child.rate, rate: rate) else {
        child.stop()
        throw RPCError(Code.deviceUnavailable, "cannot convert \(Int(child.rate)) Hz")
    }
    return (device, child, res)
}

/// The helper's end of --vp-capture.
final class VPChild: @unchecked Sendable {
    let rate: Double
    private let process: Process
    private let input: Pipe
    private let output: Pipe

    /// How long voice processing may take to start: the wait holds the
    /// capture queue, which stop and cancel wait on.
    static let startLimit: DispatchTimeInterval = .seconds(5)

    /// Starts the child and waits for its sample rate, which it writes
    /// once voice processing runs; a child not running by startLimit is
    /// ended, which ends the wait.
    init() throws {
        let process = Process(), input = Pipe(), output = Pipe()
        process.executableURL = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
        process.arguments = ["--vp-capture"]
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        try process.run()
        let limit = DispatchWorkItem {
            guard process.isRunning else { return }
            try? input.fileHandleForWriting.close() // it also ends on stdin closing
            process.terminate()
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + VPChild.startLimit, execute: limit)
        let header = output.fileHandleForReading.readData(ofLength: MemoryLayout<Double>.size)
        limit.cancel()
        guard header.count == MemoryLayout<Double>.size else {
            process.terminate()
            throw RPCError(Code.deviceUnavailable, "the voice processing process ended, or did not start within 5 s")
        }
        self.process = process
        self.input = input
        self.output = output
        rate = header.withUnsafeBytes { $0.loadUnaligned(as: Double.self) }
    }

    /// Reads the samples on a thread of its own; ended runs once the child
    /// stops sending (stopped, ended on a device change, or died).
    func start(_ deliver: @escaping @Sendable (AVAudioPCMBuffer) -> Void, ended: @escaping @Sendable () -> Void) {
        let handle = output.fileHandleForReading
        let rate = rate
        Thread {
            defer { ended() }
            guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false) else { return }
            var stream = FloatStream()
            while let data = try? handle.read(upToCount: 4096), !data.isEmpty {
                let samples = stream.push(data)
                guard !samples.isEmpty, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
                      let dst = buffer.floatChannelData?[0] else { continue }
                buffer.frameLength = AVAudioFrameCount(samples.count)
                samples.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: samples.count) }
                deliver(buffer)
            }
        }.start()
    }

    func stop() {
        try? input.fileHandleForWriting.close() // the child also ends when terminated
        process.terminate()
    }
}
