import AVFoundation
import CoreAudio

struct AudioDevice {
    let id: AudioDeviceID
    let uid: String
    let name: String
    let inputChannels: Int
    let outputChannels: Int
    let nominalSampleRate: Double
}

/// note (luojiaxuan): Read-only queries of the Core Audio device list. Nothing here changes a device
/// or the system's default devices.
enum AudioDevices {
    private static func address(_ selector: AudioObjectPropertySelector,
                                _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    }

    static func all() throws -> [AudioDevice] {
        var addr = address(kAudioHardwarePropertyDevices)
        var size: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size)
        guard status == noErr else { throw ReplayError("cannot list audio devices (\(status))") }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids)
        guard status == noErr else { throw ReplayError("cannot list audio devices (\(status))") }
        return ids.map { id in
            AudioDevice(id: id,
                        uid: string(id, kAudioDevicePropertyDeviceUID),
                        name: string(id, kAudioObjectPropertyName),
                        inputChannels: channels(id, kAudioObjectPropertyScopeInput),
                        outputChannels: channels(id, kAudioObjectPropertyScopeOutput),
                        nominalSampleRate: sampleRate(id))
        }
    }

    static func defaultDevice(input: Bool) -> AudioDeviceID {
        var addr = address(input ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice)
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id)
        return id
    }

    /// note (luojiaxuan): The output device whose name is exactly `name`.
    static func output(named name: String) throws -> AudioDevice {
        let matches = try all().filter { $0.name == name && $0.outputChannels > 0 }
        guard let device = matches.first else { throw ReplayError("no output device named \"\(name)\" (see replay-driver list-devices)") }
        guard matches.count == 1 else { throw ReplayError("\(matches.count) output devices are named \"\(name)\"") }
        return device
    }

    private static func string(_ id: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String {
        var addr = address(selector)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &value)
        guard status == noErr, let value else { return "" }
        return value.takeRetainedValue() as String
    }

    private static func channels(_ id: AudioDeviceID, _ scope: AudioObjectPropertyScope) -> Int {
        var addr = address(kAudioDevicePropertyStreamConfiguration, scope)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    private static func sampleRate(_ id: AudioDeviceID) -> Double {
        var addr = address(kAudioDevicePropertyNominalSampleRate)
        var rate = Float64(0)
        var size = UInt32(MemoryLayout<Float64>.size)
        AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &rate)
        return rate
    }
}

/// note (luojiaxuan): A segment's audio: `preSilence` seconds of silence, the wav, then `postSilence`
/// seconds of silence, as two identical channels at the wav's sample rate.
struct PaddedAudio {
    let buffer: AVAudioPCMBuffer
    let wavDuration: Double
    let paddedDuration: Double

    init(wav: URL, preSilence: Double, postSilence: Double) throws {
        let file = try AVAudioFile(forReading: wav)
        let format = file.processingFormat
        guard format.channelCount == 1 || format.channelCount == 2 else {
            throw ReplayError("\(wav.path): \(format.channelCount) channels; expected mono or stereo")
        }
        let frames = AVAudioFrameCount(file.length)
        guard let source = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(frames, 1)) else {
            throw ReplayError("\(wav.path): cannot allocate \(frames) frames")
        }
        try file.read(into: source, frameCount: frames)

        let rate = format.sampleRate
        let pre = Int((preSilence * rate).rounded()), post = Int((postSilence * rate).rounded())
        let count = Int(source.frameLength)
        let total = pre + count + post
        let stereo = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 2)!
        let padded = AVAudioPCMBuffer(pcmFormat: stereo, frameCapacity: AVAudioFrameCount(total))!
        padded.frameLength = AVAudioFrameCount(total)
        for channel in 0..<2 {
            let out = padded.floatChannelData![channel]
            out.initialize(repeating: 0, count: total)
            let input = source.floatChannelData![min(channel, Int(format.channelCount) - 1)]
            (out + pre).update(from: input, count: count)
        }
        buffer = padded
        wavDuration = Double(count) / rate
        paddedDuration = Double(total) / rate
    }

    /// note (luojiaxuan): Writes the padded audio as 16-bit PCM WAV.
    func write(to url: URL) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: buffer.format.sampleRate,
            AVNumberOfChannelsKey: 2,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
    }
}

/// note (luojiaxuan): Plays buffers in real time to one output device. The device is set on this
/// engine's own output unit; the system's default devices are left alone. The
/// engine's input node is never touched, so no microphone access is involved.
final class AudioPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private var connectedFormat: AVAudioFormat?
    let device: AudioDevice

    init(device: AudioDevice) throws {
        self.device = device
        guard let unit = engine.outputNode.audioUnit else { throw ReplayError("the engine has no output unit") }
        var id = device.id
        let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                          &id, UInt32(MemoryLayout<AudioDeviceID>.size))
        guard status == noErr else { throw ReplayError("cannot route the engine to \(device.name) (\(status))") }
        engine.attach(node)
    }

    /// note (luojiaxuan): The device the engine's output unit is bound to.
    func boundDevice() -> AudioDeviceID {
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        if let unit = engine.outputNode.audioUnit {
            AudioUnitGetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &id, &size)
        }
        return id
    }

    /// note (luojiaxuan): Plays `buffer` starting 50 ms from now and returns once the device has played
    /// it back. `start` is the host time scheduled for the first frame and `end` the
    /// moment the played-back callback arrived, both in continuous nanoseconds.
    func play(_ buffer: AVAudioPCMBuffer) throws -> (start: UInt64, end: UInt64) {
        if connectedFormat != buffer.format {
            engine.stop()
            engine.connect(node, to: engine.mainMixerNode, format: buffer.format)
            connectedFormat = buffer.format
            try engine.start()
            node.play()
        }
        let startHost = mach_absolute_time() + MachClock.ticks(nanoseconds: 50_000_000)
        let done = DispatchSemaphore(value: 0)
        node.scheduleBuffer(buffer, at: AVAudioTime(hostTime: startHost), options: [],
                            completionCallbackType: .dataPlayedBack) { _ in done.signal() }
        let seconds = Double(buffer.frameLength) / buffer.format.sampleRate
        guard done.wait(timeout: .now() + seconds + 5) == .success else {
            throw ReplayError("playback did not finish within \(seconds + 5) s")
        }
        return (MachClock.continuousNanoseconds(hostTime: startHost), MachClock.now())
    }

    func stop() {
        node.stop()
        engine.stop()
    }
}
