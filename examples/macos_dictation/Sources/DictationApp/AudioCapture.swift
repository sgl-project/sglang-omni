#if canImport(DictationCore)
import DictationCore
#endif
import AVFoundation
import Foundation

enum AudioEncoder {
    static func wav(samples: [Float], sampleRate: Double) throws -> Data {
        guard !samples.isEmpty, sampleRate > 0,
              let inputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let outputFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true),
              let input = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(samples.count)),
              let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw DictationError("音频格式无效，无法转换为 WAV。")
        }
        input.frameLength = input.frameCapacity
        samples.withUnsafeBufferPointer { source in
            input.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
        }
        let capacity = AVAudioFrameCount(ceil(Double(samples.count) * 16000 / sampleRate) + 1024)
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            throw DictationError("无法分配音频转换缓冲区。")
        }
        var supplied = false
        var pcm = Data()
        while true {
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, inputStatus in
                if supplied { inputStatus.pointee = .endOfStream; return nil }
                supplied = true
                inputStatus.pointee = .haveData
                return input
            }
            if let error { throw DictationError("音频转换失败：\(error.localizedDescription)") }
            if output.frameLength > 0 {
                pcm.append(Data(bytes: output.int16ChannelData![0], count: Int(output.frameLength) * 2))
            }
            if status == .endOfStream { break }
            guard status == .haveData, output.frameLength > 0 else {
                throw DictationError("音频转换未完整结束。")
            }
        }
        guard !pcm.isEmpty else { throw DictationError("音频为空。") }
        var wav = Data("RIFF".utf8)
        func append<T: FixedWidthInteger>(_ value: T) {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { wav.append(contentsOf: $0) }
        }
        append(UInt32(36 + pcm.count))
        wav.append(Data("WAVEfmt ".utf8))
        append(UInt32(16)); append(UInt16(1)); append(UInt16(1))
        append(UInt32(16000)); append(UInt32(32000)); append(UInt16(2)); append(UInt16(16))
        wav.append(Data("data".utf8))
        append(UInt32(pcm.count))
        wav.append(pcm)
        return wav
    }

    static func load(_ url: URL) throws -> Data {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = file.processingFormat
        guard format.sampleRate > 0, format.channelCount > 0, file.length > 0 else {
            throw DictationError("文件中没有可读取的音频。")
        }
        guard Double(file.length) / format.sampleRate <= DictationSession.maximumAudioSeconds else {
            throw DictationError("录音文件不能超过 \(Int(DictationSession.maximumAudioSeconds)) 秒，请先裁剪后再导入。")
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw DictationError("无法读取录音文件。")
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else { throw DictationError("录音文件格式不支持。") }
        var mono = [Float](repeating: 0, count: Int(buffer.frameLength))
        for channel in 0..<Int(format.channelCount) {
            for index in mono.indices { mono[index] += channels[channel][index] / Float(format.channelCount) }
        }
        return try wav(samples: mono, sampleRate: format.sampleRate)
    }
}

// The tap runs on an audio thread. Its bounded buffer never writes audio to disk.
final class CaptureBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private let rate: Double
    private let limit: Int
    private var samples: [Float] = []
    private var active = true
    private var meter: Double = 0
    private var failure: String?

    init(rate: Double) {
        self.rate = rate
        limit = Int(rate * DictationSession.maximumAudioSeconds)
        samples.reserveCapacity(limit)
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return }
        guard buffer.format.sampleRate == rate, buffer.format.channelCount > 0,
              let channels = buffer.floatChannelData else {
            failure = "麦克风格式发生变化，请重新录音。"
            return
        }
        var peak: Float = 0
        let channelCount = Int(buffer.format.channelCount)
        for index in 0..<min(Int(buffer.frameLength), limit - samples.count) {
            var sample: Float = 0
            for channel in 0..<channelCount { sample += channels[channel][index] / Float(channelCount) }
            sample = sample.isFinite ? min(1, max(-1, sample)) : 0
            samples.append(sample)
            peak = max(peak, abs(sample))
        }
        meter = min(1, Double(peak) * 5)
    }

    func markFailure() {
        lock.lock()
        failure = "麦克风设备发生变化，请检查输入设备后重新录音。"
        lock.unlock()
    }

    var status: (Double, String?) {
        lock.lock()
        defer { lock.unlock() }
        return (meter, failure)
    }

    func finish() -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        active = false
        let result = samples
        samples = []
        meter = 0
        return result
    }
}

@MainActor
final class MicrophoneRecorder: AudioRecording {
    private var engine: AVAudioEngine?
    private var captured: CaptureBuffer?
    private var rate: Double = 0
    private var generation = UUID()
    private var observer: NSObjectProtocol?
    var level: Double { captured?.status.0 ?? 0 }
    var recordingError: String? { captured?.status.1 }

    func start() async throws {
        cancel()
        let id = generation
        let allowed = await AVCaptureDevice.requestAccess(for: .audio)
        try Task.checkCancellation()
        guard id == generation else { throw CancellationError() }
        guard allowed else { throw DictationError("请在系统设置 → 隐私与安全性 → 麦克风中允许 Omni 听写。") }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0,
              format.commonFormat == .pcmFormatFloat32, !format.isInterleaved else {
            throw DictationError("麦克风不可用，请在系统设置中选择有效的声音输入设备。")
        }
        rate = format.sampleRate
        let captured = CaptureBuffer(rate: rate)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in captured.append(buffer) }
        self.engine = engine
        self.captured = captured
        do {
            engine.prepare()
            try engine.start()
            observer = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { _ in
                captured.markFailure()
            }
        } catch {
            cancel()
            throw DictationError("无法启动麦克风：\(error.localizedDescription)")
        }
    }

    func stop() throws -> Data {
        guard let captured else { throw DictationError("没有正在进行的录音。") }
        let samples = captured.finish()
        let sampleRate = rate
        cancel()
        guard samples.count >= Int(sampleRate * 0.15), samples.contains(where: { abs($0) > 0.00001 }) else {
            throw DictationError("没有录到有效声音，请检查麦克风和输入音量后重试。")
        }
        return try AudioEncoder.wav(samples: samples, sampleRate: sampleRate)
    }

    func cancel() {
        generation = UUID()
        _ = captured?.finish()
        if let observer { NotificationCenter.default.removeObserver(observer) }
        observer = nil
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        captured = nil
    }
}
