import Foundation
import AVFoundation
import AudioToolbox

extension RemoteASRTranscriber {
    func inputCaptureTapFormat(
        inputNode: AVAudioInputNode,
        activeInputDeviceID: AudioDeviceID?,
        logContext: String
    ) -> AVAudioFormat {
        let nodeOutputFormat = inputNode.outputFormat(forBus: 0)
        let hardwareSampleRate = AudioInputDeviceManager.nominalSampleRate(for: activeInputDeviceID)
        let tapFormat = AudioInputDeviceManager.captureTapFormat(
            nodeOutputFormat: nodeOutputFormat,
            hardwareSampleRate: hardwareSampleRate
        )

        if abs(tapFormat.sampleRate - nodeOutputFormat.sampleRate) > 1 {
            VoxtLog.warning(
                "\(logContext) adjusted input tap format. deviceID=\(activeInputDeviceID.map(String.init(describing:)) ?? "default"), hardwareSampleRate=\(hardwareSampleRate.map { String(Int($0.rounded())) } ?? "unknown"), nodeSampleRate=\(Int(nodeOutputFormat.sampleRate.rounded())), tapSampleRate=\(Int(tapFormat.sampleRate.rounded()))"
            )
        }

        return tapFormat
    }

    @discardableResult
    func applyPreferredInputDeviceIfNeeded(inputNode: AVAudioInputNode) -> Bool {
        guard let preferredInputDeviceID,
              preferredInputDeviceID != AudioDeviceID(kAudioObjectUnknown),
              AudioInputDeviceManager.isAvailableInputDevice(preferredInputDeviceID)
        else {
            return false
        }
        guard let audioUnit = inputNode.audioUnit else { return false }
        var deviceID = preferredInputDeviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        if status != noErr {
            VoxtLog.asrWarning("Remote ASR failed to switch preferred input device. status=\(status)")
            return false
        }
        return true
    }

    func audioLevelFromPCM16(_ data: Data) -> Float {
        guard data.count >= 2 else { return 0 }
        var sum: Float = 0
        var count: Float = 0
        data.withUnsafeBytes { rawBuffer in
            let samples = rawBuffer.bindMemory(to: Int16.self)
            for sample in samples {
                let normalized = Float(sample) / Float(Int16.max)
                sum += normalized * normalized
                count += 1
            }
        }
        guard count > 0 else { return 0 }
        let rms = sqrt(sum / count)
        return min(max(rms * 2.4, 0), 1)
    }

    nonisolated static func makeDoubaoPCM16MonoData(from buffer: AVAudioPCMBuffer) -> Data? {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return nil }

        let inputRate = max(buffer.format.sampleRate, 1)
        let targetRate = 16000.0
        let step = max(inputRate / targetRate, 1)
        let outputCount = max(Int(Double(frameCount) / step), 1)
        var output = Data(count: outputCount * MemoryLayout<Int16>.size)

        switch buffer.format.commonFormat {
        case .pcmFormatInt16:
            guard let channelData = buffer.int16ChannelData?[0] else { return nil }
            output.withUnsafeMutableBytes { rawBuffer in
                let out = rawBuffer.bindMemory(to: Int16.self)
                for index in 0..<outputCount {
                    let sourceIndex = min(Int(Double(index) * step), frameCount - 1)
                    out[index] = channelData[sourceIndex]
                }
            }
        case .pcmFormatFloat32:
            guard let channelData = buffer.floatChannelData?[0] else { return nil }
            output.withUnsafeMutableBytes { rawBuffer in
                let out = rawBuffer.bindMemory(to: Int16.self)
                for index in 0..<outputCount {
                    let sourceIndex = min(Int(Double(index) * step), frameCount - 1)
                    let clamped = max(-1.0, min(1.0, channelData[sourceIndex]))
                    out[index] = Int16(clamped * Float(Int16.max))
                }
            }
        default:
            return nil
        }

        return output
    }

    nonisolated static func makePCM16MonoData(from samples: [Float], inputSampleRate: Double) -> Data? {
        guard !samples.isEmpty, inputSampleRate > 0 else { return nil }
        let targetRate = 16000.0
        let ratio = targetRate / inputSampleRate
        let outputCount = max(Int(Double(samples.count) * ratio), 1)
        var data = Data(count: outputCount * MemoryLayout<Int16>.size)
        data.withUnsafeMutableBytes { rawBuffer in
            let out = rawBuffer.bindMemory(to: Int16.self)
            for index in 0..<outputCount {
                let sourcePosition = Double(index) / ratio
                let sourceIndex = min(Int(sourcePosition.rounded(.down)), samples.count - 1)
                let clamped = max(-1.0, min(1.0, samples[sourceIndex]))
                out[index] = Int16(clamped * Float(Int16.max))
            }
        }
        return data
    }
}
