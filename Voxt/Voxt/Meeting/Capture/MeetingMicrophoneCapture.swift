// MeetingMicrophoneCapture.swift
// Provides Meeting Microphone Capture for meeting capture.

import Foundation
import AVFoundation
import CoreAudio

final class MeetingMicrophoneCapture: @unchecked Sendable {
    enum CaptureError: LocalizedError {
        case inputUnavailable
        case engineStartFailed(Error)

        var errorDescription: String? {
            switch self {
            case .inputUnavailable:
                return "Microphone input is unavailable."
            case .engineStartFailed(let error):
                return "Microphone capture failed to start: \(error.localizedDescription)"
            }
        }
    }

    private var audioEngine: AVAudioEngine?
    private var hasTapInstalled = false
    private var preferredInputDeviceID: AudioDeviceID?
    private var hasLoggedFirstCallback = false

    deinit {
        stop()
    }

    func setPreferredInputDevice(_ deviceID: AudioDeviceID?) {
        preferredInputDeviceID = deviceID
    }

    func start(onBuffer: @escaping (AVAudioPCMBuffer, Float) -> Void) throws {
        stop()
        do {
            try startCaptureEngine(usePreferredInputDevice: true, onBuffer: onBuffer)
        } catch {
            let preferredDeviceID = preferredInputDeviceID
            let shouldRetryWithSystemDefault =
                preferredDeviceID != nil
                && preferredDeviceID != AudioDeviceID(kAudioObjectUnknown)
            guard shouldRetryWithSystemDefault else { throw error }

            VoxtLog.meetingWarning(
                "Meeting microphone preferred-input start failed; retrying with system default. deviceID=\(preferredDeviceID.map(String.init(describing:)) ?? "nil"), error=\(error.localizedDescription)"
            )
            stop()
            try startCaptureEngine(usePreferredInputDevice: false, onBuffer: onBuffer)
        }
    }

    func stop() {
        guard let audioEngine = self.audioEngine else { return }
        if hasTapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            hasTapInstalled = false
        }
        audioEngine.pause()
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.reset()
        self.audioEngine = nil
        hasLoggedFirstCallback = false
        VoxtLog.meeting("Meeting microphone capture stopped.", verbose: true)
    }

    private func startCaptureEngine(
        usePreferredInputDevice: Bool,
        onBuffer: @escaping (AVAudioPCMBuffer, Float) -> Void
    ) throws {
        let audioEngine = AVAudioEngine()
        self.audioEngine = audioEngine
        hasLoggedFirstCallback = false

        do {
            let inputNode = audioEngine.inputNode
            let didApplyPreferredInputDevice = usePreferredInputDevice
                ? applyPreferredInputDeviceIfNeeded(inputNode: inputNode)
                : false
            let activeInputDeviceID = didApplyPreferredInputDevice
                ? preferredInputDeviceID
                : AudioInputDeviceManager.defaultInputDeviceID()
            let nodeOutputFormat = inputNode.outputFormat(forBus: 0)
            guard nodeOutputFormat.sampleRate > 0, nodeOutputFormat.channelCount > 0 else {
                throw CaptureError.inputUnavailable
            }

            let hardwareSampleRate = AudioInputDeviceManager.nominalSampleRate(for: activeInputDeviceID)
            let tapFormat = AudioInputDeviceManager.captureTapFormat(
                nodeOutputFormat: nodeOutputFormat,
                hardwareSampleRate: hardwareSampleRate
            )

            if abs(tapFormat.sampleRate - nodeOutputFormat.sampleRate) > 1 {
                VoxtLog.meetingWarning(
                    "Meeting microphone adjusted input tap format. deviceID=\(activeInputDeviceID.map(String.init(describing:)) ?? "default"), hardwareSampleRate=\(hardwareSampleRate.map { String(Int($0.rounded())) } ?? "unknown"), nodeSampleRate=\(Int(nodeOutputFormat.sampleRate.rounded())), tapSampleRate=\(Int(tapFormat.sampleRate.rounded()))"
                )
            }

            inputNode.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buffer, _ in
                guard let copiedBuffer = Self.copyPCMBuffer(buffer) else { return }
                let level = Self.normalizedRMS(from: copiedBuffer)
                if self?.hasLoggedFirstCallback == false {
                    self?.hasLoggedFirstCallback = true
                    VoxtLog.meeting(
                        "Meeting microphone callback received. sampleRate=\(Int(copiedBuffer.format.sampleRate)), channels=\(copiedBuffer.format.channelCount), frames=\(copiedBuffer.frameLength)",
                        verbose: true
                    )
                }
                onBuffer(copiedBuffer, level)
            }
            hasTapInstalled = true

            audioEngine.prepare()
            do {
                try audioEngine.start()
            } catch {
                throw CaptureError.engineStartFailed(error)
            }

            VoxtLog.meeting(
                "Meeting microphone capture started. sampleRate=\(Int(tapFormat.sampleRate)), channels=\(tapFormat.channelCount), routing=\(didApplyPreferredInputDevice ? "preferred" : "system-default"), deviceID=\(activeInputDeviceID.map(String.init(describing:)) ?? "default")",
                verbose: true
            )
        } catch {
            stop()
            throw error
        }
    }

    @discardableResult
    private func applyPreferredInputDeviceIfNeeded(inputNode: AVAudioInputNode) -> Bool {
        guard let preferredInputDeviceID,
              preferredInputDeviceID != AudioDeviceID(kAudioObjectUnknown),
              AudioInputDeviceManager.isAvailableInputDevice(preferredInputDeviceID),
              let audioUnit = inputNode.audioUnit
        else {
            return false
        }

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
            VoxtLog.meetingWarning(
                "Meeting microphone capture could not switch input device. status=\(status), deviceID=\(preferredInputDeviceID)"
            )
            return false
        }
        return true
    }

    private static func normalizedRMS(from buffer: AVAudioPCMBuffer) -> Float {
        AudioLevelMeter.normalizedLevel(from: buffer)
    }

    private static func copyPCMBuffer(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let copy = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: buffer.frameLength) else {
            return nil
        }
        copy.frameLength = buffer.frameLength

        let sourceBuffers = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)
        let destinationBuffers = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        guard sourceBuffers.count == destinationBuffers.count else { return nil }

        for index in 0..<sourceBuffers.count {
            let source = sourceBuffers[index]
            let destination = destinationBuffers[index]
            let copySize = min(Int(source.mDataByteSize), Int(destination.mDataByteSize))
            guard copySize > 0,
                  let sourceData = source.mData,
                  let destinationData = destination.mData
            else {
                continue
            }
            memcpy(destinationData, sourceData, copySize)
            destinationBuffers[index].mDataByteSize = UInt32(copySize)
        }
        return copy
    }
}
