import AudioToolbox
import AVFoundation
import Foundation

private enum MLXCaptureStartError: LocalizedError {
    case engineStartTimedOut(Double)

    var errorDescription: String? {
        switch self {
        case .engineStartTimedOut(let seconds):
            return "Audio engine failed to start within \(String(format: "%.0f", seconds))s."
        }
    }
}

/// The recording-start barrier admits one start; cancellation still waits for native return.
private struct MLXAudioEngineBox: @unchecked Sendable {
    nonisolated(unsafe) let engine: AVAudioEngine
}

/// Borrows the transcriber's engine without owning capture tasks, callbacks or session state.
@MainActor
enum MLXAudioCaptureGraph {
    static func configure(
        engine: AVAudioEngine,
        preferredInputDeviceID: AudioDeviceID?,
        usePreferredInputDevice: Bool
    ) -> (format: AVAudioFormat, usedPreferredDevice: Bool) {
        if engine.isRunning {
            engine.stop()
        }
        engine.reset()

        let inputNode = engine.inputNode
        inputNode.removeTap(onBus: 0)

        let didApplyPreferredInputDevice = usePreferredInputDevice
            ? applyPreferredInputDeviceIfNeeded(inputNode: inputNode, preferredInputDeviceID: preferredInputDeviceID)
            : false
        let activeInputDeviceID = didApplyPreferredInputDevice ? preferredInputDeviceID : AudioInputDeviceManager.defaultInputDeviceID()
        let nodeOutputFormat = inputNode.outputFormat(forBus: 0)
        let hardwareSampleRate = AudioInputDeviceManager.nominalSampleRate(for: activeInputDeviceID)
        let recordingFormat = AudioInputDeviceManager.captureTapFormat(
            nodeOutputFormat: nodeOutputFormat,
            hardwareSampleRate: hardwareSampleRate
        )

        if abs(recordingFormat.sampleRate - nodeOutputFormat.sampleRate) > 1 {
            VoxtLog.warning(
                "MLX transcriber adjusted input tap format. deviceID=\(activeInputDeviceID.map(String.init(describing:)) ?? "default"), hardwareSampleRate=\(hardwareSampleRate.map { String(Int($0.rounded())) } ?? "unknown"), nodeSampleRate=\(Int(nodeOutputFormat.sampleRate.rounded())), tapSampleRate=\(Int(recordingFormat.sampleRate.rounded()))"
            )
        }
        return (recordingFormat, didApplyPreferredInputDevice)
    }

    static func logStarted(
        format: AVAudioFormat,
        usedPreferredDevice: Bool,
        preferredInputDeviceID: AudioDeviceID?
    ) {
        VoxtLog.asr(
            "MLX audio capture started. sampleRate=\(Int(format.sampleRate)), channels=\(format.channelCount), format=\(format.commonFormat.rawValue), interleaved=\(format.isInterleaved), routing=\(usedPreferredDevice ? "preferred" : "system-default"), deviceID=\(usedPreferredDevice ? (preferredInputDeviceID.map(String.init(describing:)) ?? "default") : "system-default")",
            verbose: true
        )
    }

    /// Stop requests cannot impose a hard deadline on CoreAudio's blocking start call.
    static func startConfiguredEngineWithTimeout(
        engine: AVAudioEngine,
        timeoutSeconds: Double
    ) async throws {
        let engineBox = MLXAudioEngineBox(engine: engine)
        try await withTaskCancellationHandler {
            try Task.checkCancellation()
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    // Detached so the blocking start can never run on the main actor.
                    try await Task.detached(priority: .userInitiated) {
                        try engineBox.engine.start()
                    }.value
                    if Task.isCancelled { engineBox.engine.stop() }
                    try Task.checkCancellation()
                }
                group.addTask {
                    try await Task.sleep(for: .seconds(timeoutSeconds))
                    engineBox.engine.stop()
                    throw MLXCaptureStartError.engineStartTimedOut(timeoutSeconds)
                }
                defer { group.cancelAll() }
                _ = try await group.next()
            }
        } onCancel: {
            engineBox.engine.stop()
        }
    }

    private static func applyPreferredInputDeviceIfNeeded(
        inputNode: AVAudioInputNode,
        preferredInputDeviceID: AudioDeviceID?
    ) -> Bool {
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
            VoxtLog.asrWarning("Unable to switch input device. status=\(status)")
            return false
        }
        return true
    }
}
