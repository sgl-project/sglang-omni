// TranscriberProtocol.swift
// Provides Transcriber Protocol for transcription engines.

import Foundation
import Combine
import AVFoundation
import Speech

nonisolated enum CancellableSystemCallback {
    static func wait<Value: Sendable>(
        _ start: (@escaping @Sendable (Value) -> Void) -> Void
    ) async -> Value? {
        guard !Task.isCancelled else { return nil }
        let values = AsyncStream<Value> { continuation in
            start { value in
                continuation.yield(value)
                continuation.finish()
            }
        }
        for await value in values {
            return value
        }
        return nil
    }
}

nonisolated enum RecordingPermissionRequest {
    static func microphoneAccess() async -> Bool {
        await CancellableSystemCallback.wait { completion in
            AVCaptureDevice.requestAccess(for: .audio, completionHandler: completion)
        } ?? false
    }

    static func speechRecognitionAccess() async -> Bool {
        await CancellableSystemCallback.wait { completion in
            SFSpeechRecognizer.requestAuthorization(completion)
        } == .authorized
    }
}

/// Protocol that both SpeechTranscriber (Direct Dictation) and MLXTranscriber conform to.
/// Provides a unified interface for the AppDelegate to interact with either engine.
@MainActor
protocol TranscriberProtocol: ObservableObject {
    var isRecording: Bool { get }
    var audioLevel: Float { get }
    var transcribedText: String { get }
    var isEnhancing: Bool { get set }
    var isFinalizingTranscription: Bool { get }

    var onTranscriptionFinished: ((String) -> Void)? { get set }

    func requestPermissions() async -> Bool
    func startRecording()
    func stopRecording()
}
