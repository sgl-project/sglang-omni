// TranscriptionCaptureMetrics.swift
// Provides Transcription Capture Metrics for transcription processing.

import Foundation

struct TranscriptionCaptureMetrics: Equatable, Sendable {
    let callbackCount: Int
    let sampleCount: Int
    let sampleRate: Double
    /// When the first microphone buffer arrived, for the key-down to first audio frame latency.
    let firstFrameAt: Date?

    var capturedAudioSeconds: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(sampleCount) / sampleRate
    }
}
