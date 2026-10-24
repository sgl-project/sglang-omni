// TranscriptionCapturePipeline.swift
// Provides Transcription Capture Pipeline for transcription processing.

import Foundation

enum TranscriptionCapturePipeline: String, Equatable, Sendable {
    case finalOnly
    case liveDisplay
    case noteSession

    var stageLabels: [String] {
        switch self {
        case .finalOnly:
            return [
                "record",
                "stop",
                "finalASR",
                "enhance",
                "deliver"
            ]
        case .liveDisplay:
            return [
                "record",
                "livePartial",
                "stop",
                "previewASR",
                "finalASR",
                "enhance",
                "deliver"
            ]
        case .noteSession:
            return [
                "record",
                "livePartial",
                "noteSegment",
                "stop",
                "finalASR",
                "enhance",
                "deliver"
            ]
        }
    }

    var usesLiveDisplay: Bool {
        switch self {
        case .finalOnly:
            return false
        case .liveDisplay, .noteSession:
            return true
        }
    }

    static func resolve(
        realtimeTextDisplayEnabled: Bool,
        captureSessionMode: AppDelegate.TranscriptionCaptureSessionMode
    ) -> Self {
        switch captureSessionMode {
        case .noteSession:
            return .noteSession
        case .standard:
            return realtimeTextDisplayEnabled ? .liveDisplay : .finalOnly
        }
    }
}
