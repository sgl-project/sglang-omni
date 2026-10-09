// RemoteASRStreamingContexts.swift
// Provides Remote ASRStreaming Contexts for remote ASR adapters.

import Foundation

/// A handshake result is latched even if it arrives before the upload starts waiting.
actor AsyncGate {
    private var result: Result<Void, Error>?

    func open() {
        guard result == nil else { return }
        result = .success(())
    }

    func fail(_ error: Error) {
        guard result == nil else { return }
        result = .failure(error)
    }

    func wait(timeoutSeconds: TimeInterval = 20) async throws {
        try Task.checkCancellation()
        let deadline = Date().addingTimeInterval(max(timeoutSeconds, 0))
        while result == nil {
            guard Date() < deadline else { throw URLError(.timedOut) }
            try await Task.sleep(for: .milliseconds(30))
        }
        try Task.checkCancellation()
        try result?.get()
    }
}

@MainActor
final class AliyunQwenStreamingContext {
    let session: URLSession
    let ws: URLSessionWebSocketTask
    let responseState: AliyunQwenResponseState
    let generationID: UUID
    let kind: AliyunQwenRealtimeSessionKind
    var isClosed = false
    var didStartAudioStream = false

    init(
        session: URLSession,
        ws: URLSessionWebSocketTask,
        responseState: AliyunQwenResponseState,
        generationID: UUID,
        kind: AliyunQwenRealtimeSessionKind
    ) {
        self.session = session
        self.ws = ws
        self.responseState = responseState
        self.generationID = generationID
        self.kind = kind
    }
}



@MainActor
final class StepFunStreamingContext {
    let session: URLSession
    let ws: URLSessionWebSocketTask
    let responseState: StepFunResponseState
    let generationID: UUID
    var isClosed = false
    var isSessionUpdated = false
    var didStartAudioStream = false
    var shouldCommitAfterSessionUpdate = false
    var pendingAudioChunks: [Data] = []
    var pendingAudioByteCount = 0

    init(
        session: URLSession,
        ws: URLSessionWebSocketTask,
        responseState: StepFunResponseState,
        generationID: UUID
    ) {
        self.session = session
        self.ws = ws
        self.responseState = responseState
        self.generationID = generationID
    }
}



@MainActor
final class AliyunFunStreamingContext {
    let session: URLSession
    let ws: URLSessionWebSocketTask
    let taskID: String
    let responseState: AliyunFunResponseState
    let generationID: UUID
    var isClosed = false
    var didStartAudioStream = false

    init(
        session: URLSession,
        ws: URLSessionWebSocketTask,
        taskID: String,
        responseState: AliyunFunResponseState,
        generationID: UUID
    ) {
        self.session = session
        self.ws = ws
        self.taskID = taskID
        self.responseState = responseState
        self.generationID = generationID
    }
}



@MainActor
final class GeminiLiveStreamingContext {
    let session: URLSession
    let ws: URLSessionWebSocketTask
    let responseState: GeminiLiveResponseState
    let generationID: UUID
    var isClosed = false
    var isSetupComplete = false
    var didStartAudioStream = false
    var shouldEndAudioStreamAfterSetup = false
    var didSendAudioStreamEnd = false
    var pendingAudioChunks: [Data] = []
    var pendingAudioByteCount = 0

    init(
        session: URLSession,
        ws: URLSessionWebSocketTask,
        responseState: GeminiLiveResponseState,
        generationID: UUID
    ) {
        self.session = session
        self.ws = ws
        self.responseState = responseState
        self.generationID = generationID
    }
}
