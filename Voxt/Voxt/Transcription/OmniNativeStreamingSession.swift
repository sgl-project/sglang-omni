import Foundation
import MLXAudioSTT

/// Presents an Omni live session through the native streaming session seam.
///
/// The adapter leases the runtime for the session's lifetime, so retiring the
/// runtime waits for the session to end instead of cutting it off.
nonisolated final class OmniNativeStreamingSession: MLXNativeStreamingSession, @unchecked Sendable {
    let events: AsyncStream<TranscriptionEvent>
    private let session: OmniRealtimeTranscriptionSession
    private let forwarding: Task<Void, Never>

    /// Qwen3-ASR preview over the realtime transcription socket.
    static func qwen(runtime: OmniASRRuntime, language: String?) async throws -> OmniNativeStreamingSession {
        let endpoint = try await runtime.beginUse()
        let session = OmniRealtimeTranscriptionSession(endpoint: endpoint, language: language)
        return OmniNativeStreamingSession(session: session, runtime: runtime)
    }

    private init(session: OmniRealtimeTranscriptionSession, runtime: OmniASRRuntime) {
        self.session = session
        let source = session.events
        let (events, continuation) = AsyncStream.makeStream(of: TranscriptionEvent.self)
        self.events = events
        forwarding = Task.detached {
            for await event in source {
                switch event {
                case .display(let confirmedText, let provisionalText):
                    continuation.yield(.displayUpdate(confirmedText: confirmedText, provisionalText: provisionalText))
                case .ended(let text):
                    continuation.yield(.ended(STTOutput(text: text)))
                case .failed(let message):
                    continuation.yield(.failed(StreamingFailure(message: message)))
                }
            }
            continuation.finish()
            await runtime.endUse()
        }
    }

    func feedAudio(samples: [Float]) {
        session.feedAudio(samples: samples)
    }

    func stop() {
        session.stop()
    }

    func cancel() {
        session.cancel()
    }
}
