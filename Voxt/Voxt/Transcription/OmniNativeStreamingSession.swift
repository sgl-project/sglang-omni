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

    /// MOSS-Transcribe-Diarize windows over the realtime transcription socket,
    /// with the task prompt Voxt's Swift live session used.
    static func moss(runtime: OmniASRRuntime, prompt: String?) async throws -> OmniNativeStreamingSession {
        let endpoint = try await runtime.beginUse()
        let session = OmniRealtimeTranscriptionSession(
            endpoint: endpoint,
            language: nil,
            prompt: prompt,
            joining: .lines
        )
        return OmniNativeStreamingSession(session: session, runtime: runtime)
    }

    /// Nemotron 3.5 ASR's cache-aware stream over the realtime transcription
    /// socket, with the live latency and prompt language Voxt's Swift session used.
    static func nemotron(
        runtime: OmniASRRuntime,
        language: String,
        chunkMilliseconds: Int
    ) async throws -> OmniNativeStreamingSession {
        let endpoint = try await runtime.beginUse()
        let session = OmniRealtimeTranscriptionSession(
            endpoint: endpoint,
            language: language,
            chunkMilliseconds: chunkMilliseconds
        )
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
                case .ended(let text, let segments):
                    continuation.yield(.ended(STTOutput(
                        text: text,
                        segments: segments.isEmpty ? nil : segments.map(\.transcriptSegment)
                    )))
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

nonisolated extension OmniSpeakerSegment {
    /// The MLXAudio segment Voxt's MOSS handling reads; no speaker means none.
    var transcriptSegment: STTTranscriptSegment {
        STTTranscriptSegment(
            text: text,
            startTime: startSeconds,
            endTime: endSeconds,
            speakerID: speakerID.isEmpty ? nil : speakerID
        )
    }
}
