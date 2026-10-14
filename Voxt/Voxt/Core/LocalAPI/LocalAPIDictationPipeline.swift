// LocalAPIDictationPipeline.swift
// Provides the audio-to-text steps for one Local API dictation, with the Voxt-specific
// work (ASR, enhancement, dictionary, history) injected by the app.

import AVFoundation
import Foundation

struct LocalAPIFinalText: Equatable {
    var text: String
    var dictionaryHitTerms: [String] = []
    var dictionaryCorrectedTerms: [String] = []
}

struct LocalAPIDictationOutcome: Equatable {
    var id: UUID
    var text: String
    var rawText: String
    var usedEnhancement: Bool
    var transcriptionSeconds: TimeInterval
    var enhancementSeconds: TimeInterval?
    var dictionaryHitTerms: [String]
    var dictionaryCorrectedTerms: [String]
}

private struct LocalAPIDictationResponseBody: Encodable {
    var id: String
    var text: String
    var rawText: String
    var enhanced: Bool
    var asrMs: Int
    var enhanceMs: Int
}

final class LocalAPIDictationPipeline {
    let temporaryDirectory: URL
    /// Returns why the Mac can't take a dictation right now (it is recording, in a meeting, …).
    let busyReason: () -> String?
    let isReadableAudio: (URL) -> Bool
    /// Audio file → normalized transcript, using the engine selected in Voxt.
    let transcribe: (URL) async throws -> String
    /// Transcript → cleaned-up text, using Voxt's transcription enhancement settings.
    let enhance: (String) async throws -> String
    /// Applies dictionary corrections to the final text.
    let finalize: (String) -> LocalAPIFinalText
    let record: (LocalAPIDictationOutcome) -> Void

    private var isRunning = false

    init(
        temporaryDirectory: URL,
        busyReason: @escaping () -> String?,
        isReadableAudio: @escaping (URL) -> Bool = LocalAPIDictationPipeline.isReadableAudioFile,
        transcribe: @escaping (URL) async throws -> String,
        enhance: @escaping (String) async throws -> String,
        finalize: @escaping (String) -> LocalAPIFinalText,
        record: @escaping (LocalAPIDictationOutcome) -> Void
    ) {
        self.temporaryDirectory = temporaryDirectory
        self.busyReason = busyReason
        self.isReadableAudio = isReadableAudio
        self.transcribe = transcribe
        self.enhance = enhance
        self.finalize = finalize
        self.record = record
    }

    static func isReadableAudioFile(_ url: URL) -> Bool {
        (try? AVAudioFile(forReading: url)) != nil
    }

    func run(_ request: LocalAPIDictationRequest) async -> LocalAPIHTTPResponse {
        if let reason = busyReason() {
            return .error(status: 503, reason)
        }
        guard !isRunning else {
            return .error(status: 503, "Voxt is still working on the previous dictation.")
        }
        isRunning = true
        defer { isRunning = false }

        let audioURL = temporaryDirectory.appendingPathComponent("local-api-\(UUID().uuidString).\(request.fileExtension)")
        defer { try? FileManager.default.removeItem(at: audioURL) }

        do {
            try request.audio.write(to: audioURL, options: .atomic)
        } catch {
            VoxtLog.warning("Local API could not store the audio: \(error.localizedDescription)")
            return .error(status: 500, error.localizedDescription)
        }
        guard isReadableAudio(audioURL) else {
            return .error(status: 400, "The audio could not be read.")
        }

        let rawText: String
        let transcriptionStart = Date()
        do {
            rawText = try await transcribe(audioURL).trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            VoxtLog.warning("Local API transcription failed: \(error.localizedDescription)")
            return .error(status: 500, error.localizedDescription)
        }
        let transcriptionSeconds = Date().timeIntervalSince(transcriptionStart)

        guard !rawText.isEmpty else {
            return .error(status: 422, "No speech detected.")
        }

        var text = rawText
        var usedEnhancement = false
        var enhancementSeconds: TimeInterval?
        if request.enhance {
            let enhancementStart = Date()
            do {
                let enhanced = try await enhance(rawText).trimmingCharacters(in: .whitespacesAndNewlines)
                if !enhanced.isEmpty {
                    text = enhanced
                    usedEnhancement = true
                }
            } catch {
                VoxtLog.warning("Local API enhancement failed, returning raw transcript: \(error.localizedDescription)")
            }
            enhancementSeconds = Date().timeIntervalSince(enhancementStart)
        }

        let final = finalize(text)
        let outcome = LocalAPIDictationOutcome(
            id: UUID(),
            text: final.text,
            rawText: rawText,
            usedEnhancement: usedEnhancement,
            transcriptionSeconds: transcriptionSeconds,
            enhancementSeconds: enhancementSeconds,
            dictionaryHitTerms: final.dictionaryHitTerms,
            dictionaryCorrectedTerms: final.dictionaryCorrectedTerms
        )
        record(outcome)

        return .json(status: 200, LocalAPIDictationResponseBody(
            id: outcome.id.uuidString,
            text: outcome.text,
            rawText: outcome.rawText,
            enhanced: usedEnhancement,
            asrMs: Int(transcriptionSeconds * 1000),
            enhanceMs: Int((enhancementSeconds ?? 0) * 1000)
        ))
    }
}
