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
    var mode: LocalAPIComposeMode = .dictate
}

private struct LocalAPIDictationResponseBody: Encodable {
    var id: String
    var text: String
    var rawText: String
    var enhanced: Bool
    var asrMs: Int
    var enhanceMs: Int
    var mode: LocalAPIComposeMode = .dictate
    /// "not_applied" when an edit could not be applied; `text` is then the unchanged selection.
    var status: String?
    var message: String?
}

final class LocalAPIDictationPipeline {
    let temporaryDirectory: URL
    /// Returns why the Mac can't take a dictation right now (it is recording, in a meeting, …).
    let busyReason: () -> String?
    let isReadableAudio: (URL) -> Bool
    /// Audio file → normalized transcript, using the engine selected in Voxt.
    let transcribe: (URL, ASRLanguageSelection) async throws -> String
    /// Transcript → cleaned-up text, using Voxt's transcription enhancement settings; nil when no
    /// text model ran.
    let enhance: (String) async throws -> String?
    /// Applies dictionary corrections to the final text.
    let finalize: (String) -> LocalAPIFinalText
    let record: (LocalAPIDictationOutcome) -> Void
    /// Raw transcript → mode-specific result (cleanup, translate, edit, write), using the
    /// same selected text model through a strict path that never returns raw speech.
    var composeHandler: ((String, LocalAPIComposeOptions) async throws -> String)?
    /// Original passage + correction instruction → the selected text model's raw output.
    /// Corrections do not transcribe audio and do not write history.
    var correctHandler: ((LocalAPICorrectionRequest, String) async throws -> String)?

    private var isRunning = false

    init(
        temporaryDirectory: URL,
        busyReason: @escaping () -> String?,
        isReadableAudio: @escaping (URL) -> Bool = LocalAPIDictationPipeline.isReadableAudioFile,
        transcribe: @escaping (URL, ASRLanguageSelection) async throws -> String,
        enhance: @escaping (String) async throws -> String?,
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
            rawText = try await transcribe(audioURL, request.language).trimmingCharacters(in: .whitespacesAndNewlines)
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
                if let enhanced = try await enhance(rawText)?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !enhanced.isEmpty {
                    text = enhanced
                    usedEnhancement = true
                }
            } catch {
                VoxtLog.warning("Local API enhancement failed, returning raw transcript: \(error.localizedDescription)")
            }
            enhancementSeconds = Date().timeIntervalSince(enhancementStart)
        }

        return finish(
            mode: .dictate,
            rawText: rawText,
            text: text,
            usedEnhancement: usedEnhancement,
            transcriptionSeconds: transcriptionSeconds,
            enhancementSeconds: enhancementSeconds
        )
    }

    /// Every compose mode requires a working text model. Dictate restores the ASR text
    /// when cleanup changes its script; other model failures remain errors.
    func runCompose(_ request: LocalAPIComposeRequest) async -> LocalAPIHTTPResponse {
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
            rawText = try await transcribe(audioURL, request.language).trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            VoxtLog.warning("Local API compose transcription failed: \(error.localizedDescription)")
            return .error(status: 500, error.localizedDescription)
        }
        let transcriptionSeconds = Date().timeIntervalSince(transcriptionStart)

        guard !rawText.isEmpty else {
            return .error(status: 422, "No speech detected.")
        }

        guard let composeHandler else {
            return .error(status: 503, "No text model is available for this operation.")
        }
        let enhancementStart = Date()
        let text: String
        do {
            let composed = try await composeHandler(rawText, request.options)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !composed.isEmpty else {
                return .error(status: 503, "The text model returned nothing.")
            }
            if request.options.mode == .dictate {
                text = LocalAPIComposePrompt.preservingDictationScript(composed, rawText: rawText)
                if text != composed {
                    VoxtLog.warning("Local API dictate cleanup discarded: transcript script changed.")
                }
            } else {
                text = composed
            }
        } catch is LocalAPIComposePrompt.EditNotApplied {
            // No dictionary finalization or history entry: the selection must remain
            // byte-for-byte unchanged when the edit could not be applied.
            return .json(status: 200, LocalAPIDictationResponseBody(
                id: UUID().uuidString, text: request.options.selectedText, rawText: rawText, enhanced: false,
                asrMs: Int(transcriptionSeconds * 1000),
                enhanceMs: Int(Date().timeIntervalSince(enhancementStart) * 1000),
                mode: request.options.mode, status: "not_applied", message: "Couldn't apply that edit"
            ))
        } catch {
            VoxtLog.warning("Local API compose failed: \(error.localizedDescription)")
            return .error(status: 503, error.localizedDescription)
        }
        let enhancementSeconds = Date().timeIntervalSince(enhancementStart)

        return finish(
            mode: request.options.mode,
            rawText: rawText,
            text: text,
            usedEnhancement: true,
            transcriptionSeconds: transcriptionSeconds,
            enhancementSeconds: enhancementSeconds
        )
    }

    /// Handles POST /v1/corrections. Shares the busy and in-flight guard with dictation
    /// and compose, then returns one validated word pair or null. Nothing is recorded.
    func runCorrection(_ request: LocalAPICorrectionRequest) async -> LocalAPIHTTPResponse {
        if let reason = busyReason() {
            return .error(status: 503, reason)
        }
        guard !isRunning else {
            return .error(status: 503, "Voxt is still working on the previous dictation.")
        }
        isRunning = true
        defer { isRunning = false }

        if LocalAPICorrection.isIgnorable(original: request.original, edited: request.edited) {
            return .json(status: 200, LocalAPICorrectionResponse(correction: nil))
        }
        guard let correctHandler else {
            return .error(status: 503, "No text model is available for this operation.")
        }
        let instruction = LocalAPICorrectionPrompt.instruction(original: request.original, edited: request.edited)
        let output: String
        do {
            output = try await correctHandler(request, instruction)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            VoxtLog.warning("Local API correction failed: \(error.localizedDescription)")
            return .error(status: 503, error.localizedDescription)
        }
        guard !output.isEmpty else {
            return .error(status: 503, "The text model returned nothing.")
        }
        do {
            let pair = try LocalAPICorrection.interpret(
                modelOutput: output,
                original: request.original,
                edited: request.edited
            )
            return .json(status: 200, LocalAPICorrectionResponse(correction: pair))
        } catch let failure as LocalAPICorrectionFailure {
            let message: String
            switch failure {
            case .invalidJSON:
                message = "The text model did not return valid correction JSON."
            case .mismatch:
                message = "The text model returned a correction that does not match the edit."
            }
            return .error(status: 503, message)
        } catch {
            return .error(status: 503, error.localizedDescription)
        }
    }

    private func finish(
        mode: LocalAPIComposeMode,
        rawText: String,
        text: String,
        usedEnhancement: Bool,
        transcriptionSeconds: TimeInterval,
        enhancementSeconds: TimeInterval?
    ) -> LocalAPIHTTPResponse {
        let final = finalize(text)
        let outcome = LocalAPIDictationOutcome(
            id: UUID(),
            text: final.text,
            rawText: rawText,
            usedEnhancement: usedEnhancement,
            transcriptionSeconds: transcriptionSeconds,
            enhancementSeconds: enhancementSeconds,
            dictionaryHitTerms: final.dictionaryHitTerms,
            dictionaryCorrectedTerms: final.dictionaryCorrectedTerms,
            mode: mode
        )
        record(outcome)

        return .json(status: 200, LocalAPIDictationResponseBody(
            id: outcome.id.uuidString,
            text: outcome.text,
            rawText: outcome.rawText,
            enhanced: usedEnhancement,
            asrMs: Int(transcriptionSeconds * 1000),
            enhanceMs: Int((enhancementSeconds ?? 0) * 1000),
            mode: mode
        ))
    }
}
