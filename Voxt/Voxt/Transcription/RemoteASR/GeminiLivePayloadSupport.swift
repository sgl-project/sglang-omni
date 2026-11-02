import Foundation

enum GeminiLiveTranscriptJoining {
    /// Gemini emits Chinese segments that must not be space-joined, and English
    /// turns that must be. Decide per boundary instead of per session.
    nonisolated static func join(_ segments: [String]) -> String {
        var result = ""
        for segment in segments {
            let piece = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !piece.isEmpty else { continue }
            if result.isEmpty {
                result = piece
                continue
            }
            if needsSeparator(previous: result, next: piece) {
                result += " "
            }
            result += piece
        }
        return result
    }

    private nonisolated static func needsSeparator(previous: String, next: String) -> Bool {
        guard let left = previous.unicodeScalars.last,
              let right = next.unicodeScalars.first
        else { return false }
        return !isCJK(left) && !isCJK(right)
    }

    private nonisolated static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3000...0x303F, 0x3400...0x4DBF, 0x4E00...0x9FFF,
             0xF900...0xFAFF, 0xFF00...0xFFEF,
             0x3040...0x30FF, 0xAC00...0xD7AF:
            return true
        default:
            return false
        }
    }
}

enum GeminiLivePayloadSupport {
    static let defaultModel = "gemini-3.5-transcribe-live"
    static let audioMimeType = "audio/pcm;rate=16000"

    static func setupPayload(
        model: String,
        hintPayload: ResolvedASRHintPayload
    ) -> [String: Any] {
        let resolvedModel = model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? defaultModel
            : model.trimmingCharacters(in: .whitespacesAndNewlines)
        let qualifiedModel = resolvedModel.hasPrefix("models/") ? resolvedModel : "models/\(resolvedModel)"

        var transcription: [String: Any] = [
            "languageCodes": languageCodes(from: hintPayload),
            "mode": "SMART"
        ]
        let vocabulary = customVocabulary(from: hintPayload)
        if !vocabulary.isEmpty {
            transcription["customVocabulary"] = vocabulary
        }

        return [
            "setup": [
                "model": qualifiedModel,
                "generationConfig": ["responseModalities": ["TEXT"]],
                "inputAudioTranscription": transcription
            ]
        ]
    }

    static func audioPayload(_ audio: Data) -> [String: Any] {
        [
            "realtimeInput": [
                "audio": [
                    "data": audio.base64EncodedString(),
                    "mimeType": audioMimeType
                ]
            ]
        ]
    }

    static let audioStreamEndPayload: [String: Any] = [
        "realtimeInput": ["audioStreamEnd": true]
    ]

    /// An empty array asks Gemini to auto-detect, which is also what keeps
    /// mid-sentence code-mixing working.
    static func languageCodes(from hintPayload: ResolvedASRHintPayload) -> [String] {
        var seen = Set<String>()
        return hintPayload.languageHints.compactMap { hint in
            let normalized = hint.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalized.isEmpty, seen.insert(normalized.lowercased()).inserted else { return nil }
            return normalized
        }
    }

    static func customVocabulary(from hintPayload: ResolvedASRHintPayload) -> [String] {
        var seen = Set<String>()
        let terms = hintPayload.contextualPhrases.compactMap { phrase -> String? in
            let normalized = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalized.isEmpty, seen.insert(normalized).inserted else { return nil }
            return normalized
        }
        return Array(terms.prefix(1000))
    }

    struct TranscriptUpdate {
        var interim: String?
        var final: String?
    }

    static func transcriptUpdate(from object: [String: Any]) -> TranscriptUpdate {
        guard let serverContent = value(in: object, "serverContent", "server_content") as? [String: Any] else {
            return TranscriptUpdate()
        }
        return TranscriptUpdate(
            interim: text(in: serverContent, "interimInputTranscription", "interim_input_transcription"),
            final: text(in: serverContent, "inputTranscription", "input_transcription")
        )
    }

    static func isSetupComplete(_ object: [String: Any]) -> Bool {
        value(in: object, "setupComplete", "setup_complete") != nil
    }

    static func isSessionTerminal(_ object: [String: Any]) -> Bool {
        if value(in: object, "goAway", "go_away") != nil {
            return true
        }
        guard let serverContent = value(in: object, "serverContent", "server_content") as? [String: Any] else {
            return false
        }
        return (value(in: serverContent, "turnComplete", "turn_complete") as? Bool) == true
    }

    static func errorMessage(from object: [String: Any]) -> String? {
        guard let error = value(in: object, "error", "error") as? [String: Any] else { return nil }
        if let message = error["message"] as? String, !message.isEmpty {
            return message
        }
        if let status = error["status"] as? String, !status.isEmpty {
            return status
        }
        return "Gemini live transcribe session failed."
    }

    private static func text(in object: [String: Any], _ camel: String, _ snake: String) -> String? {
        guard let container = value(in: object, camel, snake) as? [String: Any],
              let text = container["text"] as? String
        else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// The socket speaks proto JSON, which may arrive in either casing.
    private static func value(in object: [String: Any], _ camel: String, _ snake: String) -> Any? {
        object[camel] ?? object[snake]
    }
}
