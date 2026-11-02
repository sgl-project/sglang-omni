// VoiceEndCommandFlow.swift
// Provides Voice End Command Flow for recording session routing.

import Foundation

extension AppDelegate {
    func resetVoiceEndCommandState() {
        voiceEndCommandState = VoiceEndCommandState()
    }

    func sanitizedFinalTranscriptionText(_ rawText: String) -> String {
        let shouldStripTrailingCommand =
            voiceEndCommandState.didAutoStop ||
            voiceEndCommandState.pendingStrippedText != nil ||
            (voiceEndCommandEnabled && trailingVoiceEndCommandRange(in: rawText) != nil)
        guard shouldStripTrailingCommand else { return rawText }

        defer {
            voiceEndCommandState.didAutoStop = false
            voiceEndCommandState.pendingStrippedText = nil
        }

        let stripped = voiceEndCommandState.pendingStrippedText ?? removingTrailingVoiceEndCommand(from: rawText)
        if stripped != rawText {
            VoxtLog.hotkey("Voice end command removed from final transcription output.")
        }
        return stripped
    }

    func shouldStopRecordingForVoiceEndCommand() -> Bool {
        guard isSessionActive, recordingStoppedAt == nil else { return false }
        guard voiceEndCommandEnabled else {
            voiceEndCommandState.lastDetectedCommand = false
            voiceEndCommandState.pendingStrippedText = nil
            return false
        }

        guard !voiceEndCommandText.isEmpty else {
            voiceEndCommandState.lastDetectedCommand = false
            return false
        }

        let hasTrailingCommand = trailingVoiceEndCommandRange(in: overlayState.transcribedText) != nil
        if !hasTrailingCommand {
            if voiceEndCommandState.lastDetectedCommand {
                VoxtLog.hotkey("Voice end command candidate cleared because transcript tail changed.")
            }
            voiceEndCommandState.lastDetectedCommand = false
            voiceEndCommandState.pendingStrippedText = nil
            return false
        }

        if !voiceEndCommandState.lastDetectedCommand {
            VoxtLog.hotkey("Voice end command detected at transcript tail. awaitingSilenceSec=\(voiceEndCommandState.silenceDuration)")
        }
        voiceEndCommandState.lastDetectedCommand = true
        voiceEndCommandState.pendingStrippedText = removingTrailingVoiceEndCommand(from: overlayState.transcribedText)
        let silentDuration = Date().timeIntervalSince(lastSignificantAudioAt)
        return silentDuration >= voiceEndCommandState.silenceDuration
    }

    private func trailingVoiceEndCommandRange(in text: String) -> Range<String.Index>? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let command = normalizedVoiceEndCommandText()
        guard !command.isEmpty else { return nil }

        let commandEnd = trimmedEndIndexSkippingVoiceCommandDelimiters(in: trimmed)
        guard commandEnd > trimmed.startIndex else { return nil }
        guard let commandStart = trimmed.index(commandEnd, offsetBy: -command.count, limitedBy: trimmed.startIndex) else {
            return nil
        }

        let candidate = String(trimmed[commandStart..<commandEnd])
        guard candidate.compare(command, options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive], range: nil, locale: .current) == .orderedSame else {
            return nil
        }

        if commandStart > trimmed.startIndex {
            let previousIndex = trimmed.index(before: commandStart)
            guard isVoiceCommandDelimiter(trimmed[previousIndex]) else {
                return nil
            }
        }

        return commandStart..<trimmed.endIndex
    }

    private func removingTrailingVoiceEndCommand(from text: String) -> String {
        guard let commandRange = trailingVoiceEndCommandRange(in: text) else {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        var prefix = String(text[..<commandRange.lowerBound])
        prefix = prefix.trimmingCharacters(in: .whitespacesAndNewlines)

        while let last = prefix.unicodeScalars.last,
              CharacterSet.punctuationCharacters.contains(last) || CharacterSet.symbols.contains(last) {
            prefix.unicodeScalars.removeLast()
            prefix = prefix.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return prefix.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func normalizedVoiceEndCommandText() -> String {
        voiceEndCommandText
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func trimmedEndIndexSkippingVoiceCommandDelimiters(in text: String) -> String.Index {
        var end = text.endIndex
        while end > text.startIndex {
            let previous = text.index(before: end)
            guard isVoiceCommandDelimiter(text[previous]) else { break }
            end = previous
        }
        return end
    }

    private func isVoiceCommandDelimiter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            CharacterSet.whitespacesAndNewlines.contains(scalar) ||
            CharacterSet.punctuationCharacters.contains(scalar) ||
            CharacterSet.symbols.contains(scalar)
        }
    }
}
