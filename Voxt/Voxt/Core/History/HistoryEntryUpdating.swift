// HistoryEntryUpdating.swift
// Provides History Entry Updating for history storage and display support.

import Foundation

fileprivate enum HistoryEntryUpdate<Value> {
    case keep
    case set(Value)

    func resolved(current: Value) -> Value {
        switch self {
        case .keep:
            return current
        case .set(let value):
            return value
        }
    }
}

extension TranscriptionHistoryEntry {
    func updatingDictionaryCorrectionResult(
        text: String,
        dictionaryCorrectedTerms: [String],
        dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot]
    ) -> TranscriptionHistoryEntry {
        copy(
            text: text,
            dictionaryCorrectedTerms: dictionaryCorrectedTerms,
            dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots
        )
    }

    func updatingDictionaryCorrectedTerms(_ dictionaryCorrectedTerms: [String]) -> TranscriptionHistoryEntry {
        copy(dictionaryCorrectedTerms: dictionaryCorrectedTerms)
    }

    func updatingTranscriptSummary(_ summary: TranscriptSummarySnapshot?) -> TranscriptionHistoryEntry {
        copy(
            transcriptSummary: .set(summary),
            transcriptSummaryStale: .set(false)
        )
    }

    func updatingTranscriptSummaryStale(_ isStale: Bool) -> TranscriptionHistoryEntry {
        copy(transcriptSummaryStale: .set(isStale))
    }

    func updatingSummaryChatMessages(_ summaryChatMessages: [TranscriptSummaryChatMessage]) -> TranscriptionHistoryEntry {
        copy(transcriptSummaryChatMessages: .set(summaryChatMessages))
    }

    func updatingTranscriptSegments(
        _ transcriptSegments: [TranscriptSegment],
        text: String
    ) -> TranscriptionHistoryEntry {
        copy(
            text: text,
            transcriptSegments: .set(transcriptSegments)
        )
    }

    func updatingTranscriptionChatMessages(_ transcriptionChatMessages: [TranscriptSummaryChatMessage]) -> TranscriptionHistoryEntry {
        copy(transcriptionChatMessages: .set(transcriptionChatMessages))
    }

    func updatingTranscriptionEntry(
        text: String,
        createdAt: Date,
        audioDurationSeconds: TimeInterval?,
        transcriptionProcessingDurationSeconds: TimeInterval?,
        llmDurationSeconds: TimeInterval?,
        whisperWordTimings: [WhisperHistoryWordTiming]?,
        senseVoiceMetadata: SenseVoiceTranscriptMetadata?,
        transcriptionChatMessages: [TranscriptSummaryChatMessage],
        dictionaryHitTerms: [String],
        dictionaryCorrectedTerms: [String],
        dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot],
        dictionarySuggestedTerms: [DictionarySuggestionSnapshot]
    ) -> TranscriptionHistoryEntry {
        copy(
            text: text,
            createdAt: createdAt,
            audioDurationSeconds: .set(audioDurationSeconds),
            transcriptionProcessingDurationSeconds: .set(transcriptionProcessingDurationSeconds),
            llmDurationSeconds: .set(llmDurationSeconds),
            whisperWordTimings: .set(whisperWordTimings),
            senseVoiceMetadata: .set(senseVoiceMetadata),
            transcriptionChatMessages: .set(transcriptionChatMessages),
            dictionaryHitTerms: dictionaryHitTerms,
            dictionaryCorrectedTerms: dictionaryCorrectedTerms,
            dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots,
            dictionarySuggestedTerms: dictionarySuggestedTerms
        )
    }

    func updatingAudioRelativePath(_ audioRelativePath: String?) -> TranscriptionHistoryEntry {
        copy(audioRelativePath: .set(audioRelativePath))
    }

    func updatingOutputDestination(
        focusedAppName: String?,
        focusedAppBundleID: String?,
        browserURLHost: String?,
        browserURLOrigin: String?
    ) -> TranscriptionHistoryEntry {
        copy(
            focusedAppName: .set(focusedAppName),
            focusedAppBundleID: .set(focusedAppBundleID),
            browserURLHost: .set(browserURLHost),
            browserURLOrigin: .set(browserURLOrigin)
        )
    }

    private func copy(
        text: String? = nil,
        createdAt: Date? = nil,
        audioDurationSeconds: HistoryEntryUpdate<TimeInterval?> = .keep,
        transcriptionProcessingDurationSeconds: HistoryEntryUpdate<TimeInterval?> = .keep,
        llmDurationSeconds: HistoryEntryUpdate<TimeInterval?> = .keep,
        focusedAppName: HistoryEntryUpdate<String?> = .keep,
        focusedAppBundleID: HistoryEntryUpdate<String?> = .keep,
        browserURLHost: HistoryEntryUpdate<String?> = .keep,
        browserURLOrigin: HistoryEntryUpdate<String?> = .keep,
        audioRelativePath: HistoryEntryUpdate<String?> = .keep,
        whisperWordTimings: HistoryEntryUpdate<[WhisperHistoryWordTiming]?> = .keep,
        senseVoiceMetadata: HistoryEntryUpdate<SenseVoiceTranscriptMetadata?> = .keep,
        transcriptSegments: HistoryEntryUpdate<[TranscriptSegment]?> = .keep,
        transcriptSummary: HistoryEntryUpdate<TranscriptSummarySnapshot?> = .keep,
        transcriptSummaryStale: HistoryEntryUpdate<Bool> = .keep,
        transcriptSummaryChatMessages: HistoryEntryUpdate<[TranscriptSummaryChatMessage]?> = .keep,
        transcriptionChatMessages: HistoryEntryUpdate<[TranscriptSummaryChatMessage]?> = .keep,
        dictionaryHitTerms: [String]? = nil,
        dictionaryCorrectedTerms: [String]? = nil,
        dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot]? = nil,
        dictionarySuggestedTerms: [DictionarySuggestionSnapshot]? = nil
    ) -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: id,
            text: text ?? self.text,
            createdAt: createdAt ?? self.createdAt,
            transcriptionEngine: transcriptionEngine,
            transcriptionModel: transcriptionModel,
            enhancementMode: enhancementMode,
            enhancementModel: enhancementModel,
            kind: kind,
            isTranslation: isTranslation,
            audioDurationSeconds: audioDurationSeconds.resolved(current: self.audioDurationSeconds),
            transcriptionProcessingDurationSeconds: transcriptionProcessingDurationSeconds.resolved(
                current: self.transcriptionProcessingDurationSeconds
            ),
            llmDurationSeconds: llmDurationSeconds.resolved(current: self.llmDurationSeconds),
            focusedAppName: focusedAppName.resolved(current: self.focusedAppName),
            focusedAppBundleID: focusedAppBundleID.resolved(current: self.focusedAppBundleID),
            browserURLHost: browserURLHost.resolved(current: self.browserURLHost),
            browserURLOrigin: browserURLOrigin.resolved(current: self.browserURLOrigin),
            matchedGroupID: matchedGroupID,
            matchedGroupName: matchedGroupName,
            matchedAppGroupName: matchedAppGroupName,
            matchedURLGroupName: matchedURLGroupName,
            remoteASRProvider: remoteASRProvider,
            remoteASRModel: remoteASRModel,
            remoteASREndpoint: remoteASREndpoint,
            remoteLLMProvider: remoteLLMProvider,
            remoteLLMModel: remoteLLMModel,
            remoteLLMEndpoint: remoteLLMEndpoint,
            audioRelativePath: audioRelativePath.resolved(current: self.audioRelativePath),
            whisperWordTimings: whisperWordTimings.resolved(current: self.whisperWordTimings),
            senseVoiceMetadata: senseVoiceMetadata.resolved(current: self.senseVoiceMetadata),
            transcriptSegments: transcriptSegments.resolved(current: self.transcriptSegments),
            transcriptAudioRelativePath: transcriptAudioRelativePath,
            meetingCaptureMode: meetingCaptureMode,
            transcriptSummary: transcriptSummary.resolved(current: self.transcriptSummary),
            transcriptSummaryStale: transcriptSummaryStale.resolved(current: self.transcriptSummaryStale),
            transcriptSummaryChatMessages: transcriptSummaryChatMessages.resolved(
                current: self.transcriptSummaryChatMessages
            ),
            displayTitle: displayTitle,
            transcriptionChatMessages: transcriptionChatMessages.resolved(current: self.transcriptionChatMessages),
            dictionaryHitTerms: dictionaryHitTerms ?? self.dictionaryHitTerms,
            dictionaryCorrectedTerms: dictionaryCorrectedTerms ?? self.dictionaryCorrectedTerms,
            dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots ?? self.dictionaryCorrectionSnapshots,
            dictionarySuggestedTerms: dictionarySuggestedTerms ?? self.dictionarySuggestedTerms
        )
    }
}
