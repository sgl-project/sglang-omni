import Foundation

// Read-only presentation policy and configuration normalization. Published
// state, async operations and mutations remain owned by the view model.
extension MeetingDetailViewModel {
    var canExport: Bool {
        switch mode {
        case .history:
            return !segments.isEmpty
        case .live:
            return isPaused && !segments.isEmpty
        case .fileDraft:
            return false
        }
    }

    var canEditSpeakers: Bool {
        captureMode.capabilities.allowsSpeakerFeatures
            && mode == .history
            && historyEntryID != nil
            && transcriptSegmentsPersistence != nil
    }

    var canEditTranscript: Bool {
        mode == .history && historyEntryID != nil && transcriptSegmentsPersistence != nil
    }

    var canRegenerateSummary: Bool {
        mode == .history && historyEntryID != nil && !segments.isEmpty
    }

    var canSendSummaryChat: Bool {
        mode == .history
            && historyEntryID != nil
            && summary != nil
            && !segments.isEmpty
            && hasSummaryModelOptions
            && !isSummaryChatLoading
            && !summaryChatDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var hasSummaryModelOptions: Bool {
        !summaryModelOptions.isEmpty
    }

    var resolvedSummaryModelSelectionID: String {
        if summaryModelOptions.contains(where: { $0.id == summaryModelSelectionID }) {
            return summaryModelSelectionID
        }
        return summaryModelOptions.first?.id ?? summaryModelSelectionID
    }

    var summaryProviderMessage: String {
        summaryProviderStatus.message
    }

    var summaryProviderStatus: MeetingSummaryProviderStatus {
        switch mode {
        case .history:
            return summaryStatusProvider?(summarySettingsSnapshot)
                ?? MeetingSummaryProviderStatus(
                    isAvailable: false,
                    message: AppLocalization.localizedString("Meeting summary is unavailable.")
                )
        case .fileDraft:
            return MeetingSummaryProviderStatus(
                isAvailable: false,
                message: AppLocalization.localizedString("Transcription ready — speaker analysis incomplete")
            )
        case .live:
            return MeetingSummaryProviderStatus(
                isAvailable: false,
                message: AppLocalization.localizedString("Meeting summary is generated after the meeting is saved.")
            )
        }
    }

    var transcriptPresentationMode: TranscriptPresentationMode {
        guard mode != .fileDraft else { return .timeline }
        let resolved = TranscriptPresentationMode(rawValue: transcriptPresentationModeRaw) ?? .timeline
        guard captureMode.capabilities.allowsSpeakerFeatures || resolved != .speakerMarks else {
            return .timeline
        }
        return resolved
    }

    var transcriptSpeakerDisplayMode: TranscriptSpeakerDisplayMode {
        guard mode != .fileDraft else { return .source }
        guard captureMode.capabilities.allowsSpeakerFeatures else { return .source }
        return TranscriptSpeakerDisplayMode(rawValue: transcriptSpeakerDisplayModeRaw) ?? .source
    }

    var availableTranscriptPresentationModes: [TranscriptPresentationMode] {
        mode != .fileDraft && captureMode.capabilities.allowsSpeakerFeatures
            ? TranscriptPresentationMode.allCases
            : [.timeline]
    }

    var showsSpeakerDisplayModePicker: Bool {
        mode != .fileDraft && captureMode.capabilities.allowsSpeakerFeatures
    }

    func export() throws {
        guard canExport else { return }
        try MeetingTranscriptExporter.export(
            segments: segments,
            defaultFilename: MeetingTranscriptExporter.defaultFilename(prefix: "Voxt-Meeting")
        )
    }


    func timelineSpeakerTitle(for segment: MeetingTranscriptSegment) -> String {
        switch transcriptSpeakerDisplayMode {
        case .source:
            return segment.speaker.displayTitle
        case .speaker:
            return speakerTimelineTitle(for: segment)
        }
    }

    func displayedNewestSegmentID() -> UUID? {
        displayedSegments.last?.id
    }


    private func speakerTimelineTitle(for segment: MeetingTranscriptSegment) -> String {
        if let displayName = speakerDisplayNameIfUserFacing(for: segment) {
            return displayName
        }
        let ordinal = speakerOrdinalByIdentityKey[segment.speakerIdentityKey] ?? 1
        return AppLocalization.format("Speaker %d", ordinal)
    }

    private func speakerDisplayNameIfUserFacing(for segment: MeetingTranscriptSegment) -> String? {
        guard let displayName = segment.speakerDisplayName?.trimmingCharacters(in: .whitespacesAndNewlines),
              !displayName.isEmpty,
              !isAudioSourceDisplayName(displayName)
        else {
            return nil
        }
        return displayName
    }

    private func isAudioSourceDisplayName(_ displayName: String) -> Bool {
        let normalized = displayName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }
        if normalized == TranscriptSpeaker.me.displayTitle.lowercased()
            || normalized == TranscriptSpeaker.them.displayTitle.lowercased() {
            return true
        }
        if normalized.range(of: #"^me\s+\d+$"#, options: .regularExpression) != nil {
            return true
        }
        if normalized.range(of: #"^them\s+\d+$"#, options: .regularExpression) != nil {
            return true
        }
        return false
    }


    var summarySettingsSnapshot: MeetingSummarySettingsSnapshot {
        MeetingSummarySettingsSnapshot(
            autoGenerate: summaryAutoGenerate,
            promptTemplate: summaryPromptTemplate.trimmingCharacters(in: .whitespacesAndNewlines),
            modelSelectionID: resolvedSummaryModelSelectionID.isEmpty ? nil : resolvedSummaryModelSelectionID
        )
    }


    static func inferredCaptureMode(from segments: [MeetingTranscriptSegment]) -> MeetingCaptureMode {
        let meaningfulSegments = segments.filter {
            !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        let hasMicrophone = meaningfulSegments.contains { segment in
            segment.audioSource == .microphone || segment.speaker == .me
        }
        let hasSystemAudio = meaningfulSegments.contains { segment in
            segment.audioSource == .systemAudio || segment.speaker == .them
        }

        switch (hasMicrophone, hasSystemAudio) {
        case (true, true):
            return .meeting
        case (false, true):
            return .subtitles
        case (true, false):
            return .recording
        case (false, false):
            return .meeting
        }
    }

    static func resolveSummaryConfiguration(
        settings: MeetingSummarySettingsSnapshot,
        modelOptions: [MeetingSummaryModelOption],
        currentSelectionID: String?
    ) -> MeetingDetailSummaryConfiguration {
        let preferredSelectionID = settings.modelSelectionID?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let normalizedCurrentSelectionID = currentSelectionID?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        let resolvedSelectionID: String
        if modelOptions.contains(where: { $0.id == normalizedCurrentSelectionID }) {
            resolvedSelectionID = normalizedCurrentSelectionID
        } else if modelOptions.contains(where: { $0.id == preferredSelectionID }) {
            resolvedSelectionID = preferredSelectionID
        } else {
            resolvedSelectionID = modelOptions.first?.id ?? ""
        }

        return MeetingDetailSummaryConfiguration(
            autoGenerate: settings.autoGenerate,
            promptTemplate: MeetingSummarySupport.resolvedPromptTemplate(settings.promptTemplate),
            modelSelectionID: resolvedSelectionID
        )
    }
}

struct MeetingDetailSummaryConfiguration {
    let autoGenerate: Bool
    let promptTemplate: String
    let modelSelectionID: String
}
