// TranscriptionDetailViews.swift
// Provides Transcription Detail Views for window and overlay UI.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

enum TranscriptionDetailPresentationStyle {
    case popover
    case window
}

struct TranscriptionDetailContentView: View {
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @ObservedObject var viewModel: TranscriptionDetailViewModel
    let locale: Locale
    let style: TranscriptionDetailPresentationStyle
    let contentPaddingOverride: CGFloat?
    let verticalPaddingOverride: CGFloat?

    @StateObject private var playbackController: HistoryAudioPlaybackController

    init(
        viewModel: TranscriptionDetailViewModel,
        locale: Locale,
        style: TranscriptionDetailPresentationStyle,
        contentPaddingOverride: CGFloat? = nil,
        verticalPaddingOverride: CGFloat? = nil
    ) {
        self.viewModel = viewModel
        self.locale = locale
        self.style = style
        self.contentPaddingOverride = contentPaddingOverride
        self.verticalPaddingOverride = verticalPaddingOverride
        _playbackController = StateObject(wrappedValue: HistoryAudioPlaybackController(audioURL: viewModel.audioURL))
    }

    private var preferredContentWidth: CGFloat? {
        style == .popover ? 360 : nil
    }

    private var stackSpacing: CGFloat {
        style == .popover ? 10 : 14
    }

    private var contentPadding: CGFloat {
        contentPaddingOverride ?? (style == .popover ? 8 : 14)
    }

    private var verticalPadding: CGFloat {
        verticalPaddingOverride ?? (style == .popover ? 10 : 14)
    }

    private var sectionSpacing: CGFloat {
        style == .popover ? 8 : 10
    }

    private var sectionTitleFont: Font {
        style == .popover ? .headline : .system(size: 15, weight: .semibold)
    }

    private var sectionCardFill: Color {
        style == .popover
            ? Color.clear
            : Color(nsColor: .controlBackgroundColor).opacity(0.78)
    }

    private var hasDictionaryActivity: Bool {
        !viewModel.entry.dictionaryHitTerms.isEmpty ||
        !viewModel.entry.dictionaryCorrectedTerms.isEmpty ||
        !viewModel.entry.dictionarySuggestedTerms.isEmpty
    }

    var body: some View {
        let _ = interfaceLanguageRaw
        ScrollView {
            VStack(alignment: .leading, spacing: stackSpacing) {
                detailSection(
                    title: localized("Text"),
                    trailing: {
                        if viewModel.canShowManualCorrection {
                            if viewModel.isEditingCorrection {
                                HStack(spacing: 8) {
                                    Button(localized("Cancel")) {
                                        viewModel.cancelManualCorrection()
                                    }
                                    .buttonStyle(SettingsPillButtonStyle())
                                    .disabled(viewModel.isSubmittingCorrection)

                                    Button {
                                        viewModel.submitManualCorrection()
                                    } label: {
                                        HStack(spacing: 6) {
                                            if viewModel.isSubmittingCorrection {
                                                ProgressView()
                                                    .controlSize(.small)
                                                Text(localized("Loading…"))
                                            } else {
                                                Text(localized("Confirm"))
                                            }
                                        }
                                    }
                                    .buttonStyle(SettingsPillButtonStyle())
                                    .disabled(!viewModel.canConfirmManualCorrection)
                                }
                            } else {
                                Button(localized("Correct")) {
                                    viewModel.beginManualCorrection()
                                }
                                .buttonStyle(SettingsPillButtonStyle())
                            }
                        }
                    }
                ) {
                    if viewModel.isEditingCorrection {
                        TextEditor(text: $viewModel.correctionDraft)
                            .font(.system(size: style == .popover ? 13 : 14, weight: .medium))
                            .scrollContentBackground(.hidden)
                            .frame(minHeight: style == .popover ? 120 : 160)
                            .padding(8)
                            .background(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .fill(Color(nsColor: .textBackgroundColor))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .strokeBorder(Color(nsColor: .quaternaryLabelColor).opacity(0.25), lineWidth: 1)
                            )

                        if let errorMessage = viewModel.correctionErrorMessage,
                           !errorMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text(errorMessage)
                                .font(.caption)
                                .foregroundStyle(.red)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    } else {
                        correctionPreviewText
                            .font(.system(size: style == .popover ? 13 : 14, weight: .medium))
                            .multilineTextAlignment(.leading)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                }

                detailSection(title: localized("Metadata")) {
                    detailLine(label: localized("Type"), value: historyKindTitle(viewModel.entry.kind))
                    detailLine(label: localized("Created At"), value: formattedDate(viewModel.entry.createdAt))
                    detailLine(label: localized("Engine"), value: viewModel.entry.transcriptionEngine)
                    detailLine(label: localized("Model"), value: viewModel.entry.transcriptionModel)
                    detailLine(label: modeMetadataLabel(for: viewModel.entry.kind), value: viewModel.entry.enhancementMode)
                    detailLine(label: modelMetadataLabel(for: viewModel.entry.kind), value: viewModel.entry.enhancementModel)
                    optionalDetailLine(label: localized("Audio Duration"), value: formattedDuration(viewModel.entry.audioDurationSeconds))
                    optionalDetailLine(label: localized("ASR Processing"), value: formattedDuration(viewModel.entry.transcriptionProcessingDurationSeconds))
                    optionalDetailLine(label: localized("LLM Duration"), value: formattedDuration(viewModel.entry.llmDurationSeconds))
                    optionalDetailLine(label: localized("Remote ASR Provider"), value: viewModel.entry.remoteASRProvider)
                    optionalDetailLine(label: localized("Remote ASR Model"), value: viewModel.entry.remoteASRModel)
                    optionalDetailLine(label: localized("Remote ASR Endpoint"), value: viewModel.entry.remoteASREndpoint)
                    optionalDetailLine(label: localized("Remote LLM Provider"), value: viewModel.entry.remoteLLMProvider)
                    optionalDetailLine(label: localized("Remote LLM Model"), value: viewModel.entry.remoteLLMModel)
                    optionalDetailLine(label: localized("Remote LLM Endpoint"), value: viewModel.entry.remoteLLMEndpoint)
                    optionalDetailLine(label: localized("SenseVoice Language"), value: viewModel.entry.senseVoiceMetadata?.language)
                    optionalDetailLine(label: localized("SenseVoice Emotion"), value: viewModel.entry.senseVoiceMetadata?.emotion)
                    optionalDetailLine(label: localized("SenseVoice Event"), value: viewModel.entry.senseVoiceMetadata?.event)
                    optionalDetailLine(label: localized("Focused App"), value: viewModel.entry.focusedAppName)
                    optionalDetailLine(label: localized("Focused App Bundle ID"), value: viewModel.entry.focusedAppBundleID)
                    optionalDetailLine(label: localized("Matched Group"), value: viewModel.entry.matchedGroupName)
                    optionalDetailLine(label: localized("App Group"), value: viewModel.entry.matchedAppGroupName)
                    optionalDetailLine(label: localized("URL Group"), value: viewModel.entry.matchedURLGroupName)
                }

                detailSection(title: localized("Audio")) {
                    if playbackController.isAvailable {
                        HistoryAudioPlayerView(
                            controller: playbackController,
                            compact: style == .popover
                        )
                    } else {
                        HistoryAudioUnavailableView(compact: style == .popover)
                    }
                }

                if let whisperWordTimings = viewModel.entry.whisperWordTimings,
                   !whisperWordTimings.isEmpty {
                    detailSection(title: localized("Whisper Timestamps")) {
                        ForEach(Array(whisperWordTimings.enumerated()), id: \.offset) { _, timing in
                            detailLine(
                                label: timeRangeLabel(for: timing),
                                value: timing.word
                            )
                        }
                    }
                }

                if let senseVoiceMetadata = viewModel.entry.senseVoiceMetadata,
                   !senseVoiceMetadata.segments.isEmpty {
                    detailSection(title: localized("SenseVoice Segments")) {
                        detailLine(
                            label: localized("Segmentation"),
                            value: senseVoiceMetadata.usedVADSegmentation
                                ? localized("VAD Segmented")
                                : localized("Single Pass")
                        )
                        ForEach(senseVoiceMetadata.segments) { segment in
                            let start = TranscriptFormatter.timestampString(for: segment.startSeconds)
                            let end = TranscriptFormatter.timestampString(for: segment.endSeconds)
                            let labels = [segment.language, segment.emotion, segment.event]
                                .compactMap { value -> String? in
                                    let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                                    return trimmed.isEmpty ? nil : trimmed
                                }
                                .joined(separator: " / ")
                            detailLine(
                                label: labels.isEmpty ? "\(start) - \(end)" : "\(start) - \(end) · \(labels)",
                                value: segment.text
                            )
                        }
                    }
                }

                if let transcriptSegments = viewModel.entry.transcriptSegments,
                   !transcriptSegments.isEmpty {
                    detailSection(title: localized("Transcript Segments")) {
                        ForEach(transcriptSegments) { segment in
                            detailLine(
                                label: "\(TranscriptFormatter.timestampString(for: segment.startSeconds)) · \(segment.displaySpeakerTitle)",
                                value: segment.text
                            )
                        }
                    }
                }

                if hasDictionaryActivity {
                    detailSection(title: localized("Dictionary")) {
                        if !viewModel.entry.dictionaryHitTerms.isEmpty {
                            detailTagGroup(
                                title: localized("Matched dictionary terms"),
                                values: viewModel.entry.dictionaryHitTerms
                            )
                        }
                        if !viewModel.entry.dictionaryCorrectedTerms.isEmpty {
                            detailTagGroup(
                                title: localized("Corrected terms"),
                                values: viewModel.entry.dictionaryCorrectedTerms
                            )
                        }
                        if !viewModel.entry.dictionarySuggestedTerms.isEmpty {
                            detailTagGroup(
                                title: localized("Suggested terms"),
                                values: viewModel.entry.dictionarySuggestedTerms.map(\.term)
                            )
                        }
                    }
                }
            }
            .padding(.horizontal, contentPadding)
            .padding(.vertical, verticalPadding)
            .frame(width: preferredContentWidth, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onChange(of: viewModel.audioURL?.path) { _, _ in
            playbackController.loadAudio(viewModel.audioURL)
        }
        .overlay(alignment: .top) {
            if !viewModel.toastMessage.isEmpty {
                ModelDebugToast(message: viewModel.toastMessage) {
                    viewModel.dismissToast()
                }
                .padding(.top, 12)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
    }

    @ViewBuilder
    private func detailSection<Trailing: View, Content: View>(
        title: String,
        @ViewBuilder trailing: () -> Trailing,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: sectionSpacing) {
            HStack(alignment: .center, spacing: 10) {
                Text(title)
                    .font(sectionTitleFont)
                    .foregroundStyle(.primary)

                Spacer(minLength: 8)

                trailing()
            }

            VStack(alignment: .leading, spacing: 10) {
                content()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(style == .popover ? 0 : 12)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(sectionCardFill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(style == .popover ? Color.clear : Color(nsColor: .quaternaryLabelColor).opacity(0.18), lineWidth: 1)
            )
        }
    }

    @ViewBuilder
    private func detailSection<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        detailSection(title: title, trailing: { EmptyView() }, content: content)
    }

    private func detailLine(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)

            Text(verbatim: value)
                .font(.subheadline)
                .foregroundStyle(.primary)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private func optionalDetailLine(label: String, value: String?) -> some View {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmed.isEmpty {
            detailLine(label: label, value: trimmed)
        }
    }

    private func detailTagGroup(title: String, values: [String]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)

            Text(values.joined(separator: ", "))
                .font(.subheadline)
                .foregroundStyle(.primary)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
    }

    private var correctionPreviewText: Text {
        HistoryCorrectionPresentation.segments(
            for: viewModel.entry.text,
            snapshots: viewModel.entry.dictionaryCorrectionSnapshots
        ).reduce(Text("")) { partial, segment in
            partial + styledText(for: segment)
        }
    }

    private func styledText(for segment: HistoryCorrectionSegment) -> Text {
        switch segment {
        case .plain(let value):
            return Text(verbatim: value).foregroundColor(.primary)
        case .original(let value):
            return Text(verbatim: value)
                .foregroundColor(.red)
                .strikethrough(true, color: .red)
        case .corrected(let value):
            return Text(verbatim: value)
                .foregroundColor(.green)
                .fontWeight(.semibold)
        }
    }

    private func historyKindTitle(_ kind: TranscriptionHistoryKind) -> String {
        switch kind {
        case .normal:
            return localized("Transcription")
        case .translation:
            return localized("Translation")
        case .rewrite:
            return localized("Rewrite")
        case .transcript:
            return localized("Transcript")
        }
    }

    private func modeMetadataLabel(for kind: TranscriptionHistoryKind) -> String {
        switch kind {
        case .normal:
            return localized("Enhancement")
        case .translation:
            return localized("Translation Mode")
        case .rewrite:
            return localized("Rewrite Mode")
        case .transcript:
            return localized("Summary Mode")
        }
    }

    private func modelMetadataLabel(for kind: TranscriptionHistoryKind) -> String {
        switch kind {
        case .normal:
            return localized("Enhancer Model")
        case .translation:
            return localized("Translation Model")
        case .rewrite:
            return localized("Rewrite Model")
        case .transcript:
            return localized("Summary Model")
        }
    }

    private func formattedDate(_ date: Date) -> String {
        date.formatted(
            .dateTime
                .locale(AppLocalization.locale)
                .year()
                .month(.abbreviated)
                .day()
                .hour()
                .minute()
                .second()
        )
    }

    private func formattedDuration(_ seconds: TimeInterval?) -> String? {
        guard let seconds else { return nil }
        if seconds < 1 {
            let format = localized("%d ms")
            return String(format: format, locale: AppLocalization.locale, Int(seconds * 1000))
        }
        if seconds < 60 {
            let format = localized("%.1f s")
            return String(format: format, locale: AppLocalization.locale, seconds)
        }
        let minutes = Int(seconds) / 60
        let remain = Int(seconds) % 60
        let format = localized("%dm %ds")
        return String(format: format, locale: AppLocalization.locale, minutes, remain)
    }

    private func timeRangeLabel(for timing: WhisperHistoryWordTiming) -> String {
        String(
            format: localized("%.2fs → %.2fs"),
            locale: AppLocalization.locale,
            timing.startSeconds,
            timing.endSeconds
        )
    }
}
