// WaveformAnswerCard.swift
// Provides Waveform Answer Card for window and overlay UI.

import SwiftUI

struct WaveformAnswerCard: View {
    let title: String
    let content: String
    let answerInteractionMode: AnswerInteractionMode
    let conversationTurns: [RewriteConversationTurn]
    let streamingUserPromptText: String?
    let canInjectAnswer: Bool
    let canCopyAnswer: Bool
    let canContinueAnswer: Bool
    let canShowHistoryDetail: Bool
    let didCopyAnswer: Bool
    let isRecording: Bool
    let isProcessing: Bool
    let audioLevel: Float
    let shouldAnimateWave: Bool
    let streamingDraftPayload: RewriteAnswerPayload?
    let showsSessionTranslationSelector: Bool
    let sessionTranslationTargetLanguage: TranslationTargetLanguage?
    let sessionTranslationDraftLanguage: TranslationTargetLanguage?
    let isSessionTranslationTargetPickerPresented: Bool
    let onInject: () -> Void
    let onContinue: () -> Void
    let onToggleConversationRecording: () -> Void
    let onShowDetail: () -> Void
    let onCopy: () -> Void
    let onClose: () -> Void
    let onToggleSessionTranslationTargetPicker: () -> Void
    let onSelectSessionTranslationTargetLanguage: (TranslationTargetLanguage) -> Void
    let onDismissSessionTranslationTargetPicker: () -> Void

    private var displayTitle: String {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? AppLocalization.localizedString("AI Answer") : trimmed
    }

    private var showsTranslationLoadingIndicator: Bool {
        showsSessionTranslationSelector && isProcessing
    }

    private var isConversationMode: Bool {
        answerInteractionMode == .conversation
    }

    private var continueAction: () -> Void {
        isConversationMode ? onToggleConversationRecording : onContinue
    }

    private var selectedSessionTranslationLanguage: TranslationTargetLanguage? {
        sessionTranslationDraftLanguage ?? sessionTranslationTargetLanguage
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.bottom, 8)

            Rectangle()
                .fill(.white.opacity(0.08))
                .frame(height: 1)
                .padding(.bottom, 10)

            bodyContent
        }
    }

    @ViewBuilder
    private var header: some View {
        if isConversationMode {
            HStack(alignment: .center, spacing: 12) {
                TranscriptionModeIconView()
                    .frame(width: 20, height: 20)
                    .opacity(0.92)

                AnswerConversationWaveView(
                    isRecording: isRecording,
                    isProcessing: isProcessing,
                    audioLevel: audioLevel,
                    shouldAnimate: shouldAnimateWave
                )
                .frame(width: 96, height: 20, alignment: .leading)

                Spacer(minLength: 12)

                headerActions(showsContinue: canContinueAnswer)
            }
            .frame(minHeight: 24, alignment: .center)
        } else {
            HStack(alignment: .center, spacing: 12) {
                AnswerIconView()
                    .frame(width: 20, height: 20)

                Text(displayTitle)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .truncationMode(.tail)

                if showsTranslationLoadingIndicator {
                    LoadingSpinnerIconView(isAnimating: true)
                        .frame(width: 12, height: 12)
                        .opacity(0.88)
                }

                Spacer(minLength: 12)

                headerActions(showsContinue: canContinueAnswer)
            }
            .frame(minHeight: 24, alignment: .center)
        }
    }

    @ViewBuilder
    private func headerActions(showsContinue: Bool) -> some View {
        if showsSessionTranslationSelector {
            sessionTranslationSelector
        }

        if showsContinue {
            AnswerContinueButton(action: continueAction)
        }

        if canInjectAnswer {
            AnswerHeaderActionButton(
                accessibilityLabel: AppLocalization.localizedString("Inject into Current Input"),
                action: onInject,
                isEnabled: true
            ) {
                InjectAnswerIconView()
                    .frame(width: 15, height: 15)
                    .opacity(0.92)
            }
        }

        if canShowHistoryDetail && !showsSessionTranslationSelector {
            AnswerHeaderActionButton(
                accessibilityLabel: AppLocalization.localizedString("Detail"),
                action: onShowDetail,
                isEnabled: true
            ) {
                Image(systemName: "list.bullet.rectangle")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.9))
            }
        }

        AnswerHeaderActionButton(
            accessibilityLabel: AppLocalization.localizedString("Copy Answer"),
            action: onCopy,
            isEnabled: canCopyAnswer
        ) {
            if didCopyAnswer {
                CopySuccessIconView()
                    .frame(width: 15, height: 15)
            } else {
                CopyIconView()
                    .frame(width: 15, height: 15)
            }
        }

        AnswerHeaderActionButton(
            accessibilityLabel: AppLocalization.localizedString("Close"),
            action: onClose,
            isEnabled: true
        ) {
            Image(systemName: "xmark")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.9))
        }
    }

    private var sessionTranslationSelector: some View {
        AnswerSessionTranslationMenuPicker(
            selectedLanguage: selectedSessionTranslationLanguage,
            isPresented: isSessionTranslationTargetPickerPresented,
            onTogglePresentation: onToggleSessionTranslationTargetPicker,
            onDismissPresentation: onDismissSessionTranslationTargetPicker,
            onSelectLanguage: onSelectSessionTranslationTargetLanguage,
            fixedWidth: 82
        )
    }

    @ViewBuilder
    private var bodyContent: some View {
        if isConversationMode {
            conversationBody
        } else {
            singleResultBody
        }
    }

    private var singleResultBody: some View {
        ScrollView(.vertical, showsIndicators: true) {
            Text(content)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.white.opacity(0.9))
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
                .padding(.trailing, 10)
        }
        .frame(maxWidth: .infinity, maxHeight: 220, alignment: .topLeading)
    }

    private var conversationBody: some View {
        AnswerConversationBodyView(
            conversationTurns: conversationTurns,
            streamingUserPromptText: streamingUserPromptText,
            streamingDraftPayload: streamingDraftPayload,
            isProcessing: isProcessing
        )
        .frame(maxWidth: .infinity, maxHeight: 220, alignment: .topLeading)
    }
}
