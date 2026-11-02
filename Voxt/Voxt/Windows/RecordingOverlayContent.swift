// RecordingOverlayContent.swift
// Provides Recording Overlay Content for window and overlay UI.

import SwiftUI

// MARK: - SwiftUI content hosted inside the panel

struct OverlayContent: View {
    @ObservedObject var state: OverlayState
    let onInject: () -> Void
    let onContinue: () -> Void
    let onToggleConversationRecording: () -> Void
    let onShowDetail: () -> Void
    let onClose: () -> Void
    let onToggleSessionTranslationTargetPicker: () -> Void
    let onSelectSessionTranslationTargetLanguage: (TranslationTargetLanguage) -> Void
    let onDismissSessionTranslationTargetPicker: () -> Void

    var body: some View {
        WaveformView(
            displayMode: state.displayMode,
            sessionIconMode: state.sessionIconMode,
            isModelInitializing: state.isModelInitializing,
            initializingEngine: state.initializingEngine,
            audioLevel: state.audioLevel,
            isRecording: state.isRecording,
            shouldAnimate: state.shouldAnimateVisuals,
            transcribedText: state.transcribedText,
            statusMessage: state.statusMessage,
            isEnhancing: state.isEnhancing,
            isRequesting: state.isRequesting,
            isFinalizingTranscription: state.isFinalizingTranscription,
            isCompleting: state.isCompleting,
            answerTitle: state.answerTitle,
            answerContent: state.answerContent,
            isStreamingAnswer: state.isStreamingAnswer,
            answerInteractionMode: state.answerInteractionMode,
            rewriteConversationTurns: state.rewriteConversationTurns,
            latestRewriteResult: state.latestRewriteResult,
            canInjectAnswer: state.canInjectAnswer,
            canCopyAnswer: state.canCopyLatestAnswer,
            canContinueAnswer: state.showsRewriteContinueButton,
            canShowHistoryDetail: state.canShowLatestHistoryDetail,
            compactLeadingIconImage: state.compactLeadingIconImage,
            sessionTranslationTargetLanguage: state.sessionTranslationTargetLanguage,
            sessionTranslationDraftLanguage: state.sessionTranslationDraftLanguage,
            isSessionTranslationTargetPickerPresented: state.isSessionTranslationTargetPickerPresented,
            isSessionTranslationLanguageHovering: state.isSessionTranslationLanguageHovering,
            allowsSessionTranslationLanguageSwitching: state.allowsSessionTranslationLanguageSwitching,
            onInject: onInject,
            onContinue: onContinue,
            onToggleConversationRecording: onToggleConversationRecording,
            onShowHistoryDetail: onShowDetail,
            onClose: onClose,
            onSessionTranslationLanguageHoverChanged: state.setSessionTranslationLanguageHovering,
            onToggleSessionTranslationTargetPicker: onToggleSessionTranslationTargetPicker,
            onSelectSessionTranslationTargetLanguage: onSelectSessionTranslationTargetLanguage,
            onDismissSessionTranslationTargetPicker: onDismissSessionTranslationTargetPicker
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.top, 8)
    }
}
