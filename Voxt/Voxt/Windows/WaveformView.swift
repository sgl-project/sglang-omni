// WaveformView.swift
// Provides Waveform View for window and overlay UI.

import SwiftUI
import Foundation
import AppKit

private struct TranscriptWidthPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

struct WaveformView: View {
    @AppStorage(AppPreferenceKey.overlayCardOpacity) private var overlayCardOpacity = 82
    @AppStorage(AppPreferenceKey.overlayCardCornerRadius) private var overlayCardCornerRadius = 24
    @AppStorage(AppPreferenceKey.realtimeTextDisplayEnabled) private var realtimeTextDisplayEnabled = true
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue

    static let defaultWaveformBarWidth: CGFloat = 3.2
    static let defaultWaveformBarSpacing: CGFloat = 2.5
    static let defaultWaveformSlotWidth: CGFloat = 94
    static let defaultSessionLanguagePickerWidth: CGFloat = 72

    static func waveformVisualWidth(
        barCount: Int = 16,
        barWidth: CGFloat = Self.defaultWaveformBarWidth,
        barSpacing: CGFloat = Self.defaultWaveformBarSpacing
    ) -> CGFloat {
        guard barCount > 0 else { return 0 }
        return (CGFloat(barCount) * barWidth) + (CGFloat(barCount - 1) * barSpacing)
    }

    static func shouldShowSessionTranslationLanguagePill(
        displayMode: OverlayDisplayMode,
        allowsSwitching: Bool,
        sessionTranslationTargetLanguage: TranslationTargetLanguage?,
        isHovering: Bool,
        isPickerPresented: Bool
    ) -> Bool {
        displayMode == .recording &&
            allowsSwitching &&
            sessionTranslationTargetLanguage != nil &&
            (isHovering || isPickerPresented)
    }

    private let waveformBarWidth: CGFloat = Self.defaultWaveformBarWidth
    private let waveformBarSpacing: CGFloat = Self.defaultWaveformBarSpacing
    var displayMode: OverlayDisplayMode
    var sessionIconMode: OverlaySessionIconMode
    var isModelInitializing: Bool = false
    var initializingEngine: TranscriptionEngine? = nil
    var audioLevel: Float
    var isRecording: Bool
    var shouldAnimate: Bool
    var transcribedText: String
    var statusMessage: String = ""
    var isEnhancing: Bool = false
    var isRequesting: Bool = false
    var isFinalizingTranscription: Bool = false
    var isCompleting: Bool = false
    var answerTitle: String = ""
    var answerContent: String = ""
    var isStreamingAnswer: Bool = false
    var answerInteractionMode: AnswerInteractionMode = .singleResult
    var rewriteConversationTurns: [RewriteConversationTurn] = []
    var latestRewriteResult: RewriteAnswerPayload? = nil
    var canInjectAnswer: Bool = false
    var canCopyAnswer: Bool = false
    var canContinueAnswer: Bool = false
    var canShowHistoryDetail: Bool = false
    var compactLeadingIconImage: NSImage? = nil
    var sessionTranslationTargetLanguage: TranslationTargetLanguage? = nil
    var sessionTranslationDraftLanguage: TranslationTargetLanguage? = nil
    var isSessionTranslationTargetPickerPresented: Bool = false
    var isSessionTranslationLanguageHovering: Bool = false
    var allowsSessionTranslationLanguageSwitching: Bool = false
    var onInject: () -> Void = {}
    var onContinue: () -> Void = {}
    var onToggleConversationRecording: () -> Void = {}
    var onShowHistoryDetail: () -> Void = {}
    var onClose: () -> Void = {}
    var onSessionTranslationLanguageHoverChanged: (Bool) -> Void = { _ in }
    var onToggleSessionTranslationTargetPicker: () -> Void = {}
    var onSelectSessionTranslationTargetLanguage: (TranslationTargetLanguage) -> Void = { _ in }
    var onDismissSessionTranslationTargetPicker: () -> Void = {}

    private let iconSlotSize = CGSize(width: 16, height: 28)
    private let barAreaHeight: CGFloat = 28
    private let waveformSlotWidth: CGFloat = Self.defaultWaveformSlotWidth
    private let transcriptAnimationGrowthThreshold: CGFloat = 6
    private let barCount = 16
    @State private var appeared = false
    @State private var textScrollID = UUID()
    @State private var measuredTranscriptWidth: CGFloat = 0
    @State private var didCopyAnswer = false
    @State private var copyFeedbackToken = UUID()
    @StateObject private var waveformState = RecentAudioWaveformState(
        barCount: 16,
        historyDuration: 0.9,
        framesPerSecond: 20,
        silenceFloor: 0.01,
        peakHoldFrames: 1,
        peakDecayFactor: 0.74,
        riseSmoothing: 0.82,
        fallSmoothing: 0.24
    )

    private var displayText: String {
        let message = statusMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        if !message.isEmpty { return message }
        guard isAnswerMode || realtimeTextDisplayEnabled else { return "" }
        return sanitizedDisplayText(transcribedText)
    }

    private var hasText: Bool { !displayText.isEmpty }
    private var isAnswerMode: Bool { displayMode == .answer }
    private var isCompact: Bool { !hasText && !isAnswerMode }
    private var cornerRadius: CGFloat { CGFloat(min(max(overlayCardCornerRadius, 0), 40)) }
    private var cardOpacity: Double { Double(min(max(overlayCardOpacity, 0), 100)) / 100.0 }
    private var textOverflows: Bool { displayText.count > 38 }
    private var showsLoadingSpinner: Bool {
        !isCompleting && (isEnhancing || isRequesting || isFinalizingTranscription)
    }
    private var showsInitializationIcon: Bool { isModelInitializing && !showsLoadingSpinner }
    private var showsSessionTranslationLanguagePill: Bool {
        Self.shouldShowSessionTranslationLanguagePill(
            displayMode: displayMode,
            allowsSwitching: allowsSessionTranslationLanguageSwitching,
            sessionTranslationTargetLanguage: sessionTranslationTargetLanguage,
            isHovering: isSessionTranslationLanguageHovering,
            isPickerPresented: isSessionTranslationTargetPickerPresented
        )
    }
    private var selectedSessionTranslationLanguage: TranslationTargetLanguage? {
        sessionTranslationDraftLanguage ?? sessionTranslationTargetLanguage
    }
    private var showsAnswerTranslationSelector: Bool {
        isAnswerMode &&
            answerInteractionMode == .singleResult &&
            sessionIconMode == .translation &&
            allowsSessionTranslationLanguageSwitching &&
            sessionTranslationTargetLanguage != nil
    }
    private var waveformVisualWidth: CGFloat {
        Self.waveformVisualWidth(
            barCount: barCount,
            barWidth: waveformBarWidth,
            barSpacing: waveformBarSpacing
        )
    }
    private var displayedAnswerPayload: RewriteAnswerPayload {
        let hasDraft = !answerTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            !answerContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if hasDraft {
            return RewriteAnswerPayload(title: answerTitle, content: answerContent)
        }
        return latestRewriteResult ?? RewriteAnswerPayload(title: answerTitle, content: answerContent)
    }

    private var copyableAnswerPayload: RewriteAnswerPayload? {
        if let latestRewriteResult {
            return latestRewriteResult
        }
        guard !isStreamingAnswer else { return nil }
        let trimmed = answerContent.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return RewriteAnswerPayload(title: answerTitle, content: answerContent)
    }

    var body: some View {
        let _ = interfaceLanguageRaw
        VStack(alignment: .leading, spacing: 10) {
            cardView
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .scaleEffect(appeared ? 1.0 : 0.5, anchor: .bottom)
        .opacity(appeared ? 1.0 : 0.0)
        .animation(.spring(response: 0.35, dampingFraction: 0.5, blendDuration: 0.1), value: appeared)
        .onHover { hovering in
            guard allowsSessionTranslationLanguageSwitching else { return }
            onSessionTranslationLanguageHoverChanged(hovering)
        }
        .onAppear {
            updateWaveformStateActivity()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.02) {
                appeared = true
            }
        }
        .onDisappear {
            waveformState.setActive(false)
            appeared = false
        }
        .onChange(of: shouldAnimate) {
            updateWaveformStateActivity()
        }
        .onChange(of: displayMode) {
            updateWaveformStateActivity()
        }
        .onChange(of: isRecording) {
            updateWaveformStateActivity()
        }
        .onChange(of: isModelInitializing) {
            updateWaveformStateActivity()
        }
        .onChange(of: showsLoadingSpinner) {
            updateWaveformStateActivity()
        }
        .onChange(of: audioLevel) {
            waveformState.ingest(level: emphasizedWaveformInputLevel(audioLevel))
        }
    }

    private var cardView: some View {
        Group {
            if isAnswerMode {
                answerCard
                    .transition(.opacity.combined(with: .scale(scale: 0.96, anchor: .top)))
            } else {
                compactCard
                    .transition(.opacity.combined(with: .scale(scale: 0.98, anchor: .top)))
            }
        }
        .padding(.horizontal, isAnswerMode ? 18 : (isCompact ? 14 : 20))
        .padding(.vertical, isAnswerMode ? 16 : (isCompact ? 10 : 12))
        .background(cardBackground)
        .compositingGroup()
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .animation(.spring(response: 0.38, dampingFraction: 0.78), value: displayMode)
        .animation(.spring(response: 0.4, dampingFraction: 0.55, blendDuration: 0.1), value: isCompact)
        .frame(maxWidth: .infinity, alignment: .top)
        .contentShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }

    private var compactCard: some View {
        VStack(spacing: isCompact ? 0 : 8) {
            HStack(spacing: 10) {
                leadingStatusIcon
                    .frame(width: iconSlotSize.width, height: iconSlotSize.height, alignment: .center)
                    .transition(.opacity)

                waveformSlot
            }
            .frame(height: barAreaHeight)
            .animation(.easeInOut(duration: 0.25), value: showsLoadingSpinner)
            .animation(.spring(response: 0.22, dampingFraction: 0.82), value: showsSessionTranslationLanguagePill)

            if hasText {
                ScrollViewReader { proxy in
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 0) {
                            Text(displayText)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(.white.opacity(0.85))
                                .lineLimit(1)
                                .fixedSize(horizontal: true, vertical: false)
                                .background(
                                    GeometryReader { geometry in
                                        Color.clear.preference(
                                            key: TranscriptWidthPreferenceKey.self,
                                            value: geometry.size.width
                                        )
                                    }
                                )
                                .id(textScrollID)

                            Spacer().frame(width: 4)
                        }
                    }
                    .frame(maxWidth: 260)
                    .mask(
                        HStack(spacing: 0) {
                            if textOverflows {
                                LinearGradient(
                                    colors: [.clear, .white],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                                .frame(width: 16)
                                .transition(.opacity)
                            }
                            Color.white
                        }
                    )
                    .onPreferenceChange(TranscriptWidthPreferenceKey.self) { width in
                        measuredTranscriptWidth = width
                    }
                    .onChange(of: displayText) {
                        let previousWidth = measuredTranscriptWidth
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                            let grew = measuredTranscriptWidth > previousWidth + transcriptAnimationGrowthThreshold
                            let shouldAnimate = previousWidth > 0 && grew
                            scrollTranscriptToTrailing(proxy, animated: shouldAnimate)
                        }
                    }
                }
                .transition(.opacity)
            }
        }
    }

    @ViewBuilder
    private var waveformSlot: some View {
        if showsLoadingSpinner {
            processingBars
                .frame(width: waveformVisualWidth, height: barAreaHeight, alignment: .center)
                .frame(width: waveformSlotWidth, height: barAreaHeight, alignment: .center)
                .transition(.opacity)
        } else if showsSessionTranslationLanguagePill {
            ZStack {
                Color.clear

                sessionTranslationLanguagePill
            }
            .frame(width: waveformSlotWidth, height: barAreaHeight, alignment: .center)
            .transition(.opacity.combined(with: .scale(scale: 0.92)))
        } else {
            waveformBars
                .frame(width: waveformVisualWidth, height: barAreaHeight, alignment: .center)
                .frame(width: waveformSlotWidth, height: barAreaHeight, alignment: .center)
                .transition(.opacity)
        }
    }

    private var sessionTranslationLanguagePill: some View {
        AnswerSessionTranslationMenuPicker(
            selectedLanguage: selectedSessionTranslationLanguage,
            isPresented: isSessionTranslationTargetPickerPresented,
            onTogglePresentation: onToggleSessionTranslationTargetPicker,
            onDismissPresentation: onDismissSessionTranslationTargetPicker,
            onSelectLanguage: onSelectSessionTranslationTargetLanguage,
            style: .compact
        )
    }

    private var answerCard: some View {
        WaveformAnswerCard(
            title: displayedAnswerPayload.title,
            content: displayedAnswerPayload.content,
            answerInteractionMode: answerInteractionMode,
            conversationTurns: rewriteConversationTurns,
            streamingUserPromptText: conversationStreamingUserPromptText,
            canInjectAnswer: canInjectAnswer,
            canCopyAnswer: canCopyAnswer,
            canContinueAnswer: canContinueAnswer,
            canShowHistoryDetail: showsAnswerTranslationSelector ? false : canShowHistoryDetail,
            didCopyAnswer: didCopyAnswer,
            isRecording: isRecording,
            isProcessing: isEnhancing || isRequesting || isFinalizingTranscription,
            audioLevel: audioLevel,
            shouldAnimateWave: shouldAnimate,
            streamingDraftPayload: answerInteractionMode == .conversation && isStreamingAnswer
                ? displayedAnswerPayload
                : nil,
            showsSessionTranslationSelector: showsAnswerTranslationSelector,
            sessionTranslationTargetLanguage: sessionTranslationTargetLanguage,
            sessionTranslationDraftLanguage: sessionTranslationDraftLanguage,
            isSessionTranslationTargetPickerPresented: isSessionTranslationTargetPickerPresented,
            onInject: onInject,
            onContinue: onContinue,
            onToggleConversationRecording: onToggleConversationRecording,
            onShowDetail: onShowHistoryDetail,
            onCopy: copyAnswerToPasteboard,
            onClose: onClose,
            onToggleSessionTranslationTargetPicker: onToggleSessionTranslationTargetPicker,
            onSelectSessionTranslationTargetLanguage: onSelectSessionTranslationTargetLanguage,
            onDismissSessionTranslationTargetPicker: onDismissSessionTranslationTargetPicker
        )
    }

    private var conversationStreamingUserPromptText: String? {
        guard answerInteractionMode == .conversation else { return nil }
        let trimmed = transcribedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, (isRecording || isEnhancing || isRequesting || isFinalizingTranscription) else { return nil }
        return trimmed
    }

    @ViewBuilder
    private var leadingStatusIcon: some View {
        WaveformCompactLeadingStatusIconView(
            isCompleting: isCompleting,
            showsInitializationIcon: showsInitializationIcon,
            shouldAnimate: shouldAnimate,
            compactLeadingIconImage: compactLeadingIconImage,
            sessionIconMode: sessionIconMode,
            displayMode: displayMode
        )
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .fill(.black.opacity(cardOpacity))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
            )
    }

    private func sanitizedDisplayText(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        if !(trimmed.hasPrefix("{") || trimmed.hasPrefix("[")) {
            return trimmed
        }

        if let data = trimmed.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data),
           let text = extractText(from: object),
           !text.isEmpty {
            return text
        }

        if let text = extractLooseText(from: trimmed), !text.isEmpty {
            return text
        }

        return trimmed
    }

    private func extractText(from object: Any) -> String? {
        if let value = object as? String {
            return value.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let dict = object as? [String: Any] {
            for key in ["text", "transcript", "delta", "result_text", "content"] {
                if let value = dict[key], let extracted = extractText(from: value), !extracted.isEmpty {
                    return extracted
                }
            }
            for value in dict.values {
                if let extracted = extractText(from: value), !extracted.isEmpty {
                    return extracted
                }
            }
        }
        if let array = object as? [Any] {
            for value in array {
                if let extracted = extractText(from: value), !extracted.isEmpty {
                    return extracted
                }
            }
        }
        return nil
    }

    private func extractLooseText(from value: String) -> String? {
        let patterns = [
            #"(?:["']?text["']?\s*:\s*["'])([^"']+)(?:["'])"#,
            #"(?:["']?transcript["']?\s*:\s*["'])([^"']+)(?:["'])"#,
            #"(?:["']?delta["']?\s*:\s*["'])([^"']+)(?:["'])"#,
            #"(?:["']?text["']?\s*:\s*)([^,}\]]+)"#,
            #"(?:["']?transcript["']?\s*:\s*)([^,}\]]+)"#,
            #"(?:["']?delta["']?\s*:\s*)([^,}\]]+)"#
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
                continue
            }
            let range = NSRange(value.startIndex..<value.endIndex, in: value)
            guard let match = regex.firstMatch(in: value, options: [], range: range),
                  match.numberOfRanges > 1,
                  let textRange = Range(match.range(at: 1), in: value) else {
                continue
            }
            var result = String(value[textRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if (result.hasPrefix("\"") && result.hasSuffix("\"")) ||
                (result.hasPrefix("'") && result.hasSuffix("'")) {
                result.removeFirst()
                result.removeLast()
                result = result.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if !result.isEmpty { return result }
        }
        return nil
    }

    private var waveformBars: some View {
        HStack(alignment: .center, spacing: waveformBarSpacing) {
            ForEach(0..<barCount, id: \.self) { index in
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.98), Color.white.opacity(0.80)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: waveformBarWidth, height: waveformBarHeight(for: index))
                    .shadow(color: .white.opacity(waveformGlowOpacity(for: index)), radius: 3, x: 0, y: 0)
            }
        }
        .frame(height: barAreaHeight)
    }

    private var processingBars: some View {
        WaveformProcessingLoaderView(
            isAnimating: shouldAnimate,
            itemCount: 5,
            itemSize: CGSize(width: 5, height: 5),
            spacing: 4,
            color: .white
        )
        .frame(height: barAreaHeight)
    }

    private func barHeight(for index: Int) -> CGFloat {
        let minH: CGFloat = 4
        let phase = Double(index) * 0.4
        let sine = (sin(phase) + 1) / 2

        if isModelInitializing {
            let quietPattern: [CGFloat] = [4.0, 4.7, 5.4, 6.0, 5.0, 4.3, 5.2, 5.8]
            return quietPattern[index % quietPattern.count]
        }

        if isRecording {
            let level = waveformState.barLevels.indices.contains(index) ? waveformState.barLevels[index] : audioLevel
            return WaveformBarVisuals.barHeight(level: level, minHeight: 3.2, maxHeight: 22)
        }

        if displayMode == .processing {
            return minH + CGFloat(3.5 * sine)
        }

        return staticBarHeight(for: index)
    }

    private func glowOpacity(for index: Int) -> Double {
        if isModelInitializing {
            return 0.03
        }
        guard isRecording else { return 0.08 }
        let level = waveformState.barLevels.indices.contains(index) ? waveformState.barLevels[index] : audioLevel
        return WaveformBarVisuals.glowOpacity(level: level, base: 0.04, gain: 0.26, cap: 0.24)
    }

    private func normalizedAudioLevel(_ raw: Float) -> CGFloat {
        let clamped = max(0, min(raw, 1))
        let gained = min(1.0, pow(Double(clamped), 1.08) * 0.56)
        return CGFloat(gained)
    }

    private func waveformBarHeight(for index: Int) -> CGFloat {
        barHeight(for: index)
    }

    private func waveformGlowOpacity(for index: Int) -> Double {
        glowOpacity(for: index)
    }

    private var shouldDriveWaveformHistory: Bool {
        shouldAnimate && isRecording && !showsLoadingSpinner && !isModelInitializing
    }

    private func updateWaveformStateActivity() {
        waveformState.setActive(shouldDriveWaveformHistory)
    }

    private func emphasizedWaveformInputLevel(_ level: Float) -> Float {
        let clamped = max(0, min(level, 1))
        let expanded = min(1.0, pow(Double(clamped), 0.72) * 1.24)
        return Float(expanded)
    }

    private func staticBarHeight(for index: Int) -> CGFloat {
        let pattern: [CGFloat] = [5, 7, 10, 12, 9, 6, 8, 11]
        return pattern[index % pattern.count]
    }

    private func copyAnswerToPasteboard() {
        guard let payload = copyableAnswerPayload else { return }
        copyTextToPasteboard(payload.content)
        let token = UUID()
        copyFeedbackToken = token
        didCopyAnswer = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            guard copyFeedbackToken == token else { return }
            didCopyAnswer = false
        }
    }

    private func copyTextToPasteboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    private func scrollTranscriptToTrailing(_ proxy: ScrollViewProxy, animated: Bool) {
        if animated {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(textScrollID, anchor: .trailing)
            }
        } else {
            proxy.scrollTo(textScrollID, anchor: .trailing)
        }
    }
}
