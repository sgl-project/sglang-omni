// WaveformAnswerCardComponents.swift
// Provides Waveform Answer Card Components for window and overlay UI.

import SwiftUI
import AppKit

struct RewriteConversationBottomVisibilityPreferenceKey: PreferenceKey {
    static var defaultValue = true

    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = nextValue()
    }
}

struct AnswerConversationBodyView: View {
    private let conversationBottomAnchorID = "rewrite-conversation-bottom-anchor"

    let conversationTurns: [RewriteConversationTurn]
    let streamingUserPromptText: String?
    let streamingDraftPayload: RewriteAnswerPayload?
    let isProcessing: Bool

    @State private var isScrolledToConversationBottom = true
    @State private var wasScrolledToConversationBottom = true
    @State private var hasUnreadConversationMessages = false
    @State private var pendingScrollRequestToken = UUID()

    var body: some View {
        GeometryReader { outerProxy in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottomTrailing) {
                    ScrollView(.vertical, showsIndicators: true) {
                        LazyVStack(alignment: .leading, spacing: 14) {
                            ForEach(conversationTurns) { turn in
                                if !turn.userPromptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                    HStack {
                                        Spacer(minLength: 48)
                                        conversationBubble(
                                            title: AppLocalization.localizedString("You"),
                                            content: turn.userPromptText,
                                            alignment: .trailing,
                                            isUser: true
                                        )
                                    }
                                }

                                HStack {
                                    conversationBubble(
                                        title: assistantPayload(for: turn).title,
                                        content: assistantPayload(for: turn).content,
                                        alignment: .leading,
                                        isUser: false
                                    )
                                    Spacer(minLength: 48)
                                }
                                .id(turn.id)
                            }

                            if let streamingUserPromptText,
                               !streamingUserPromptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                HStack {
                                    Spacer(minLength: 48)
                                    conversationBubble(
                                        title: AppLocalization.localizedString("You"),
                                        content: streamingUserPromptText,
                                        alignment: .trailing,
                                        isUser: true,
                                        isStreaming: true
                                    )
                                }
                                .id("streaming-user-draft")
                            }

                            if let streamingDraftPayload,
                               !streamingDraftPayload.trimmedTitle.isEmpty || !streamingDraftPayload.trimmedContent.isEmpty || isProcessing {
                                HStack {
                                    conversationBubble(
                                        title: streamingDraftPayload.title,
                                        content: streamingDraftPayload.trimmedContent.isEmpty ? "…" : streamingDraftPayload.content,
                                        alignment: .leading,
                                        isUser: false,
                                        isStreaming: true
                                    )
                                    Spacer(minLength: 48)
                                }
                                .id("streaming-draft")
                            } else if isProcessing {
                                HStack {
                                    conversationBubble(
                                        title: "",
                                        content: "…",
                                        alignment: .leading,
                                        isUser: false,
                                        isStreaming: true
                                    )
                                    Spacer(minLength: 48)
                                }
                                .id("streaming-placeholder")
                            }

                            GeometryReader { geo in
                                Color.clear
                                    .preference(
                                        key: RewriteConversationBottomVisibilityPreferenceKey.self,
                                        value: abs(geo.frame(in: .named("RewriteConversationScroll")).maxY - outerProxy.size.height) < 36
                                    )
                            }
                            .frame(height: 1)
                            .id(conversationBottomAnchorID)
                        }
                        .padding(.trailing, 10)
                    }
                    .coordinateSpace(name: "RewriteConversationScroll")
                    .frame(maxWidth: .infinity, maxHeight: 220, alignment: .topLeading)
                    .onPreferenceChange(RewriteConversationBottomVisibilityPreferenceKey.self) { isVisible in
                        wasScrolledToConversationBottom = isScrolledToConversationBottom
                        isScrolledToConversationBottom = isVisible
                        if isVisible {
                            hasUnreadConversationMessages = false
                        }
                    }
                    .onAppear {
                        scrollConversationToBottom(using: proxy)
                    }
                    .onChange(of: conversationTurns.count) { oldValue, newValue in
                        guard newValue > oldValue else { return }
                        handleConversationMessagesUpdate(using: proxy, animated: true)
                    }
                    .onChange(of: isProcessing) { oldValue, newValue in
                        guard newValue, newValue != oldValue else { return }
                        handleConversationMessagesUpdate(using: proxy, forceScroll: true, animated: true)
                    }
                    .onChange(of: streamingDraftPayload?.content ?? "") { _, _ in
                        handleConversationMessagesUpdate(using: proxy, animated: false)
                    }
                    .onChange(of: streamingDraftPayload?.trimmedTitle ?? "") { oldValue, newValue in
                        guard oldValue != newValue else { return }
                        handleConversationMessagesUpdate(
                            using: proxy,
                            forceScroll: !newValue.isEmpty && oldValue.isEmpty,
                            animated: oldValue.isEmpty && !newValue.isEmpty
                        )
                    }
                    .onChange(of: streamingUserPromptText ?? "") { oldValue, newValue in
                        guard oldValue != newValue else { return }
                        handleConversationMessagesUpdate(
                            using: proxy,
                            forceScroll: !newValue.isEmpty && oldValue.isEmpty,
                            animated: oldValue.isEmpty && !newValue.isEmpty
                        )
                    }

                    if hasUnreadConversationMessages {
                        Button {
                            hasUnreadConversationMessages = false
                            scrollConversationToBottom(using: proxy)
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.down")
                                    .font(.system(size: 10, weight: .semibold))
                                Text(AppLocalization.localizedString("Latest"))
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .foregroundStyle(.white.opacity(0.92))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(.black.opacity(0.78))
                            )
                            .overlay(
                                Capsule(style: .continuous)
                                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                        .padding(.trailing, 8)
                        .padding(.bottom, 4)
                    }
                }
            }
        }
    }

    private func handleConversationMessagesUpdate(
        using proxy: ScrollViewProxy,
        forceScroll: Bool = false,
        animated: Bool = false
    ) {
        if forceScroll || isScrolledToConversationBottom || wasScrolledToConversationBottom {
            hasUnreadConversationMessages = false
            scrollConversationToBottom(using: proxy, animated: animated)
        } else {
            hasUnreadConversationMessages = true
        }
    }

    private func scrollConversationToBottom(using proxy: ScrollViewProxy, animated: Bool = true) {
        let token = UUID()
        pendingScrollRequestToken = token
        DispatchQueue.main.async {
            guard token == pendingScrollRequestToken else { return }
            if animated {
                withAnimation(.easeOut(duration: 0.18)) {
                    proxy.scrollTo(conversationBottomAnchorID, anchor: .bottom)
                }
            } else {
                proxy.scrollTo(conversationBottomAnchorID, anchor: .bottom)
            }
        }
    }

    private func assistantPayload(for turn: RewriteConversationTurn) -> RewriteAnswerPayload {
        let rawPayload = RewriteAnswerPayload(
            title: turn.resultTitle,
            content: turn.resultContent
        )
        if rawPayload.trimmedTitle.isEmpty { return rawPayload }
        return RewriteAnswerPayloadParser.normalize(rawPayload)
    }

    private func conversationBubble(
        title: String,
        content: String,
        alignment: Alignment,
        isUser: Bool,
        isStreaming: Bool = false
    ) -> some View {
        RewriteConversationBubble(
            title: title,
            content: content,
            alignment: alignment,
            isUser: isUser,
            isStreaming: isStreaming
        )
    }
}

struct AnswerHeaderActionButton<Label: View>: View {
    let accessibilityLabel: String
    let action: () -> Void
    let isEnabled: Bool
    @ViewBuilder let label: () -> Label

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            label()
                .frame(width: 24, height: 24)
                .opacity(isEnabled ? 1 : 0.42)
                .background(
                    Circle()
                        .fill(isEnabled ? (isHovered ? .white.opacity(0.16) : .white.opacity(0.08)) : .white.opacity(0.04))
                )
                .overlay(
                    Circle()
                        .strokeBorder(.white.opacity(isEnabled && isHovered ? 0.18 : 0), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .accessibilityLabel(Text(accessibilityLabel))
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }
}

struct RewriteConversationBubble: View {
    let title: String
    let content: String
    let alignment: Alignment
    let isUser: Bool
    let isStreaming: Bool

    @State private var isHovered = false
    @State private var didCopy = false
    @State private var copyFeedbackToken = UUID()

    private var resolvedTitle: String {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? AppLocalization.localizedString("AI Answer") : trimmed
    }

    private var copyText: String {
        let trimmedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedContent.isEmpty else { return "" }
        guard !isUser, !trimmedTitle.isEmpty else { return trimmedContent }
        return "\(trimmedTitle)\n\n\(trimmedContent)"
    }

    var body: some View {
        VStack(alignment: isUser ? .trailing : .leading, spacing: 7) {
            if isUser || !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(resolvedTitle)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white.opacity(isUser ? 0.7 : (isStreaming ? 0.54 : 0.64)))
            }

            Group {
                if isStreaming {
                    Text(content)
                } else {
                    Text(content)
                        .textSelection(.enabled)
                }
            }
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(.white.opacity(isStreaming ? 0.84 : 0.92))
            .frame(maxWidth: .infinity, alignment: alignment)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: 340, alignment: alignment)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(isUser ? .white.opacity(0.1) : .white.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isUser ? .white.opacity(0.12) : .white.opacity(0.08), lineWidth: 1)
        )
        .overlay(alignment: .topTrailing) {
            if !isUser && isHovered {
                Button(action: copyToPasteboard) {
                    HStack(spacing: 4) {
                        Group {
                            if didCopy {
                                CopySuccessIconView()
                            } else {
                                CopyIconView()
                            }
                        }
                        .frame(width: 12, height: 12)

                        Text(didCopy ? AppLocalization.localizedString("Copied") : AppLocalization.localizedString("Copy"))
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .foregroundStyle(.white.opacity(0.94))
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(
                        Capsule(style: .continuous)
                            .fill(.black.opacity(0.74))
                    )
                    .overlay(
                        Capsule(style: .continuous)
                            .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
                .padding(.top, 8)
                .padding(.trailing, 8)
                .transition(.opacity.combined(with: .scale(scale: 0.94, anchor: .topTrailing)))
            }
        }
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }

    private func copyToPasteboard() {
        guard !copyText.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(copyText, forType: .string)
        copyFeedbackToken = UUID()
        let token = copyFeedbackToken
        didCopy = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            guard token == copyFeedbackToken else { return }
            didCopy = false
        }
    }
}

struct AnswerContinueButton: View {
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Text(AppLocalization.localizedString("Continue"))
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.94))
                .padding(.horizontal, 10)
                .frame(height: 24)
                .background(
                    Capsule()
                        .fill(isHovered ? .white.opacity(0.16) : .white.opacity(0.08))
                )
                .overlay(
                    Capsule()
                        .strokeBorder(.white.opacity(isHovered ? 0.18 : 0.1), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text(AppLocalization.localizedString("Continue")))
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }
}

struct AnswerConversationWaveView: View {
    let isRecording: Bool
    let isProcessing: Bool
    let audioLevel: Float
    let shouldAnimate: Bool

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

    var body: some View {
        SessionMiniWaveform(
            waveformState: waveformState,
            isSubdued: !isRecording || isProcessing
        )
        .scaleEffect(x: 0.84, y: 0.82, anchor: .leading)
        .onAppear {
            waveformState.setActive(shouldAnimate && isRecording && !isProcessing)
        }
        .onChange(of: shouldAnimate) {
            waveformState.setActive(shouldAnimate && isRecording && !isProcessing)
        }
        .onChange(of: isRecording) {
            waveformState.setActive(shouldAnimate && isRecording && !isProcessing)
        }
        .onChange(of: isProcessing) {
            waveformState.setActive(shouldAnimate && isRecording && !isProcessing)
        }
        .onChange(of: audioLevel) {
            waveformState.ingest(level: emphasizedWaveformInputLevel(audioLevel))
        }
        .onDisappear {
            waveformState.setActive(false)
        }
    }

    private func emphasizedWaveformInputLevel(_ level: Float) -> Float {
        let clamped = max(0, min(level, 1))
        let expanded = min(1.0, pow(Double(clamped), 0.72) * 1.24)
        return Float(expanded)
    }
}

private struct SessionMiniWaveform: View {
    @ObservedObject var waveformState: RecentAudioWaveformState
    var isSubdued = false

    var body: some View {
        HStack(alignment: .center, spacing: 2.5) {
            ForEach(0..<waveformState.barCount, id: \.self) { index in
                RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                    .fill(WaveformBarVisuals.barGradient)
                    .frame(width: 4, height: barHeight(for: index))
                    .shadow(color: .white.opacity(glowOpacity(for: index)), radius: 2.5, x: 0, y: 0)
            }
        }
        .frame(height: 28, alignment: .center)
    }

    private func barHeight(for index: Int) -> CGFloat {
        if isSubdued {
            let quietPattern: [CGFloat] = [3.2, 3.9, 4.6, 5.1, 4.2, 3.5, 4.4, 4.9]
            return quietPattern[index % quietPattern.count]
        }
        let baseLevel = waveformState.barLevels.indices.contains(index) ? waveformState.barLevels[index] : 0
        return WaveformBarVisuals.barHeight(
            level: baseLevel,
            minHeight: 2.5,
            maxHeight: 22
        )
    }

    private func glowOpacity(for index: Int) -> Double {
        if isSubdued {
            return 0.03
        }
        let baseLevel = waveformState.barLevels.indices.contains(index) ? waveformState.barLevels[index] : 0
        return WaveformBarVisuals.glowOpacity(level: baseLevel, base: 0.03, gain: 0.18, cap: 0.22)
    }
}
