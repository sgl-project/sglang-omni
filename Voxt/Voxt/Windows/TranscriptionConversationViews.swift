// TranscriptionConversationViews.swift
// Provides Transcription Conversation Views for window and overlay UI.

import AppKit
import SwiftUI

private struct TranscriptionDetailBottomVisibilityPreferenceKey: PreferenceKey {
    static var defaultValue = true

    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = nextValue()
    }
}

struct TranscriptionDetailConversationView: View {
    @ObservedObject var viewModel: TranscriptionDetailViewModel

    @State private var isScrolledToBottom = true
    @State private var wasScrolledToBottom = true
    @State private var hasUnreadMessages = false
    @State private var pendingScrollRequestToken = UUID()

    private let bottomAnchorID = "transcription-detail-bottom-anchor"

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
                .overlay(DetailPanelUIStyle.dividerColor)
            conversationBody
            composer
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 10) {
            Color.clear
                .frame(width: 62, height: 1)

            Text(viewModel.title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)

            Text(viewModel.headerMetaText)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 320, alignment: .trailing)
        }
        .padding(.horizontal, 20)
        .padding(.top, 12)
        .padding(.bottom, 10)
    }

    private var conversationBody: some View {
        GeometryReader { outerProxy in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottomTrailing) {
                    ScrollView {
                        VStack(spacing: 12) {
                            ForEach(viewModel.displayMessages) { message in
                                TranscriptionDetailMessageRow(message: message)
                            }

                            if viewModel.isLoading {
                                TranscriptionDetailLoadingRow()
                            }

                            if let errorMessage = viewModel.errorMessage,
                               !errorMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                Text(errorMessage)
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundStyle(.red.opacity(0.9))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.top, 2)
                            }

                            GeometryReader { geo in
                                Color.clear
                                    .preference(
                                        key: TranscriptionDetailBottomVisibilityPreferenceKey.self,
                                        value: abs(geo.frame(in: .named("TranscriptionDetailScroll")).maxY - outerProxy.size.height) < 36
                                    )
                            }
                            .frame(height: 1)
                            .id(bottomAnchorID)
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 16)
                    }
                    .coordinateSpace(name: "TranscriptionDetailScroll")
                    .onAppear {
                        scrollToBottom(using: proxy, animated: false)
                    }
                    .onPreferenceChange(TranscriptionDetailBottomVisibilityPreferenceKey.self) { isVisible in
                        wasScrolledToBottom = isScrolledToBottom
                        isScrolledToBottom = isVisible
                        if isVisible {
                            hasUnreadMessages = false
                        }
                    }
                    .onChange(of: viewModel.displayMessages.count) { oldValue, newValue in
                        guard newValue > oldValue else { return }
                        handleMessagesUpdate(using: proxy)
                    }
                    .onChange(of: viewModel.isLoading) { _, isLoading in
                        guard isLoading else { return }
                        handleMessagesUpdate(using: proxy, forceScroll: true)
                    }

                    if hasUnreadMessages {
                        Button {
                            hasUnreadMessages = false
                            scrollToBottom(using: proxy, animated: true)
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.down")
                                    .font(.system(size: 10, weight: .semibold))
                                Text(AppLocalization.localizedString("New Message"))
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .foregroundStyle(.white.opacity(0.94))
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
                        .padding(.trailing, 12)
                        .padding(.bottom, 12)
                    }
                }
            }
        }
        .padding(.top, 6)
        .padding(.bottom, 16)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("Follow-up Input"))
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                TextField(AppLocalization.localizedString("Ask about this saved result"), text: $viewModel.draft)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(DetailPanelUIStyle.controlFillColor)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
                    )
                    .disabled(!viewModel.providerStatus.isAvailable || viewModel.isLoading)
                    .onSubmit {
                        viewModel.send()
                    }

                DetailFollowUpSendButton(
                    action: { viewModel.send() },
                    isDisabled: !viewModel.canSend
                )
            }

            if !viewModel.providerStatus.isAvailable {
                Text(viewModel.providerStatus.message)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.orange.opacity(0.92))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 18)
        .padding(.top, 2)
    }

    private func handleMessagesUpdate(using proxy: ScrollViewProxy, forceScroll: Bool = false) {
        if forceScroll || isScrolledToBottom || wasScrolledToBottom {
            hasUnreadMessages = false
            scrollToBottom(using: proxy, animated: true)
        } else {
            hasUnreadMessages = true
        }
    }

    private func scrollToBottom(using proxy: ScrollViewProxy, animated: Bool) {
        let token = UUID()
        pendingScrollRequestToken = token
        DispatchQueue.main.async {
            guard token == pendingScrollRequestToken else { return }
            if animated {
                withAnimation(.easeOut(duration: 0.18)) {
                    proxy.scrollTo(bottomAnchorID, anchor: .bottom)
                }
            } else {
                proxy.scrollTo(bottomAnchorID, anchor: .bottom)
            }
        }
    }
}

private struct TranscriptionDetailMessageRow: View {
    @Environment(\.locale) private var locale

    let message: TranscriptSummaryChatMessage

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            if message.role == .assistant {
                bubble
                Spacer(minLength: 36)
            } else {
                Spacer(minLength: 36)
                bubble
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var bubble: some View {
        TranscriptionDetailMessageBubble(
            message: message,
            timestampText: formattedTimestamp(message.createdAt, locale: locale)
        )
        .frame(maxWidth: 500, alignment: message.role == .assistant ? .leading : .trailing)
    }

    private func formattedTimestamp(_ date: Date, locale: Locale) -> String {
        let calendar = Calendar.current
        if calendar.isDateInToday(date) {
            return date.formatted(
                .dateTime
                    .locale(locale)
                    .hour()
                    .minute()
                    .second()
            )
        }

        return date.formatted(
            .dateTime
                .locale(locale)
                .month(.abbreviated)
                .day()
                .hour()
                .minute()
        )
    }
}

private struct TranscriptionDetailMessageBubble: View {
    let message: TranscriptSummaryChatMessage
    let timestampText: String

    @State private var isHovered = false
    @State private var didCopy = false
    @State private var copyFeedbackToken = UUID()

    private var isUserMessage: Bool {
        message.role == .user
    }

    private var headerTitle: String {
        isUserMessage ? AppLocalization.localizedString("You") : AppLocalization.localizedString("Assistant")
    }

    private var fillColor: Color {
        isUserMessage
            ? Color.accentColor.opacity(0.10)
            : DetailPanelUIStyle.mutedFillColor
    }

    private var borderColor: Color {
        isUserMessage
            ? Color.accentColor.opacity(0.20)
            : DetailPanelUIStyle.softBorderColor
    }

    var body: some View {
        VStack(alignment: isUserMessage ? .trailing : .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 8) {
                if isUserMessage {
                    Text(timestampText)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary.opacity(0.9))

                    Spacer(minLength: 8)

                    Text(headerTitle)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                } else {
                    Text(headerTitle)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)

                    Spacer(minLength: 8)

                    Text(timestampText)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary.opacity(0.9))
                }
            }

            Text(message.content)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.primary.opacity(0.94))
                .multilineTextAlignment(isUserMessage ? .trailing : .leading)
                .frame(maxWidth: .infinity, alignment: isUserMessage ? .trailing : .leading)
                .textSelection(.enabled)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(fillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(borderColor, lineWidth: 1)
        )
        .overlay(alignment: .topTrailing) {
            if !isUserMessage && isHovered {
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
                    .foregroundStyle(.white.opacity(0.96))
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
            guard !isUserMessage else { return }
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }

    private func copyToPasteboard() {
        let trimmed = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(trimmed, forType: .string)

        copyFeedbackToken = UUID()
        let token = copyFeedbackToken
        didCopy = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            guard token == copyFeedbackToken else { return }
            didCopy = false
        }
    }
}

private struct TranscriptionDetailLoadingRow: View {
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text(AppLocalization.localizedString("Assistant"))
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text(AppLocalization.localizedString("Thinking…"))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                }

                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)

                    Text(AppLocalization.localizedString("Generating answer…"))
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.primary.opacity(0.86))
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(DetailPanelUIStyle.mutedFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(DetailPanelUIStyle.softBorderColor, lineWidth: 1)
            )

            Spacer(minLength: 52)
        }
    }
}

private struct DetailFollowUpSendButton: View {
    let action: () -> Void
    let isDisabled: Bool

    var body: some View {
        Button(action: action) {
            Image(systemName: "paperplane.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white.opacity(0.96))
                .frame(width: 34, height: 34)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.accentColor.opacity(0.96))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Color.accentColor.opacity(0.28), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .opacity(isDisabled ? 0.55 : 1)
    }
}
