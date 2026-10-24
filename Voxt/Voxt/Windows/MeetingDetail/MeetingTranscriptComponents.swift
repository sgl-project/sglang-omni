// MeetingTranscriptComponents.swift
// Provides Meeting Transcript Components for meeting detail windows.

import SwiftUI

private struct MeetingBottomVisibilityPreferenceKey: PreferenceKey {
    static var defaultValue = true

    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = nextValue()
    }
}

struct MeetingTranscriptScrollView: View {
    let segments: [MeetingTranscriptSegment]
    let onCopySegment: (MeetingTranscriptSegment) -> Void

    @State private var bottomVisible = true
    @State private var autoScrollPinnedToBottom = true
    @State private var hasUnreadAtBottom = false
    @State private var copiedSegmentID: UUID?
    @State private var copyFeedbackToken = UUID()
    @State private var lastScrollContentSignature = ""

    var body: some View {
        GeometryReader { outerProxy in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottomTrailing) {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            if segments.isEmpty {
                                VStack(spacing: 10) {
                                    Text(AppLocalization.localizedString("The transcript timeline for Me / Them will appear here once the meeting starts."))
                                        .font(.system(size: 14, weight: .medium))
                                        .foregroundStyle(.white.opacity(0.7))

                                    Text(AppLocalization.localizedString("Automatic scrolling pauses when you scroll away from the bottom."))
                                        .font(.system(size: 12, weight: .medium))
                                        .foregroundStyle(.white.opacity(0.42))
                                }
                                .frame(maxWidth: .infinity, minHeight: 160, alignment: .center)
                            } else {
                                ForEach(segments) { segment in
                                    MeetingTranscriptRow(
                                        segment: segment,
                                        onTap: {
                                            onCopySegment(segment)
                                            let token = UUID()
                                            copyFeedbackToken = token
                                            copiedSegmentID = segment.id
                                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
                                                guard copyFeedbackToken == token else { return }
                                                copiedSegmentID = nil
                                            }
                                        },
                                        isCopied: copiedSegmentID == segment.id
                                    )
                                }
                            }

                            GeometryReader { geo in
                                Color.clear
                                    .preference(
                                        key: MeetingBottomVisibilityPreferenceKey.self,
                                        value: abs(geo.frame(in: .named("MeetingTranscriptScroll")).maxY - outerProxy.size.height) < 36
                                    )
                            }
                            .frame(height: 1)
                            .id("meeting-bottom-anchor")
                        }
                        .padding(.trailing, 10)
                    }
                    .coordinateSpace(name: "MeetingTranscriptScroll")
                    .onPreferenceChange(MeetingBottomVisibilityPreferenceKey.self) { isVisible in
                        bottomVisible = isVisible
                        if isVisible {
                            autoScrollPinnedToBottom = true
                            hasUnreadAtBottom = false
                        } else if scrollContentSignature == lastScrollContentSignature {
                            autoScrollPinnedToBottom = false
                        }
                    }
                    .onChange(of: scrollContentSignature) { oldValue, newValue in
                        guard oldValue != newValue else { return }
                        handleTranscriptContentChange(proxy: proxy)
                    }
                    .onAppear {
                        lastScrollContentSignature = scrollContentSignature
                        scrollToBottom(proxy: proxy, animated: false)
                    }

                    if hasUnreadAtBottom {
                        Button {
                            hasUnreadAtBottom = false
                            scrollToBottom(proxy: proxy, animated: true)
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

    private var scrollContentSignature: String {
        segments.map { segment in
            [
                segment.id.uuidString,
                segment.text,
                segment.translatedText ?? "",
                segment.isTranslationPending ? "1" : "0"
            ].joined(separator: "\u{1F}")
        }
        .joined(separator: "\u{1E}")
    }

    private func handleTranscriptContentChange(proxy: ScrollViewProxy) {
        defer {
            lastScrollContentSignature = scrollContentSignature
        }

        guard lastScrollContentSignature != scrollContentSignature else { return }
        if bottomVisible || autoScrollPinnedToBottom {
            autoScrollPinnedToBottom = true
            scrollToBottom(proxy: proxy, animated: true)
        } else {
            hasUnreadAtBottom = true
        }
    }

    private func scrollToBottom(proxy: ScrollViewProxy, animated: Bool) {
        DispatchQueue.main.async {
            if animated {
                withAnimation(.easeOut(duration: 0.18)) {
                    proxy.scrollTo("meeting-bottom-anchor", anchor: .bottom)
                }
            } else {
                proxy.scrollTo("meeting-bottom-anchor", anchor: .bottom)
            }
        }
    }
}

private struct MeetingTranscriptRow: View {
    let segment: MeetingTranscriptSegment
    let onTap: () -> Void
    let isCopied: Bool

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 8) {
                    HStack(spacing: 8) {
                        Text(MeetingTranscriptFormatter.timestampString(for: segment.startSeconds))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.48))

                        Text(segment.displaySpeakerTitle)
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(segment.speaker == .me ? Color(red: 0.55, green: 0.78, blue: 1.0) : Color(red: 0.56, green: 0.93, blue: 0.72))
                    }

                    Spacer(minLength: 8)

                    if isCopied {
                        CopySuccessIconView()
                            .frame(width: 14, height: 14)
                    }
                }

                Text(segment.text)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white.opacity(0.9))
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)

                if let translatedText = segment.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !translatedText.isEmpty {
                    Text(translatedText)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.white.opacity(0.58))
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else if segment.isTranslationPending {
                    Text(AppLocalization.localizedString("Translating…"))
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.44))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 11)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(.white.opacity(0.05))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(.white.opacity(0.06), lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct MeetingMiniWaveform: View {
    @ObservedObject var waveformState: RecentAudioWaveformState
    var isSubdued = false
    var showsProcessingLoader = false
    var isAnimatingLoader = true

    var body: some View {
        Group {
            if showsProcessingLoader {
                WaveformProcessingLoaderView(
                    isAnimating: isAnimatingLoader,
                    itemCount: 5,
                    itemSize: CGSize(width: 5, height: 5),
                    spacing: 4,
                    color: .white
                )
                .frame(maxWidth: .infinity, alignment: .center)
            } else {
                HStack(alignment: .center, spacing: 2.5) {
                    ForEach(0..<waveformState.barCount, id: \.self) { index in
                        RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                            .fill(WaveformBarVisuals.barGradient)
                            .frame(width: 4, height: barHeight(for: index))
                            .shadow(color: .white.opacity(glowOpacity(for: index)), radius: 2.5, x: 0, y: 0)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .center)
            }
        }
        .frame(height: 28)
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

struct MeetingDetailTranscriptListPane: View, Equatable {
    let rows: [MeetingTranscriptVirtualRow]
    let showsTranslation: Bool
    let scrollRequest: MeetingTranscriptScrollRequest?
    let canEditTranscript: Bool
    let editingSegmentID: UUID?
    let editingText: String
    let onSelectSegment: (MeetingTranscriptSegment) -> Void
    let onBeginEditing: (MeetingTranscriptSegment) -> Void
    let onEditingTextChanged: (String) -> Void
    let onSaveEditing: () -> Void
    let onCancelEditing: () -> Void
    let onDeleteSegment: (MeetingTranscriptSegment) -> Void
    let onToggleHighlight: (MeetingTranscriptSegment) -> Void

    static func == (
        lhs: MeetingDetailTranscriptListPane,
        rhs: MeetingDetailTranscriptListPane
    ) -> Bool {
        lhs.rows == rhs.rows
            && lhs.showsTranslation == rhs.showsTranslation
            && lhs.scrollRequest == rhs.scrollRequest
            && lhs.canEditTranscript == rhs.canEditTranscript
            && lhs.editingSegmentID == rhs.editingSegmentID
            && lhs.editingText == rhs.editingText
    }

    var body: some View {
        MeetingTranscriptVirtualList(
            rows: rows,
            showsTranslation: showsTranslation,
            scrollRequest: scrollRequest,
            canEditTranscript: canEditTranscript,
            editingSegmentID: editingSegmentID,
            editingText: editingText,
            onSelectSegment: onSelectSegment,
            onBeginEditing: onBeginEditing,
            onEditingTextChanged: onEditingTextChanged,
            onSaveEditing: onSaveEditing,
            onCancelEditing: onCancelEditing,
            onDeleteSegment: onDeleteSegment,
            onToggleHighlight: onToggleHighlight
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
