// MeetingDetailSegmentRow.swift
// Segment and speaker-header rows for meeting detail transcript lists.

import SwiftUI

struct MeetingDetailSegmentRow: View, Equatable {
    let segment: MeetingTranscriptSegment
    let speakerTitle: String
    let isActive: Bool
    let showsTranslation: Bool
    let isSearchMatch: Bool
    let canEditTranscript: Bool
    let isEditing: Bool
    let editingText: String
    let onSelect: () -> Void
    let onBeginEditing: () -> Void
    let onEditingTextChanged: (String) -> Void
    let onSaveEditing: () -> Void
    let onCancelEditing: () -> Void
    let onDelete: () -> Void
    let onToggleHighlight: () -> Void

    @State private var isHovered = false
    @State private var didCopy = false
    @State private var copyFeedbackToken = UUID()

    static func == (lhs: MeetingDetailSegmentRow, rhs: MeetingDetailSegmentRow) -> Bool {
        lhs.segment == rhs.segment
            && lhs.speakerTitle == rhs.speakerTitle
            && lhs.isActive == rhs.isActive
            && lhs.showsTranslation == rhs.showsTranslation
            && lhs.isSearchMatch == rhs.isSearchMatch
            && lhs.canEditTranscript == rhs.canEditTranscript
            && lhs.isEditing == rhs.isEditing
            && lhs.editingText == rhs.editingText
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 8) {
                Text(MeetingTranscriptFormatter.timestampString(for: segment.startSeconds))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)

                Text(speakerTitle)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(
                        segment.speaker == .me
                            ? Color(red: 0.16, green: 0.47, blue: 0.88)
                            : Color(red: 0.12, green: 0.58, blue: 0.32)
                    )

                Spacer(minLength: 8)

                if canEditTranscript {
                    if isEditing {
                        MeetingDetailSegmentActionButton(
                            action: onCancelEditing,
                            tint: Color.secondary,
                            isActive: false,
                            helpText: AppLocalization.localizedString("Cancel"),
                            accessibilityText: AppLocalization.localizedString("Cancel")
                        ) {
                            MeetingDetailCancelEditIcon(color: .secondary)
                        }

                        MeetingDetailSegmentActionButton(
                            action: onSaveEditing,
                            tint: Color.accentColor,
                            isActive: false,
                            helpText: AppLocalization.localizedString("Save"),
                            accessibilityText: AppLocalization.localizedString("Save")
                        ) {
                            MeetingDetailConfirmEditIcon(color: .accentColor)
                        }
                    } else {
                        MeetingDetailSegmentActionButton(
                            action: copySegmentText,
                            tint: Color.accentColor,
                            isActive: didCopy,
                            helpText: AppLocalization.localizedString("Copy"),
                            accessibilityText: AppLocalization.localizedString("Copy")
                        ) {
                            if didCopy {
                                CopySuccessIconView(color: Color.accentColor)
                            } else {
                                CopyIconView(color: .secondary)
                            }
                        }

                        MeetingDetailSegmentActionButton(
                            action: onToggleHighlight,
                            tint: Color.orange,
                            isActive: segment.isHighlighted,
                            helpText: AppLocalization.localizedString(segment.isHighlighted ? "Remove Highlight" : "Highlight"),
                            accessibilityText: AppLocalization.localizedString(segment.isHighlighted ? "Remove Highlight" : "Highlight")
                        ) {
                            MeetingDetailMarkIcon(color: segment.isHighlighted ? Color.orange : Color.secondary)
                        }

                        MeetingDetailSegmentActionButton(
                            action: onBeginEditing,
                            tint: Color.secondary,
                            isActive: false,
                            helpText: AppLocalization.localizedString("Edit"),
                            accessibilityText: AppLocalization.localizedString("Edit")
                        ) {
                            MeetingDetailEditIcon(color: .secondary)
                        }

                        MeetingDetailSegmentActionButton(
                            action: onDelete,
                            tint: Color.red,
                            isActive: false,
                            helpText: AppLocalization.localizedString("Delete"),
                            accessibilityText: AppLocalization.localizedString("Delete")
                        ) {
                            MeetingDetailDeleteIcon(color: .red.opacity(0.82))
                        }
                    }
                } else if isHovered || didCopy {
                    MeetingDetailSegmentActionButton(
                        action: copySegmentText,
                        tint: Color.accentColor,
                        isActive: didCopy,
                        helpText: AppLocalization.localizedString("Copy"),
                        accessibilityText: AppLocalization.localizedString("Copy")
                    ) {
                        if didCopy {
                            CopySuccessIconView(color: Color.accentColor)
                        } else {
                            CopyIconView(color: .secondary)
                        }
                    }
                }
            }

            if isEditing {
                TextEditor(
                    text: Binding(
                        get: { editingText },
                        set: onEditingTextChanged
                    )
                )
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.primary)
                .scrollContentBackground(.hidden)
                .padding(.horizontal, 10)
                .padding(.vertical, 9)
                .frame(minHeight: 88, maxHeight: 180)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(MeetingDetailUIStyle.controlFillColor)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(MeetingDetailUIStyle.borderColor, lineWidth: 1)
                )
            } else {
                Text(segment.text)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.primary.opacity(0.94))
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }

            if showsTranslation,
               let translatedText = segment.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines),
               !translatedText.isEmpty {
                Text(translatedText)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            } else if showsTranslation, segment.isTranslationPending {
                Text(AppLocalization.localizedString("Translating…"))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary.opacity(0.75))
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(backgroundColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
            .stroke(borderColor, lineWidth: 1)
        )
        .onTapGesture(perform: onSelect)
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }

    private func copySegmentText() {
        let trimmed = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        copyStringToPasteboard(trimmed)
        copyFeedbackToken = UUID()
        let token = copyFeedbackToken
        didCopy = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            guard token == copyFeedbackToken else { return }
            withAnimation(.easeOut(duration: 0.12)) {
                didCopy = false
            }
        }
    }

    private var backgroundColor: Color {
        if isActive {
            return speakerAccentColor.opacity(0.16)
        }
        if segment.isHighlighted {
            return Color.orange.opacity(0.16)
        }
        if isSearchMatch {
            return Color.orange.opacity(0.12)
        }
        return speakerAccentColor.opacity(0.06)
    }

    private var borderColor: Color {
        if isActive {
            return speakerAccentColor.opacity(0.32)
        }
        if segment.isHighlighted {
            return Color.orange.opacity(0.42)
        }
        if isSearchMatch {
            return Color.orange.opacity(0.28)
        }
        return speakerAccentColor.opacity(0.16)
    }

    private var speakerAccentColor: Color {
        segment.speaker == .me
            ? Color(red: 0.16, green: 0.47, blue: 0.88)
            : Color(red: 0.12, green: 0.58, blue: 0.32)
    }
}

struct MeetingDetailSpeakerHeaderRow: View, Equatable {
    let title: String
    let count: Int

    var body: some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.primary)

            Text(AppLocalization.format("%d", count))
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(
                    Capsule(style: .continuous)
                        .fill(MeetingDetailUIStyle.mutedFillColor)
                )

            Spacer(minLength: 0)
        }
        .padding(.top, 4)
    }
}
