// HistorySettingsComponents.swift
// Provides History Settings Components for history settings.

import SwiftUI
import AppKit
import UniformTypeIdentifiers

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

enum HistoryFilterTab: String, CaseIterable, Hashable, Identifiable {
    case transcription
    case translation
    case rewrite
    case note
    case transcript

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        LocalizedStringKey(rawTitleKey)
    }

    var title: String {
        localized(rawTitleKey)
    }

    var correspondingFeatureTab: FeatureSettingsTab {
        switch self {
        case .transcription:
            return .transcription
        case .translation:
            return .translation
        case .transcript:
            return .meeting
        case .rewrite:
            return .rewrite
        case .note:
            return .note
        }
    }

    private var rawTitleKey: String {
        switch self {
        case .transcription:
            return "Transcription"
        case .translation:
            return "Translation"
        case .transcript:
            return "Meeting"
        case .rewrite:
            return "Rewrite"
        case .note:
            return "Notes"
        }
    }

    func matches(_ entry: TranscriptionHistoryEntry) -> Bool {
        switch self {
        case .transcription:
            return entry.kind == .normal
        case .translation:
            return entry.kind == .translation
        case .transcript:
            return entry.kind == .transcript
        case .rewrite:
            return entry.kind == .rewrite
        case .note:
            return false
        }
    }
}

struct HistoryDayHeader: View {
    @Environment(\.locale) private var locale
    let date: Date

    var body: some View {
        let isToday = Calendar.current.isDateInToday(date)
        let title = isToday ? localized("Today") : date.formatted(
            .dateTime
                .locale(locale)
                .year()
                .month(.defaultDigits)
                .day()
        )

        Text(title)
        .font(.system(size: 14, weight: .medium))
        .foregroundStyle(.primary)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
        .padding(.leading, 2)
        .padding(.bottom, 5)
    }
}

struct HistoryRow: View {
    @State private var isHovered = false

    let entry: TranscriptionHistoryEntry
    let audioURL: URL?
    let isCompact: Bool
    let onCopy: () -> Void
    let onShowInfo: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Text(timeText)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 48, alignment: .leading)
                    .padding(.top, 1)

                Button(action: onCopy) {
                    Text(displayText)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(.primary)
                        .multilineTextAlignment(.leading)
                        .lineSpacing(2)
                        .lineLimit(3)
                        .truncationMode(.tail)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                        .help(localized("Copy"))
                }
                .buttonStyle(.plain)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

            HStack(spacing: 6) {
                Button(action: onShowInfo) {
                    HistoryActionIcon(kind: .detail)
                }
                .buttonStyle(SettingsCompactIconButtonStyle(size: 26))

                Button(role: .destructive, action: onDelete) {
                    HistoryActionIcon(kind: .delete)
                }
                .buttonStyle(SettingsCompactIconButtonStyle(size: 26))
            }
            .opacity(isHovered ? 1 : 0)
            .allowsHitTesting(isHovered)
            .animation(.easeInOut(duration: 0.12), value: isHovered)
            .frame(width: 58)
            .frame(maxHeight: .infinity, alignment: .center)
        }
        .padding(.horizontal, 9.5)
        .padding(.vertical, isCompact ? 5 : 7)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: HistoryRowStyle.cornerRadius, style: .continuous)
                .fill(HistoryRowStyle.fillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: HistoryRowStyle.cornerRadius, style: .continuous)
                .strokeBorder(isHovered ? HistoryRowStyle.hoverBorderColor : HistoryRowStyle.borderColor, lineWidth: 1)
        )
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }

    private var displayText: String {
        Self.displayText(for: entry)
    }

    static func displayText(for entry: TranscriptionHistoryEntry) -> String {
        let corrected = HistoryCorrectionPresentation.correctedText(
            for: entry.text,
            snapshots: entry.dictionaryCorrectionSnapshots
        )
        if !corrected.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return corrected
        }
        return entry.displayTitle ?? entry.meetingCaptureMode?.title ?? localized("Recording")
    }

    private var timeText: String {
        RelativeNoteTimestampFormatter.historyListTime(for: entry.createdAt)
    }
}

struct HistoryListRow: View {
    @State private var isHovered = false

    let timeText: String
    let displayText: String
    let onCopy: () -> Void
    let onShowInfo: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Text(timeText)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 48, alignment: .leading)
                    .padding(.top, 1)

                Button(action: onCopy) {
                    Text(displayText)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(.primary)
                        .multilineTextAlignment(.leading)
                        .lineSpacing(2)
                        .lineLimit(3)
                        .truncationMode(.tail)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                        .help(localized("Copy"))
                }
                .buttonStyle(.plain)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

            HStack(spacing: 6) {
                Button(action: onShowInfo) {
                    HistoryActionIcon(kind: .detail)
                }
                .buttonStyle(SettingsCompactIconButtonStyle(size: 26))

                Button(role: .destructive, action: onDelete) {
                    HistoryActionIcon(kind: .delete)
                }
                .buttonStyle(SettingsCompactIconButtonStyle(size: 26))
            }
            .opacity(isHovered ? 1 : 0)
            .allowsHitTesting(isHovered)
            .animation(.easeInOut(duration: 0.12), value: isHovered)
            .frame(width: 58)
            .frame(maxHeight: .infinity, alignment: .center)
        }
        .padding(.horizontal, 9.5)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: HistoryRowStyle.cornerRadius, style: .continuous)
                .fill(HistoryRowStyle.fillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: HistoryRowStyle.cornerRadius, style: .continuous)
                .strokeBorder(isHovered ? HistoryRowStyle.hoverBorderColor : HistoryRowStyle.borderColor, lineWidth: 1)
        )
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }
}

enum HistoryActionIconKind {
    case detail
    case delete
}

struct HistoryActionIcon: View {
    let kind: HistoryActionIconKind
    var color: Color = HistoryRowStyle.actionIconColor

    var body: some View {
        AppSVGIcon(
            kind: kind == .detail ? .historyDetails : .delete,
            color: color,
            size: 17
        )
        .contentShape(Rectangle())
    }
}

enum HistoryRowStyle {
    static let cornerRadius: CGFloat = 12

    static var fillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor(calibratedWhite: 0.972, alpha: 1),
            dark: NSColor(calibratedWhite: 0.155, alpha: 1)
        ))
    }

    static var linearCardFillColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.white,
            dark: NSColor(calibratedWhite: 0.165, alpha: 1)
        ))
    }

    static var borderColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.035),
            dark: NSColor.white.withAlphaComponent(0.055)
        ))
    }

    static var hoverBorderColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black.withAlphaComponent(0.075),
            dark: NSColor.white.withAlphaComponent(0.105)
        ))
    }

    static var actionIconColor: Color {
        Color(nsColor: dynamicColor(
            light: NSColor.black,
            dark: NSColor.white.withAlphaComponent(0.92)
        ))
    }

    private static func dynamicColor(light: NSColor, dark: NSColor) -> NSColor {
        NSColor(name: nil) { appearance in
            switch appearance.bestMatch(from: [.darkAqua, .aqua]) {
            case .darkAqua:
                return dark
            default:
                return light
            }
        }
    }
}

struct HistoryToolbarDeleteButtonStyle: ButtonStyle {
    var size: CGFloat = 28

    func makeBody(configuration: Configuration) -> some View {
        HistoryToolbarDeleteButtonBody(configuration: configuration, size: size)
    }
}

private struct HistoryToolbarDeleteButtonBody: View {
    let configuration: HistoryToolbarDeleteButtonStyle.Configuration
    let size: CGFloat
    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 11, weight: .semibold))
            .frame(width: size, height: size)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(stroke, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            .opacity(configuration.isPressed ? 0.92 : 1)
            .onHover { isHovered = $0 }
    }

    private var fill: Color {
        if configuration.isPressed {
            return .red.opacity(0.16)
        }
        if isHovered {
            return .red.opacity(0.13)
        }
        return SettingsUIStyle.subtleFillColor
    }

    private var stroke: Color {
        if configuration.isPressed || isHovered {
            return .red.opacity(isHovered ? 0.30 : 0.22)
        }
        return SettingsUIStyle.subtleBorderColor
    }
}

struct HistoryToolbarSettingsIcon: View {
    var color: Color = .secondary
    var size: CGFloat = 16

    var body: some View {
        ZStack {
            SVGPathShape(pathData: "M12 15.75C9.93 15.75 8.25 14.07 8.25 12C8.25 9.93 9.93 8.25 12 8.25C14.07 8.25 15.75 9.93 15.75 12C15.75 14.07 14.07 15.75 12 15.75ZM12 9.75C10.76 9.75 9.75 10.76 9.75 12C9.75 13.24 10.76 14.25 12 14.25C13.24 14.25 14.25 13.24 14.25 12C14.25 10.76 13.24 9.75 12 9.75Z")
                .fill(color)
            SVGPathShape(pathData: "M15.21 22.1903C15 22.1903 14.79 22.1603 14.58 22.1103C13.96 21.9403 13.44 21.5503 13.11 21.0003L12.99 20.8003C12.4 19.7803 11.59 19.7803 11 20.8003L10.89 20.9903C10.56 21.5503 10.04 21.9503 9.42 22.1103C8.79 22.2803 8.14 22.1903 7.59 21.8603L5.87 20.8703C5.26 20.5203 4.82 19.9503 4.63 19.2603C4.45 18.5703 4.54 17.8603 4.89 17.2503C5.18 16.7403 5.26 16.2803 5.09 15.9903C4.92 15.7003 4.49 15.5303 3.9 15.5303C2.44 15.5303 1.25 14.3403 1.25 12.8803V11.1203C1.25 9.66029 2.44 8.47029 3.9 8.47029C4.49 8.47029 4.92 8.30029 5.09 8.01029C5.26 7.72029 5.19 7.26029 4.89 6.75029C4.54 6.14029 4.45 5.42029 4.63 4.74029C4.81 4.05029 5.25 3.48029 5.87 3.13029L7.6 2.14029C8.73 1.47029 10.22 1.86029 10.9 3.01029L11.02 3.21029C11.61 4.23029 12.42 4.23029 13.01 3.21029L13.12 3.02029C13.8 1.86029 15.29 1.47029 16.43 2.15029L18.15 3.14029C18.76 3.49029 19.2 4.06029 19.39 4.75029C19.57 5.44029 19.48 6.15029 19.13 6.76029C18.84 7.27029 18.76 7.73029 18.93 8.02029C19.1 8.31029 19.53 8.48029 20.12 8.48029C21.58 8.48029 22.77 9.67029 22.77 11.1303V12.8903C22.77 14.3503 21.58 15.5403 20.12 15.5403C19.53 15.5403 19.1 15.7103 18.93 16.0003C18.76 16.2903 18.83 16.7503 19.13 17.2603C19.48 17.8703 19.58 18.5903 19.39 19.2703C19.21 19.9603 18.77 20.5303 18.15 20.8803L16.42 21.8703C16.04 22.0803 15.63 22.1903 15.21 22.1903ZM12 18.4903C12.89 18.4903 13.72 19.0503 14.29 20.0403L14.4 20.2303C14.52 20.4403 14.72 20.5903 14.96 20.6503C15.2 20.7103 15.44 20.6803 15.64 20.5603L17.37 19.5603C17.63 19.4103 17.83 19.1603 17.91 18.8603C17.99 18.5603 17.95 18.2503 17.8 17.9903C17.23 17.0103 17.16 16.0003 17.6 15.2303C18.04 14.4603 18.95 14.0203 20.09 14.0203C20.73 14.0203 21.24 13.5103 21.24 12.8703V11.1103C21.24 10.4803 20.73 9.96029 20.09 9.96029C18.95 9.96029 18.04 9.52029 17.6 8.75029C17.16 7.98029 17.23 6.97029 17.8 5.99029C17.95 5.73029 17.99 5.42029 17.91 5.12029C17.83 4.82029 17.64 4.58029 17.38 4.42029L15.65 3.43029C15.22 3.17029 14.65 3.32029 14.39 3.76029L14.28 3.95029C13.71 4.94029 12.88 5.50029 11.99 5.50029C11.1 5.50029 10.27 4.94029 9.7 3.95029L9.59 3.75029C9.34 3.33029 8.78 3.18029 8.35 3.43029L6.62 4.43029C6.36 4.58029 6.16 4.83029 6.08 5.13029C6 5.43029 6.04 5.74029 6.19 6.00029C6.76 6.98029 6.83 7.99029 6.39 8.76029C5.95 9.53029 5.04 9.97029 3.9 9.97029C3.26 9.97029 2.75 10.4803 2.75 11.1203V12.8803C2.75 13.5103 3.26 14.0303 3.9 14.0303C5.04 14.0303 5.95 14.4703 6.39 15.2403C6.83 16.0103 6.76 17.0203 6.19 18.0003C6.04 18.2603 6 18.5703 6.08 18.8703C6.16 19.1703 6.35 19.4103 6.61 19.5703L8.34 20.5603C8.55 20.6903 8.8 20.7203 9.03 20.6603C9.27 20.6003 9.47 20.4403 9.6 20.2303L9.71 20.0403C10.28 19.0603 11.11 18.4903 12 18.4903Z")
                .fill(color)
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}
