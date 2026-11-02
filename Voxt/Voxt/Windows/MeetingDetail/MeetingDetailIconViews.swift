// MeetingDetailIconViews.swift
// Provides Meeting Detail Icon Views for meeting detail windows.

import SwiftUI

struct MeetingDetailFollowUpSendButton: View {
    let action: () -> Void
    let isDisabled: Bool

    var body: some View {
        Button(action: action) {
            MeetingDetailFollowUpSendIcon()
                .frame(width: 18, height: 18)
        }
        .buttonStyle(MeetingAccentIconButtonStyle())
        .disabled(isDisabled)
    }
}

struct MeetingDetailSegmentActionButton<Label: View>: View {
    let action: () -> Void
    let tint: Color
    let isActive: Bool
    let helpText: String
    let accessibilityText: String
    var isDisabled = false
    var contentWidth: CGFloat = 16
    var buttonWidth: CGFloat = 28
    @ViewBuilder let label: () -> Label

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            label()
                .frame(width: contentWidth, height: 16)
        }
        .buttonStyle(
            MeetingDetailSegmentActionButtonStyle(
                tint: tint,
                isHovered: isHovered,
                isActive: isActive,
                buttonWidth: buttonWidth
            )
        )
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
        .disabled(isDisabled)
        .opacity(isDisabled ? 0.42 : 1)
        .help(helpText)
        .accessibilityLabel(accessibilityText)
    }
}

private struct MeetingDetailSegmentActionButtonStyle: ButtonStyle {
    let tint: Color
    let isHovered: Bool
    let isActive: Bool
    let buttonWidth: CGFloat

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(width: buttonWidth, height: 28)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(backgroundColor(configuration: configuration))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(borderColor, lineWidth: 1)
            )
            .scaleEffect(configuration.isPressed ? 0.94 : 1)
            .animation(.easeOut(duration: 0.1), value: configuration.isPressed)
    }

    private func backgroundColor(configuration: Configuration) -> Color {
        if isActive {
            return tint.opacity(configuration.isPressed ? 0.2 : 0.13)
        }
        if isHovered {
            return tint.opacity(configuration.isPressed ? 0.16 : 0.09)
        }
        return .clear
    }

    private var borderColor: Color {
        if isActive {
            return tint.opacity(0.24)
        }
        return isHovered ? tint.opacity(0.18) : .clear
    }
}

struct MeetingDetailMarkIcon: View {
    let color: Color

    var body: some View {
        ZStack {
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.markStem)
                .stroke(
                    color,
                    style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round)
                )

            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.markFlag)
                .stroke(
                    color,
                    style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round)
                )
        }
    }
}

struct MeetingDetailEditIcon: View {
    let color: Color

    var body: some View {
        ZStack {
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.editBody)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.editFold)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.editBaseline)
                .fill(color)
        }
    }
}

struct MeetingDetailDeleteIcon: View {
    let color: Color

    var body: some View {
        ZStack {
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.deleteBaseline)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.deleteBody)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.deleteSlashOne)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.deleteSlashTwo)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.deleteSlashThree)
                .fill(color)
        }
    }
}

struct MeetingDetailCancelEditIcon: View {
    let color: Color

    var body: some View {
        SVGPathShape(pathData: MeetingDetailSegmentIconPathData.cancelEdit)
            .fill(color)
    }
}

struct MeetingDetailConfirmEditIcon: View {
    let color: Color

    var body: some View {
        ZStack {
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.confirmEditCircle)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailSegmentIconPathData.confirmEditCheckmark)
                .fill(color)
        }
    }
}

struct MeetingDetailSearchIcon: View {
    let color: Color

    var body: some View {
        ZStack {
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.searchLens)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.searchHandle)
                .fill(color)
        }
    }
}

struct MeetingDetailTranslateIcon: View {
    let color: Color

    var body: some View {
        ZStack {
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.translateRule)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.translateStem)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.translateLeftCurve)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.translateRightCurve)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.translateCircle)
                .fill(color)
        }
    }
}

struct MeetingDetailExportIcon: View {
    let color: Color

    var body: some View {
        ZStack {
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.exportContainer)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.exportChevron)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.exportStem)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.exportCurve)
                .fill(color)
        }
    }
}

struct MeetingDetailSummaryCollapseIcon: View {
    let color: Color

    var body: some View {
        ZStack {
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.summaryContainer)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.summaryDivider)
                .fill(color)
            SVGPathShape(pathData: MeetingDetailHeaderIconPathData.summaryChevron)
                .fill(color)
        }
    }
}

private enum MeetingDetailSegmentIconPathData {
    static let markStem = "M5.15002 2V22"
    static let markFlag = "M5.15002 4H16.35C19.05 4 19.65 5.5 17.75 7.4L16.55 8.6C15.75 9.4 15.75 10.7 16.55 11.4L17.75 12.6C19.65 14.5 18.95 16 16.35 16H5.15002"

    static let editBody = "M5.53999 19.5196C4.92999 19.5196 4.35999 19.3096 3.94999 18.9196C3.42999 18.4296 3.17999 17.6896 3.26999 16.8896L3.63999 13.6496C3.70999 13.0396 4.07999 12.2296 4.50999 11.7896L12.72 3.09956C14.77 0.929561 16.91 0.869561 19.08 2.91956C21.25 4.96956 21.31 7.10956 19.26 9.27956L11.05 17.9696C10.63 18.4196 9.84999 18.8396 9.23999 18.9396L6.01999 19.4896C5.84999 19.4996 5.69999 19.5196 5.53999 19.5196ZM15.93 2.90956C15.16 2.90956 14.49 3.38956 13.81 4.10956L5.59999 12.8096C5.39999 13.0196 5.16999 13.5196 5.12999 13.8096L4.75999 17.0496C4.71999 17.3796 4.79999 17.6496 4.97999 17.8196C5.15999 17.9896 5.42999 18.0496 5.75999 17.9996L8.97999 17.4496C9.26999 17.3996 9.74999 17.1396 9.94999 16.9296L18.16 8.23956C19.4 6.91956 19.85 5.69956 18.04 3.99956C17.24 3.22956 16.55 2.90956 15.93 2.90956Z"
    static let editFold = "M17.3399 10.9508C17.3199 10.9508 17.2899 10.9508 17.2699 10.9508C14.1499 10.6408 11.6399 8.27083 11.1599 5.17083C11.0999 4.76083 11.3799 4.38083 11.7899 4.31083C12.1999 4.25083 12.5799 4.53083 12.6499 4.94083C13.0299 7.36083 14.9899 9.22083 17.4299 9.46083C17.8399 9.50083 18.1399 9.87083 18.0999 10.2808C18.0499 10.6608 17.7199 10.9508 17.3399 10.9508Z"
    static let editBaseline = "M21 22.75H3C2.59 22.75 2.25 22.41 2.25 22C2.25 21.59 2.59 21.25 3 21.25H21C21.41 21.25 21.75 21.59 21.75 22C21.75 22.41 21.41 22.75 21 22.75Z"

    static let deleteBaseline = "M21 22.75H9C8.59 22.75 8.25 22.41 8.25 22C8.25 21.59 8.59 21.25 9 21.25H21C21.41 21.25 21.75 21.59 21.75 22C21.75 22.41 21.41 22.75 21 22.75Z"
    static let deleteBody = "M8.54013 22.71C7.54013 22.71 6.59017 22.32 5.89017 21.61L2.38016 18.1C0.920156 16.64 0.920156 14.26 2.38016 12.8L12.8101 2.37C14.2201 0.96 16.7001 0.96 18.1101 2.37L21.6201 5.87999C23.0801 7.33999 23.0801 9.72 21.6201 11.18L11.1902 21.61C10.4902 22.33 9.55013 22.71 8.54013 22.71ZM3.44015 17.05L6.95016 20.56C7.80016 21.41 9.29016 21.41 10.1302 20.56L20.5601 10.13C21.4401 9.24999 21.4401 7.82999 20.5601 6.94999L17.0501 3.44C16.2101 2.6 14.7201 2.59 13.8701 3.44L3.44015 13.87C2.56015 14.74 2.56015 16.17 3.44015 17.05Z"
    static let deleteSlashOne = "M14.87 17.6303C14.68 17.6303 14.4901 17.5603 14.3401 17.4103L6.59006 9.66032C6.30006 9.37032 6.30006 8.89031 6.59006 8.60031C6.88006 8.31031 7.36006 8.31031 7.65006 8.60031L15.4001 16.3503C15.6901 16.6403 15.6901 17.1203 15.4001 17.4103C15.2501 17.5603 15.06 17.6303 14.87 17.6303Z"
    static let deleteSlashTwo = "M3.51999 18.4095C3.32999 18.4095 3.13996 18.3395 2.98996 18.1895C2.69996 17.8995 2.69996 17.4194 2.98996 17.1294L8.65 11.4695C8.94 11.1795 9.41999 11.1795 9.70999 11.4695C9.99999 11.7595 9.99999 12.2395 9.70999 12.5295L4.04996 18.1895C3.89996 18.3395 3.70999 18.4095 3.51999 18.4095Z"
    static let deleteSlashThree = "M6.33981 21.2395C6.14981 21.2395 5.95979 21.1695 5.80979 21.0195C5.51979 20.7295 5.51979 20.2495 5.80979 19.9595L11.4698 14.2995C11.7598 14.0095 12.2398 14.0095 12.5298 14.2995C12.8198 14.5895 12.8198 15.0695 12.5298 15.3595L6.86978 21.0195C6.72978 21.1595 6.53981 21.2395 6.33981 21.2395Z"

    static let cancelEdit = "M15.13 19.0596H7.13C6.72 19.0596 6.38 18.7196 6.38 18.3096C6.38 17.8996 6.72 17.5596 7.13 17.5596H15.13C17.47 17.5596 19.38 15.6496 19.38 13.3096C19.38 10.9696 17.47 9.05957 15.13 9.05957H4.13C3.72 9.05957 3.38 8.71957 3.38 8.30957C3.38 7.89957 3.72 7.55957 4.13 7.55957H15.13C18.3 7.55957 20.88 10.1396 20.88 13.3096C20.88 16.4796 18.3 19.0596 15.13 19.0596Z M6.43006 11.5599C6.24006 11.5599 6.05006 11.4899 5.90006 11.3399L3.34006 8.77988C3.05006 8.48988 3.05006 8.00988 3.34006 7.71988L5.90006 5.15988C6.19006 4.86988 6.67006 4.86988 6.96006 5.15988C7.25006 5.44988 7.25006 5.92988 6.96006 6.21988L4.93006 8.24988L6.96006 10.2799C7.25006 10.5699 7.25006 11.0499 6.96006 11.3399C6.82006 11.4899 6.62006 11.5599 6.43006 11.5599Z"
    static let confirmEditCircle = "M12 22.75C6.07 22.75 1.25 17.93 1.25 12C1.25 6.07 6.07 1.25 12 1.25C17.93 1.25 22.75 6.07 22.75 12C22.75 17.93 17.93 22.75 12 22.75ZM12 2.75C6.9 2.75 2.75 6.9 2.75 12C2.75 17.1 6.9 21.25 12 21.25C17.1 21.25 21.25 17.1 21.25 12C21.25 6.9 17.1 2.75 12 2.75Z"
    static let confirmEditCheckmark = "M10.5799 15.5796C10.3799 15.5796 10.1899 15.4996 10.0499 15.3596L7.21994 12.5296C6.92994 12.2396 6.92994 11.7596 7.21994 11.4696C7.50994 11.1796 7.98994 11.1796 8.27994 11.4696L10.5799 13.7696L15.7199 8.62961C16.0099 8.33961 16.4899 8.33961 16.7799 8.62961C17.0699 8.91961 17.0699 9.39961 16.7799 9.68961L11.1099 15.3596C10.9699 15.4996 10.7799 15.5796 10.5799 15.5796Z"
}

private enum MeetingDetailHeaderIconPathData {
    static let searchLens = "M11.5 21.75C5.85 21.75 1.25 17.15 1.25 11.5C1.25 5.85 5.85 1.25 11.5 1.25C17.15 1.25 21.75 5.85 21.75 11.5C21.75 17.15 17.15 21.75 11.5 21.75ZM11.5 2.75C6.67 2.75 2.75 6.68 2.75 11.5C2.75 16.32 6.67 20.25 11.5 20.25C16.33 20.25 20.25 16.32 20.25 11.5C20.25 6.68 16.33 2.75 11.5 2.75Z"
    static let searchHandle = "M21.9999 22.7499C21.8099 22.7499 21.6199 22.6799 21.4699 22.5299L19.4699 20.5299C19.1799 20.2399 19.1799 19.7599 19.4699 19.4699C19.7599 19.1799 20.2399 19.1799 20.5299 19.4699L22.5299 21.4699C22.8199 21.7599 22.8199 22.2399 22.5299 22.5299C22.3799 22.6799 22.1899 22.7499 21.9999 22.7499Z"

    static let translateRule = "M16.9897 9.70996H7.00977C6.59977 9.70996 6.25977 9.36996 6.25977 8.95996C6.25977 8.54996 6.59977 8.20996 7.00977 8.20996H16.9897C17.3997 8.20996 17.7397 8.54996 17.7397 8.95996C17.7397 9.36996 17.3997 9.70996 16.9897 9.70996Z"
    static let translateStem = "M12 9.71027C11.59 9.71027 11.25 9.37027 11.25 8.96027V7.28027C11.25 6.87027 11.59 6.53027 12 6.53027C12.41 6.53027 12.75 6.87027 12.75 7.28027V8.96027C12.75 9.37027 12.41 9.71027 12 9.71027Z"
    static let translateLeftCurve = "M7 17.4694C6.59 17.4694 6.25 17.1294 6.25 16.7194C6.25 16.3094 6.59 15.9694 7 15.9694C10.72 15.9694 13.75 12.8195 13.75 8.93945C13.75 8.52945 14.09 8.18945 14.5 8.18945C14.91 8.18945 15.25 8.52945 15.25 8.93945C15.25 13.6495 11.55 17.4694 7 17.4694Z"
    static let translateRightCurve = "M17.0002 17.4702C15.0302 17.4702 13.2002 16.4903 11.8602 14.7003C11.6102 14.3703 11.6802 13.9003 12.0102 13.6503C12.3402 13.4003 12.8102 13.4702 13.0602 13.8002C14.1202 15.2002 15.5202 15.9702 17.0102 15.9702C17.4202 15.9702 17.7602 16.3102 17.7602 16.7202C17.7602 17.1302 17.4102 17.4702 17.0002 17.4702Z"
    static let translateCircle = "M12 22.75C6.07 22.75 1.25 17.93 1.25 12C1.25 6.07 6.07 1.25 12 1.25C17.93 1.25 22.75 6.07 22.75 12C22.75 17.93 17.93 22.75 12 22.75ZM12 2.75C6.9 2.75 2.75 6.9 2.75 12C2.75 17.1 6.9 21.25 12 21.25C17.1 21.25 21.25 17.1 21.25 12C21.25 6.9 17.1 2.75 12 2.75Z"

    static let exportContainer = "M15 22.75H9C3.57 22.75 1.25 20.43 1.25 15V9C1.25 3.57 3.57 1.25 9 1.25H15C20.43 1.25 22.75 3.57 22.75 9V15C22.75 20.43 20.43 22.75 15 22.75ZM9 2.75C4.39 2.75 2.75 4.39 2.75 9V15C2.75 19.61 4.39 21.25 9 21.25H15C19.61 21.25 21.25 19.61 21.25 15V9C21.25 4.39 19.61 2.75 15 2.75H9Z"
    static let exportChevron = "M11.9999 15.2602C11.8099 15.2602 11.6199 15.1902 11.4699 15.0402L8.46994 12.0402C8.17994 11.7502 8.17994 11.2702 8.46994 10.9802C8.75994 10.6902 9.23994 10.6902 9.52994 10.9802L11.9999 13.4502L14.4699 10.9802C14.7599 10.6902 15.2399 10.6902 15.5299 10.9802C15.8199 11.2702 15.8199 11.7502 15.5299 12.0402L12.5299 15.0402C12.3799 15.1902 12.1899 15.2602 11.9999 15.2602Z"
    static let exportStem = "M12 15.2598C11.59 15.2598 11.25 14.9198 11.25 14.5098V6.50977C11.25 6.09977 11.59 5.75977 12 5.75977C12.41 5.75977 12.75 6.09977 12.75 6.50977V14.5098C12.75 14.9298 12.41 15.2598 12 15.2598Z"
    static let exportCurve = "M11.9999 18.2302C9.88994 18.2302 7.76995 17.8902 5.75995 17.2202C5.36995 17.0902 5.15995 16.6602 5.28995 16.2702C5.41995 15.8802 5.83994 15.6602 6.23994 15.8002C9.95994 17.0402 14.0499 17.0402 17.7699 15.8002C18.1599 15.6702 18.5899 15.8802 18.7199 16.2702C18.8499 16.6602 18.6399 17.0902 18.2499 17.2202C16.2299 17.9002 14.1099 18.2302 11.9999 18.2302Z"

    static let summaryContainer = "M14.97 22.75H8.96997C3.53997 22.75 1.21997 20.43 1.21997 15V9C1.21997 3.57 3.53997 1.25 8.96997 1.25H14.97C20.4 1.25 22.72 3.57 22.72 9V15C22.72 20.43 20.41 22.75 14.97 22.75ZM8.96997 2.75C4.35997 2.75 2.71997 4.39 2.71997 9V15C2.71997 19.61 4.35997 21.25 8.96997 21.25H14.97C19.58 21.25 21.22 19.61 21.22 15V9C21.22 4.39 19.58 2.75 14.97 2.75H8.96997Z"
    static let summaryDivider = "M14.97 22.75C14.56 22.75 14.22 22.41 14.22 22V2C14.22 1.59 14.56 1.25 14.97 1.25C15.38 1.25 15.72 1.59 15.72 2V22C15.72 22.41 15.39 22.75 14.97 22.75Z"
    static let summaryChevron = "M7.96991 15.3109C7.77991 15.3109 7.58991 15.2409 7.43991 15.0909C7.14991 14.8009 7.14991 14.3209 7.43991 14.0309L9.46991 12.0009L7.43991 9.97086C7.14991 9.68086 7.14991 9.20086 7.43991 8.91086C7.72991 8.62086 8.20991 8.62086 8.49991 8.91086L11.0599 11.4709C11.3499 11.7609 11.3499 12.2409 11.0599 12.5309L8.49991 15.0909C8.35991 15.2409 8.16991 15.3109 7.96991 15.3109Z"
}

private struct MeetingAccentIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(width: 34, height: 34)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color.accentColor.opacity(configuration.isPressed ? 0.82 : 0.96))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Color.accentColor.opacity(0.28), lineWidth: 1)
            )
            .opacity(configuration.isPressed ? 0.94 : 1)
    }
}

private struct MeetingDetailFollowUpSendIcon: View {
    var body: some View {
        ZStack {
            MeetingDetailFollowUpSparkShape()
                .stroke(.white, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))

            MeetingDetailFollowUpPlaneShape()
                .stroke(.white, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))

            MeetingDetailFollowUpTrailShape()
                .stroke(Color.white.opacity(0.4), style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
        }
    }
}

private enum MeetingDetailIconGrid {
    static let size: CGFloat = 24

    static func point(in rect: CGRect, x: CGFloat, y: CGFloat) -> CGPoint {
        CGPoint(
            x: rect.minX + rect.width * x / size,
            y: rect.minY + rect.height * y / size
        )
    }
}

private struct MeetingDetailFollowUpSparkShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: MeetingDetailIconGrid.point(in: rect, x: 19.83, y: 15.6))
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 18.69, y: 15.86))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 17.04, y: 17.51),
            control1: MeetingDetailIconGrid.point(in: rect, x: 17.87, y: 16.05),
            control2: MeetingDetailIconGrid.point(in: rect, x: 17.23, y: 16.69)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 16.77, y: 18.65))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 16.54, y: 18.65),
            control1: MeetingDetailIconGrid.point(in: rect, x: 16.74, y: 18.77),
            control2: MeetingDetailIconGrid.point(in: rect, x: 16.57, y: 18.77)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 16.28, y: 17.51))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 14.63, y: 15.86),
            control1: MeetingDetailIconGrid.point(in: rect, x: 16.09, y: 16.69),
            control2: MeetingDetailIconGrid.point(in: rect, x: 15.45, y: 16.05)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 13.49, y: 15.59))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 13.49, y: 15.36),
            control1: MeetingDetailIconGrid.point(in: rect, x: 13.37, y: 15.56),
            control2: MeetingDetailIconGrid.point(in: rect, x: 13.37, y: 15.39)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 14.63, y: 15.1))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 16.28, y: 13.45),
            control1: MeetingDetailIconGrid.point(in: rect, x: 15.45, y: 14.91),
            control2: MeetingDetailIconGrid.point(in: rect, x: 16.09, y: 14.27)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 16.55, y: 12.31))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 16.78, y: 12.31),
            control1: MeetingDetailIconGrid.point(in: rect, x: 16.58, y: 12.19),
            control2: MeetingDetailIconGrid.point(in: rect, x: 16.75, y: 12.19)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 17.04, y: 13.45))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 18.69, y: 15.1),
            control1: MeetingDetailIconGrid.point(in: rect, x: 17.23, y: 14.27),
            control2: MeetingDetailIconGrid.point(in: rect, x: 17.87, y: 14.91)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 19.83, y: 15.37))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 19.83, y: 15.6),
            control1: MeetingDetailIconGrid.point(in: rect, x: 19.95, y: 15.4),
            control2: MeetingDetailIconGrid.point(in: rect, x: 19.95, y: 15.57)
        )
        return path
    }
}

private struct MeetingDetailFollowUpPlaneShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: MeetingDetailIconGrid.point(in: rect, x: 12.31, y: 18.37))
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 9.51, y: 19.77))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 4.28, y: 14.54),
            control1: MeetingDetailIconGrid.point(in: rect, x: 3.75, y: 22.65),
            control2: MeetingDetailIconGrid.point(in: rect, x: 1.4, y: 20.29)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 5.15, y: 12.81))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 5.15, y: 11.2),
            control1: MeetingDetailIconGrid.point(in: rect, x: 5.37, y: 12.37),
            control2: MeetingDetailIconGrid.point(in: rect, x: 5.37, y: 11.64)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 4.28, y: 9.46))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 9.51, y: 4.23),
            control1: MeetingDetailIconGrid.point(in: rect, x: 1.4, y: 3.71),
            control2: MeetingDetailIconGrid.point(in: rect, x: 3.76, y: 1.35)
        )
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 18.07, y: 8.51))
        path.addCurve(
            to: MeetingDetailIconGrid.point(in: rect, x: 20.78, y: 12.92),
            control1: MeetingDetailIconGrid.point(in: rect, x: 20.46, y: 9.71),
            control2: MeetingDetailIconGrid.point(in: rect, x: 21.36, y: 11.37)
        )
        return path
    }
}

private struct MeetingDetailFollowUpTrailShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: MeetingDetailIconGrid.point(in: rect, x: 5.44, y: 12))
        path.addLine(to: MeetingDetailIconGrid.point(in: rect, x: 10.84, y: 12))
        return path
    }
}
