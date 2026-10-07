// MeetingOverlayView.swift
// Provides Meeting Overlay View for window and overlay UI.

import SwiftUI

struct MeetingOverlayContainerView: View {
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @ObservedObject var state: MeetingOverlayState
    let onClose: () -> Void
    let onToggleCollapse: () -> Void
    let onTogglePause: () -> Void
    let onShowDetail: () -> Void
    let onRealtimeTranslateToggle: (Bool) -> Void
    let onCaptureModeChange: (MeetingCaptureMode) -> Void
    let onToggleCaptureModePicker: () -> Void
    let onDismissCaptureModePicker: () -> Void
    let onConfirmRealtimeTranslationLanguage: () -> Void
    let onCancelRealtimeTranslationLanguage: () -> Void
    let onConfirmCancelMeeting: () -> Void
    let onConfirmFinishMeeting: () -> Void
    let onDismissCloseConfirmation: () -> Void
    let onCopySegment: (MeetingTranscriptSegment) -> Void

    var body: some View {
        let _ = interfaceLanguageRaw
        MeetingOverlayCard(
            state: state,
            onClose: onClose,
            onToggleCollapse: onToggleCollapse,
            onTogglePause: onTogglePause,
            onShowDetail: onShowDetail,
            onRealtimeTranslateToggle: onRealtimeTranslateToggle,
            onCaptureModeChange: onCaptureModeChange,
            onToggleCaptureModePicker: onToggleCaptureModePicker,
            onDismissCaptureModePicker: onDismissCaptureModePicker,
            onConfirmRealtimeTranslationLanguage: onConfirmRealtimeTranslationLanguage,
            onCancelRealtimeTranslationLanguage: onCancelRealtimeTranslationLanguage,
            onConfirmCancelMeeting: onConfirmCancelMeeting,
            onConfirmFinishMeeting: onConfirmFinishMeeting,
            onDismissCloseConfirmation: onDismissCloseConfirmation,
            onCopySegment: onCopySegment
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.top, state.isCollapsed ? 0 : 8)
    }
}

private struct MeetingOverlayCard: View {
    private let captureModePickerWidth: CGFloat = 224
    private let translationLanguageDialogMaxHeight: CGFloat = 300
    private let translationLanguageListMaxHeight: CGFloat = 156

    @AppStorage(AppPreferenceKey.overlayCardOpacity) private var overlayCardOpacity = 82
    @AppStorage(AppPreferenceKey.overlayCardCornerRadius) private var overlayCardCornerRadius = 24

    @ObservedObject var state: MeetingOverlayState
    let onClose: () -> Void
    let onToggleCollapse: () -> Void
    let onTogglePause: () -> Void
    let onShowDetail: () -> Void
    let onRealtimeTranslateToggle: (Bool) -> Void
    let onCaptureModeChange: (MeetingCaptureMode) -> Void
    let onToggleCaptureModePicker: () -> Void
    let onDismissCaptureModePicker: () -> Void
    let onConfirmRealtimeTranslationLanguage: () -> Void
    let onCancelRealtimeTranslationLanguage: () -> Void
    let onConfirmCancelMeeting: () -> Void
    let onConfirmFinishMeeting: () -> Void
    let onDismissCloseConfirmation: () -> Void
    let onCopySegment: (MeetingTranscriptSegment) -> Void

    var body: some View {
        ZStack {
            VStack(alignment: .leading, spacing: 0) {
                header

                if !state.isCollapsed {
                    Rectangle()
                        .fill(.white.opacity(0.08))
                        .frame(height: 1)
                        .padding(.top, 6)
                        .padding(.bottom, 8)

                    if let safetyMessage = state.safetyMessage {
                        Text(safetyMessage)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.orange.opacity(0.95))
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.bottom, 8)
                    }

                    transcriptContent
                }
            }
            .frame(maxWidth: .infinity, maxHeight: state.isCollapsed ? nil : .infinity, alignment: .topLeading)
            .padding(.horizontal, 18)
            .padding(.vertical, state.isCollapsed ? 10 : 14)
            .background(cardBackground)
            .compositingGroup()
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
            )

            if state.isCloseConfirmationPresented || state.isRealtimeTranslationLanguagePickerPresented {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(.black.opacity(0.22))
                    .contentShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                    .onTapGesture {
                        if state.isCloseConfirmationPresented {
                            onDismissCloseConfirmation()
                        }
                    }

                if state.isCloseConfirmationPresented {
                    meetingCloseConfirmationDialog
                } else {
                    realtimeTranslationLanguageDialog
                }
            }
        }
        .overlayPreferenceValue(MeetingCaptureModeSelectorBoundsPreferenceKey.self) { anchor in
            GeometryReader { proxy in
                if state.isCaptureModePickerPresented,
                   !state.isCloseConfirmationPresented,
                   !state.isRealtimeTranslationLanguagePickerPresented,
                   let anchor {
                    let buttonFrame = proxy[anchor]
                    MeetingCaptureModePicker(
                        selection: state.captureMode,
                        onSelect: onCaptureModeChange
                    )
                    .offset(
                        x: buttonFrame.midX - (captureModePickerWidth / 2),
                        y: buttonFrame.maxY + 8
                    )
                    .transition(.meetingCaptureModePickerReveal)
                    .zIndex(3)
                }
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.84), value: state.isCaptureModePickerPresented)
        .padding(.horizontal, 12)
        .shadow(
            color: .black.opacity(state.isCollapsed ? 0 : 0.18),
            radius: state.isCollapsed ? 0 : 18,
            y: state.isCollapsed ? 0 : 10
        )
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 12) {
                HStack(alignment: .center, spacing: 10) {
                    Group {
                        if state.isModelInitializing {
                            ModelInitializingIconView()
                        } else {
                            TranscriptionModeIconView()
                        }
                    }
                    .frame(width: 18, height: 18)

                    MeetingMiniWaveform(
                        waveformState: state.waveformState,
                        isSubdued: state.isModelInitializing,
                        showsProcessingLoader: showsHeaderWaveformLoader,
                        isAnimatingLoader: state.isPresented
                    )
                        .frame(width: state.isCollapsed ? 96 : 116, height: 28)
                }

                Spacer(minLength: 12)

                if !state.isCollapsed {
                    HStack(spacing: 8) {
                        Toggle(
                            AppLocalization.localizedString("Translate"),
                            isOn: Binding(
                                get: { state.realtimeTranslateEnabled },
                                set: { onRealtimeTranslateToggle($0) }
                            )
                        )
                        .labelsHidden()
                        .toggleStyle(MeetingInlineSwitchStyle())

                        MeetingCaptureModeSelectorButton(
                            selection: state.captureMode,
                            isPickerPresented: state.isCaptureModePickerPresented,
                            onToggle: {
                                onToggleCaptureModePicker()
                            }
                        )
                    }

                    Rectangle()
                        .fill(.white.opacity(0.08))
                        .frame(width: 1, height: 18)

                    AnswerHeaderActionButton(
                        accessibilityLabel: state.isPaused ? AppLocalization.localizedString("Resume") : AppLocalization.localizedString("Pause"),
                        action: onTogglePause,
                        isEnabled: true
                    ) {
                        Image(systemName: state.isPaused ? "play.fill" : "pause.fill")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.9))
                    }

                    AnswerHeaderActionButton(
                        accessibilityLabel: AppLocalization.localizedString("Detail"),
                        action: onShowDetail,
                        isEnabled: true
                    ) {
                        Image(systemName: "list.bullet.rectangle")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.9))
                    }

                    AnswerHeaderActionButton(
                        accessibilityLabel: AppLocalization.localizedString("Collapse"),
                        action: onToggleCollapse,
                        isEnabled: true
                    ) {
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.9))
                    }
                } else {
                    AnswerHeaderActionButton(
                        accessibilityLabel: state.isPaused ? AppLocalization.localizedString("Resume") : AppLocalization.localizedString("Pause"),
                        action: onTogglePause,
                        isEnabled: true
                    ) {
                        Image(systemName: state.isPaused ? "play.fill" : "pause.fill")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.9))
                    }

                    AnswerHeaderActionButton(
                        accessibilityLabel: AppLocalization.localizedString("Expand"),
                        action: onToggleCollapse,
                        isEnabled: true
                    ) {
                        Image(systemName: "arrow.down.left.and.arrow.up.right")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.9))
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

        }
        .onChange(of: state.isCollapsed) { _, isCollapsed in
            if isCollapsed {
                onDismissCaptureModePicker()
            }
        }
    }

    private var transcriptContent: some View {
        MeetingTranscriptScrollView(
            segments: state.segments,
            onCopySegment: onCopySegment
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.top, 14)
    }

    private var realtimeTranslationLanguageDialog: some View {
        VStack(alignment: .leading, spacing: 14) {
            realtimeTranslationLanguageDialogHeader
            realtimeTranslationLanguageList
            realtimeTranslationLanguageDialogActions
        }
        .padding(16)
        .frame(width: 280)
        .frame(maxHeight: translationLanguageDialogMaxHeight)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.black.opacity(0.94))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.white.opacity(0.12), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.28), radius: 20, y: 12)
    }

    private var realtimeTranslationLanguageDialogHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("Choose Translation Language"))
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.white.opacity(0.92))

            Text(AppLocalization.localizedString("Realtime translation only translates Them segments."))
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.white.opacity(0.6))
        }
    }

    private var realtimeTranslationLanguageList: some View {
        ScrollView {
            VStack(spacing: 6) {
                ForEach(TranslationTargetLanguage.allCases) { language in
                    realtimeTranslationLanguageRow(language)
                }
            }
        }
        .frame(maxHeight: translationLanguageListMaxHeight)
    }

    private func realtimeTranslationLanguageRow(_ language: TranslationTargetLanguage) -> some View {
        let isSelected = state.realtimeTranslationDraftLanguageRaw == language.rawValue

        return Button {
            state.realtimeTranslationDraftLanguageRaw = language.rawValue
        } label: {
            HStack(spacing: 10) {
                Text(language.title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.white.opacity(0.92))

                Spacer(minLength: 8)

                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Color.accentColor.opacity(0.95))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(realtimeTranslationLanguageRowBackground(isSelected: isSelected))
            .overlay(realtimeTranslationLanguageRowBorder(isSelected: isSelected))
        }
        .buttonStyle(.plain)
    }

    private func realtimeTranslationLanguageRowBackground(isSelected: Bool) -> some View {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
            .fill(isSelected ? Color.accentColor.opacity(0.20) : .white.opacity(0.05))
    }

    private func realtimeTranslationLanguageRowBorder(isSelected: Bool) -> some View {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
            .strokeBorder(
                isSelected ? Color.accentColor.opacity(0.36) : .white.opacity(0.08),
                lineWidth: 1
            )
    }

    private var realtimeTranslationLanguageDialogActions: some View {
        HStack(spacing: 10) {
            Button(AppLocalization.localizedString("Cancel")) {
                onCancelRealtimeTranslationLanguage()
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white.opacity(0.94))
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(realtimeTranslationActionBackground(isPrimary: false))
            .overlay(realtimeTranslationActionBorder(isPrimary: false))

            Button(AppLocalization.localizedString("Start Translation")) {
                onConfirmRealtimeTranslationLanguage()
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white.opacity(0.94))
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(realtimeTranslationActionBackground(isPrimary: true))
            .overlay(realtimeTranslationActionBorder(isPrimary: true))
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    private func realtimeTranslationActionBackground(isPrimary: Bool) -> some View {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
            .fill(isPrimary ? Color.accentColor.opacity(0.22) : .white.opacity(0.06))
    }

    private func realtimeTranslationActionBorder(isPrimary: Bool) -> some View {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
            .strokeBorder(isPrimary ? Color.accentColor.opacity(0.35) : .white.opacity(0.1), lineWidth: 1)
    }

    private var meetingCloseConfirmationDialog: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(AppLocalization.localizedString("End this meeting transcription?"))
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.white.opacity(0.92))

            Text(AppLocalization.localizedString("Canceling will discard this meeting; finishing will save it and open Meeting Details."))
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.white.opacity(0.6))

            HStack(spacing: 10) {
                cancelMeetingButton
                finishMeetingButton
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(16)
        .frame(width: 340)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.black.opacity(0.94))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.white.opacity(0.12), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.28), radius: 20, y: 12)
    }

    private var cancelMeetingButton: some View {
        Button(AppLocalization.localizedString("Cancel Transcription")) {
            onConfirmCancelMeeting()
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white.opacity(0.94))
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.red.opacity(0.16))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.red.opacity(0.28), lineWidth: 1)
        )
    }

    private var finishMeetingButton: some View {
        Button(AppLocalization.localizedString("Finish Transcription")) {
            onConfirmFinishMeeting()
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white.opacity(0.94))
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.accentColor.opacity(0.22))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.accentColor.opacity(0.35), lineWidth: 1)
        )
    }

    private var cornerRadius: CGFloat {
        CGFloat(min(max(overlayCardCornerRadius, 0), 40))
    }

    private var cardOpacity: Double {
        Double(min(max(overlayCardOpacity, 0), 100)) / 100.0
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .fill(.black.opacity(cardOpacity))
    }

    private var showsHeaderWaveformLoader: Bool {
        state.isPresented && !state.isModelInitializing && !state.isRecording && !state.isPaused
    }

}

private struct MeetingCaptureModeSelectorBoundsPreferenceKey: PreferenceKey {
    static var defaultValue: Anchor<CGRect>?

    static func reduce(value: inout Anchor<CGRect>?, nextValue: () -> Anchor<CGRect>?) {
        value = nextValue() ?? value
    }
}

private struct MeetingCaptureModeSelectorButton: View {
    let selection: MeetingCaptureMode
    let isPickerPresented: Bool
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(spacing: 5) {
                Text(selection.title)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)

                Image(systemName: isPickerPresented ? "chevron.up" : "chevron.down")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.white.opacity(0.72))
            }
            .foregroundStyle(.white.opacity(0.92))
            .padding(.horizontal, 10)
            .frame(height: 24)
            .background(
                Capsule()
                    .fill(.white.opacity(0.08))
            )
            .overlay(
                Capsule()
                    .strokeBorder(
                        isPickerPresented ? Color.accentColor.opacity(0.28) : .white.opacity(0.12),
                        lineWidth: 1
                    )
            )
        }
        .buttonStyle(.plain)
        .fixedSize(horizontal: true, vertical: false)
        .accessibilityLabel(Text(AppLocalization.localizedString("Meeting Capture Mode")))
        .help(selection.accessibilityDescription)
        .anchorPreference(
            key: MeetingCaptureModeSelectorBoundsPreferenceKey.self,
            value: .bounds
        ) { anchor in
            anchor
        }
        .zIndex(isPickerPresented ? 1 : 0)
    }
}

private struct MeetingCaptureModePickerRevealModifier: ViewModifier {
    let opacity: Double
    let verticalScale: CGFloat

    func body(content: Content) -> some View {
        content
            .opacity(opacity)
            .scaleEffect(x: 1, y: verticalScale, anchor: .top)
    }
}

private extension AnyTransition {
    static var meetingCaptureModePickerReveal: AnyTransition {
        .modifier(
            active: MeetingCaptureModePickerRevealModifier(opacity: 0, verticalScale: 0.88),
            identity: MeetingCaptureModePickerRevealModifier(opacity: 1, verticalScale: 1)
        )
    }
}

private struct MeetingCaptureModePicker: View {
    private let pickerWidth: CGFloat = 224
    private let pickerRowHeight: CGFloat = 30

    let selection: MeetingCaptureMode
    let onSelect: (MeetingCaptureMode) -> Void

    var body: some View {
        VStack(spacing: 3) {
            ForEach(MeetingCaptureMode.allCases) { mode in
                Button {
                    onSelect(mode)
                } label: {
                    captureModeRow(for: mode)
                }
                .buttonStyle(.plain)
                .help(mode.accessibilityDescription)
            }
        }
        .padding(5)
        .frame(width: pickerWidth, alignment: .top)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.black.opacity(0.96))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(0.12), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.22), radius: 16, y: 10)
        .accessibilityLabel(Text(AppLocalization.localizedString("Meeting Capture Mode")))
    }

    private func captureModeRow(for mode: MeetingCaptureMode) -> some View {
        let isSelected = selection == mode
        let backgroundColor: Color = isSelected ? Color.accentColor.opacity(0.20) : .white.opacity(0.05)
        let borderColor: Color = isSelected ? Color.accentColor.opacity(0.36) : .white.opacity(0.08)

        return HStack(spacing: 7) {
            Text(mode.title)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.white.opacity(0.92))
                .lineLimit(1)

            Text(mode.sourceDescription)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(.white.opacity(0.52))
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer(minLength: 8)

            if isSelected {
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.accentColor.opacity(0.95))
            }
        }
        .padding(.horizontal, 10)
        .frame(height: pickerRowHeight)
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(backgroundColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(borderColor, lineWidth: 1)
        )
    }
}

private struct MeetingInlineSwitchStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        Button {
            configuration.isOn.toggle()
        } label: {
            ZStack(alignment: configuration.isOn ? .trailing : .leading) {
                Capsule(style: .continuous)
                    .fill(configuration.isOn ? Color.accentColor.opacity(0.92) : .white.opacity(0.10))
                    .frame(width: 36, height: 20)

                SVGPathShape(pathData: Self.translationIconPath)
                    .fill(.white.opacity(0.96))
                    .frame(width: 16, height: 16)
                    .padding(2)
            }
            .animation(.easeInOut(duration: 0.16), value: configuration.isOn)
        }
        .frame(width: 36, height: 20)
        .contentShape(Capsule(style: .continuous))
        .buttonStyle(.plain)
        .accessibilityAddTraits(.isButton)
    }

    private static let translationIconPath = "M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM17 17.47C15.29 17.47 13.69 16.73 12.41 15.36C10.96 16.67 9.07 17.47 7 17.47C6.59 17.47 6.25 17.13 6.25 16.72C6.25 16.31 6.59 15.97 7 15.97C10.47 15.97 13.34 13.22 13.71 9.7H12H7.01C6.6 9.7 6.26 9.36 6.26 8.95C6.26 8.54 6.6 8.21 7.01 8.21H11.25V7.28C11.25 6.87 11.59 6.53 12 6.53C12.41 6.53 12.75 6.87 12.75 7.28V8.21H14.44C14.46 8.21 14.48 8.2 14.5 8.2C14.52 8.2 14.54 8.21 14.56 8.21H16.99C17.4 8.21 17.74 8.55 17.74 8.96C17.74 9.37 17.4 9.71 16.99 9.71H15.21C15.06 11.42 14.42 12.99 13.44 14.27C14.44 15.38 15.69 15.98 17 15.98C17.41 15.98 17.75 16.32 17.75 16.73C17.75 17.14 17.41 17.47 17 17.47Z"
}
