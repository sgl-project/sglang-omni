// FeatureSettingsRows.swift
// Provides Feature Settings Rows for feature settings.

import SwiftUI
import AppKit

struct FeatureStatusBadge: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.orange)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.orange.opacity(0.12))
            )
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(Color.orange.opacity(0.24), lineWidth: 1)
            )
    }
}

struct FeatureToggleRow: View {
    let title: String
    var badgeText: String? = nil
    let detail: String
    @Binding var isOn: Bool
    var isEmbedded = false

    var body: some View {
        FeatureRowScaffold(
            title: title,
            badgeText: badgeText,
            detail: detail,
            isEmbedded: isEmbedded
        ) {
            Toggle("", isOn: $isOn)
                .labelsHidden()
                .toggleStyle(.switch)
        }
    }
}

struct FeatureInlinePickerRow<PickerContent: View>: View {
    let title: String
    let detail: String
    var isEmbedded = false
    @ViewBuilder let picker: PickerContent

    init(title: String, detail: String, isEmbedded: Bool = false, @ViewBuilder picker: () -> PickerContent) {
        self.title = title
        self.detail = detail
        self.isEmbedded = isEmbedded
        self.picker = picker()
    }

    var body: some View {
        FeatureRowScaffold(
            title: title,
            detail: detail,
            isEmbedded: isEmbedded
        ) {
            picker
        }
    }
}

struct FeatureInlineTextFieldRow: View {
    let title: String
    let detail: String
    @Binding var text: String
    let placeholder: String
    let width: CGFloat
    var isEmbedded = false

    var body: some View {
        FeatureRowScaffold(
            title: title,
            detail: detail,
            isEmbedded: isEmbedded
        ) {
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .settingsFieldSurface(width: width)
                .multilineTextAlignment(.leading)
        }
    }
}

struct FeatureDirectorySelectionRow: View {
    private let pathFieldWidth: CGFloat = 184
    private let actionButtonWidth: CGFloat = 26

    let title: String
    let detail: String
    let path: String
    let buttonTitle: String
    let action: () -> Void
    var isEmbedded = false

    var body: some View {
        FeatureRowScaffold(
            title: title,
            detail: detail,
            spacerMinLength: 12,
            isEmbedded: isEmbedded
        ) {
            HStack(alignment: .center, spacing: 8) {
                Text(path)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .frame(width: pathFieldWidth, alignment: .leading)
                    .settingsFieldSurface(width: pathFieldWidth, minHeight: 32)

                Button(action: action) {
                    Text(buttonTitle)
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                        .frame(minWidth: actionButtonWidth)
                }
                .buttonStyle(SettingsPillButtonStyle())
            }
        }
    }
}

struct FeatureEmbeddedFieldGroup<Content: View>: View {
    let spacing: CGFloat
    @ViewBuilder let content: Content

    init(
        spacing: CGFloat = 20,
        @ViewBuilder content: () -> Content
    ) {
        self.spacing = spacing
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: spacing) {
            content
        }
    }
}

struct FeatureDisclosureSection<Content: View>: View {
    let title: String
    var badgeText: String? = nil
    let detail: String
    let embeddedSpacing: CGFloat = 12
    var onExpand: (() -> Void)? = nil
    @ViewBuilder let content: Content

    @State private var isExpanded: Bool

    init(
        title: String,
        badgeText: String? = nil,
        detail: String = "",
        onExpand: (() -> Void)? = nil,
        initiallyExpanded: Bool = false,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.badgeText = badgeText
        self.detail = detail
        self.onExpand = onExpand
        self.content = content()
        _isExpanded = State(initialValue: initiallyExpanded)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Button(action: toggleExpanded) {
                FeatureRowScaffold(
                    title: title,
                    badgeText: badgeText,
                    detail: detail,
                    isEmbedded: false
                ) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 16, height: 16)
                }
                .contentShape(Rectangle())
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)

            if isExpanded {
                FeatureEmbeddedFieldGroup(spacing: embeddedSpacing) {
                    content
                }
                .clipped()
                .transition(.opacity)
            }
        }
        .clipped()
    }

    private func toggleExpanded() {
        let shouldExpand = !isExpanded
        withAnimation(.easeInOut(duration: 0.16)) {
            isExpanded.toggle()
        }
        guard shouldExpand else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.18) {
            onExpand?()
        }
    }
}

struct FeatureSelectorRow: View {
    let title: String
    let value: String
    let action: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            Text(title)
                .font(.body.weight(.semibold))
                .foregroundStyle(.primary.opacity(0.92))
            Spacer(minLength: 0)
            SettingsSelectionButton(width: 280, action: action) {
                Text(value)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
        }
    }
}

struct SettingsShortcutCaptureField: View {
    let title: LocalizedStringKey
    let hotkey: HotkeyPreference.Hotkey
    let isRecording: Bool
    let isPendingConfirmation: Bool
    let distinguishModifierSides: Bool
    var displayTextOverride: String? = nil
    var isReadOnly: Bool = false
    var modeButtonTitle: String? = nil
    var isModeButtonSelected = false
    var showsTitle = true
    var onModeButtonToggle: (() -> Void)? = nil
    var controlWidth: CGFloat = 320
    let onFocus: () -> Void
    let onReset: () -> Void
    let onCancelPending: () -> Void
    let onConfirmPending: () -> Void
    @State private var isHovered = false

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            if showsTitle {
                Text(title)
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.primary.opacity(0.92))
                Spacer()
            }

            HStack(spacing: 8) {
                if let modeButtonTitle, let onModeButtonToggle {
                    Button(action: onModeButtonToggle) {
                        Text(featureSettingsLocalized(modeButtonTitle))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(isModeButtonSelected ? .white : .secondary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(isModeButtonSelected ? Color.accentColor : Color.secondary.opacity(0.10))
                            )
                    }
                    .buttonStyle(.plain)
                }

                Text(
                    displayTextOverride
                    ?? (isRecording && !isPendingConfirmation
                        ? featureSettingsLocalized("Listening...")
                        : HotkeyPreference.displayString(for: hotkey, distinguishModifierSides: distinguishModifierSides))
                )
                .font(.system(.body, design: .rounded))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.middle)
                .minimumScaleFactor(0.9)
                .layoutPriority(1)
                .frame(maxWidth: .infinity, alignment: .leading)

                if isPendingConfirmation {
                    Button(featureSettingsLocalized("Cancel"), action: onCancelPending)
                        .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 8, height: 24))
                    Button(featureSettingsLocalized("Confirm"), action: onConfirmPending)
                        .buttonStyle(SettingsPrimaryButtonStyle(horizontalPadding: 8, height: 24))
                } else if isRecording {
                    Button(featureSettingsLocalized("Cancel"), action: onCancelPending)
                        .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 8, height: 24))
                } else if !isReadOnly {
                    Button(action: onReset) {
                        Image(systemName: "arrow.counterclockwise")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help(Text(featureSettingsLocalized("Reset shortcut")))
                }
            }
            .frame(minHeight: 18)
            .padding(.horizontal, 10)
            .frame(height: 34)
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .fill(SettingsUIStyle.controlFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .strokeBorder(isHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous))
            .onHover { isHovered = $0 }
            .onTapGesture {
                guard !isReadOnly else { return }
                onFocus()
            }
            .frame(width: controlWidth, alignment: .trailing)
        }
    }
}

private struct FeatureRowChromeModifier: ViewModifier {
    let isEmbedded: Bool

    func body(content: Content) -> some View {
        content
    }
}

private struct FeatureRowScaffold<TrailingContent: View>: View {
    let title: String
    let badgeText: String?
    let detail: String
    var spacerMinLength: CGFloat = 0
    let isEmbedded: Bool
    @ViewBuilder let trailingContent: TrailingContent

    init(
        title: String,
        badgeText: String? = nil,
        detail: String,
        spacerMinLength: CGFloat = 0,
        isEmbedded: Bool,
        @ViewBuilder trailingContent: () -> TrailingContent
    ) {
        self.title = title
        self.badgeText = badgeText
        self.detail = detail
        self.spacerMinLength = spacerMinLength
        self.isEmbedded = isEmbedded
        self.trailingContent = trailingContent()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            FeatureRowLabelStack(title: title, badgeText: badgeText, detail: detail)
                .frame(maxWidth: .infinity, alignment: .leading)
            Spacer(minLength: spacerMinLength)
            trailingContent
                .fixedSize(horizontal: true, vertical: false)
        }
        .modifier(FeatureRowChromeModifier(isEmbedded: isEmbedded))
    }
}

private struct FeatureRowLabelStack: View {
    let title: String
    let badgeText: String?
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .center, spacing: 8) {
                Text(title)
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.primary.opacity(0.92))

                if let badgeText, !badgeText.isEmpty {
                    FeatureStatusBadge(text: badgeText)
                }

                Spacer(minLength: 0)
            }
            if !detail.isEmpty {
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
