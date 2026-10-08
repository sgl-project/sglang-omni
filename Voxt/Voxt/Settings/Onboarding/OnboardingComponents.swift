// OnboardingComponents.swift
// Provides Onboarding Components for onboarding settings.

import SwiftUI
import AVKit

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct OnboardingVideoPlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        view.videoGravity = .resizeAspect
        view.updatesNowPlayingInfoCenter = false
        player.pause()
        player.actionAtItemEnd = .pause
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
        player.rate = 0
    }
}

struct OnboardingStatusBadge: View {
    let status: OnboardingStepStatus
    let isSelected: Bool

    var body: some View {
        Text(status.titleKey)
            .font(.system(size: 10, weight: .semibold))
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(
                Capsule(style: .continuous)
                    .fill(status.tint.opacity(isSelected ? 0.18 : 0.12))
            )
            .foregroundStyle(status.tint)
    }
}

struct OnboardingSummaryCard: View {
    let title: LocalizedStringKey
    let lines: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.weight(.medium))

            ForEach(lines.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }, id: \.self) { line in
                Text(line)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .settingsCardSurface(cornerRadius: SettingsUIStyle.compactCornerRadius, fillOpacity: 1)
    }
}

struct OnboardingPermissionStatusBadge: View {
    let isGranted: Bool

    var body: some View {
        Text(isGranted ? LocalizedStringKey("Enabled") : LocalizedStringKey("Disabled"))
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                Capsule(style: .continuous)
                    .fill((isGranted ? Color.green : Color.orange).opacity(0.16))
            )
            .foregroundStyle(isGranted ? Color.green : Color.orange)
    }
}

struct OnboardingTabItem<Value: Hashable & Sendable>: Identifiable {
    let value: Value
    let title: LocalizedStringKey

    var id: String { String(describing: value) }
}

struct OnboardingSegmentedTabs<Value: Hashable & Sendable>: View {
    @Binding var selection: Value
    let items: [OnboardingTabItem<Value>]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(items) { item in
                Button {
                    selection = item.value
                } label: {
                    Text(item.title)
                        .padding(.horizontal, 8)
                }
                .buttonStyle(SettingsSegmentedButtonStyle(isSelected: selection == item.value))
            }
        }
        .padding(2)
        .fixedSize(horizontal: true, vertical: false)
        .settingsCardSurface(cornerRadius: SettingsUIStyle.compactCornerRadius, fillOpacity: 1)
    }
}

struct OnboardingExampleRow: View {
    let title: LocalizedStringKey
    let detail: LocalizedStringKey

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline.weight(.medium))
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

struct OnboardingFeatureSummaryRow: View {
    let iconKind: SettingsSidebarIconKind
    let title: LocalizedStringKey
    let detail: LocalizedStringKey
    let status: String
    var statusTint: Color = .secondary

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            SettingsSidebarIconView(kind: iconKind)
                .foregroundStyle(Color.accentColor)
                .frame(width: 18, height: 18)
                .frame(width: 34, height: 34)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.accentColor.opacity(0.10))
                )

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.subheadline.weight(.medium))
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 12)

            Text(status)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(statusTint)
                .padding(.horizontal, 8)
                .frame(height: 24)
                .background(
                    Capsule(style: .continuous)
                        .fill(statusTint.opacity(0.12))
                )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct LocalModelPickerCard<PickerContent: View>: View {
    let title: LocalizedStringKey
    let selectionTitle: String
    let selectionDescription: String
    let isInstalled: Bool
    var showsCardSurface: Bool = true
    var isInstalling: Bool = false
    var isPaused: Bool = false
    var isInstallEnabled: Bool = true
    let installLabel: LocalizedStringKey
    let openLabel: LocalizedStringKey
    var downloadStatus: ModelDownloadStatusSnapshot? = nil
    var errorMessage: String? = nil
    let onChoose: () -> Void
    @ViewBuilder let pickerContent: () -> PickerContent
    let onInstall: () -> Void
    let onOpen: () -> Void
    var onPause: (() -> Void)? = nil
    var onResume: (() -> Void)? = nil
    var onCancel: (() -> Void)? = nil
    var onUninstall: (() -> Void)? = nil
    @State private var isShowingUninstallConfirmation = false
    @State private var isRunningUninstall = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            pickerContent()

            Text(selectionDescription)
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                if isInstalled {
                    Text(localized("Installed"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.green)
                    Button(openLabel, action: onOpen)
                        .buttonStyle(SettingsPillButtonStyle())
                    if onUninstall != nil {
                        Button(isRunningUninstall ? localized("Uninstalling…") : localized("Uninstall"), role: .destructive) {
                            isShowingUninstallConfirmation = true
                        }
                            .buttonStyle(SettingsPillButtonStyle(tone: .destructive))
                            .disabled(isRunningUninstall)
                    }
                } else if isInstalling {
                    Text(localized("Downloading"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Color.accentColor)
                    if let onPause {
                        Button(localized("Pause"), action: onPause)
                            .buttonStyle(SettingsPillButtonStyle())
                    }
                    if let onCancel {
                        Button(localized("Cancel"), role: .destructive, action: onCancel)
                            .buttonStyle(SettingsPillButtonStyle(tone: .destructive))
                    }
                } else if isPaused {
                    Text(localized("Paused"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.orange)
                    Button(localized("Continue"), action: onResume ?? onInstall)
                        .buttonStyle(SettingsPillButtonStyle())
                    if let onCancel {
                        Button(localized("Cancel"), role: .destructive, action: onCancel)
                            .buttonStyle(SettingsPillButtonStyle(tone: .destructive))
                    }
                } else {
                    Text(localized("Not installed"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.orange)
                    Button(installLabel, action: onInstall)
                        .buttonStyle(SettingsPillButtonStyle())
                        .disabled(!isInstallEnabled)
                }
            }

            if let downloadStatus {
                ModelDownloadStatusView(status: downloadStatus)
            }

            if let errorMessage, !errorMessage.isEmpty {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .modifier(LocalModelPickerCardSurfaceModifier(isEnabled: showsCardSurface))
        .alert(localized("Uninstall Model?"), isPresented: $isShowingUninstallConfirmation) {
            Button(localized("Cancel"), role: .cancel) {}
            Button(localized("Uninstall"), role: .destructive) {
                guard let onUninstall else { return }
                isRunningUninstall = true
                Task { @MainActor in
                    await Task.yield()
                    onUninstall()
                    isRunningUninstall = false
                }
            }
        } message: {
            Text(localized("This removes the downloaded model files from this Mac. You can download them again later."))
        }
    }
}

private struct LocalModelPickerCardSurfaceModifier: ViewModifier {
    let isEnabled: Bool

    func body(content: Content) -> some View {
        if isEnabled {
            content
                .padding(12)
                .settingsCardSurface(cornerRadius: SettingsUIStyle.compactCornerRadius, fillOpacity: 1)
        } else {
            content
        }
    }
}

struct ProviderStatusRow: View {
    let title: String
    let status: String
    let onConfigure: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.medium))
                Spacer()
                Button(localized("Configure"), action: onConfigure)
                    .buttonStyle(SettingsPillButtonStyle())
            }

            Text(status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .settingsCardSurface(cornerRadius: SettingsUIStyle.compactCornerRadius, fillOpacity: 1)
    }
}
