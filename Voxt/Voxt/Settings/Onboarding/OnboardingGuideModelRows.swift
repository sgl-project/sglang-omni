import SwiftUI
import AppKit

extension OnboardingGuideView {
    func localModelRow(
        title: String,
        repo: String,
        sizeText: String,
        ratingText: String,
        isSelected: Bool,
        isInstalled: Bool,
        isPaused: Bool,
        status: ModelDownloadStatusSnapshot?,
        errorMessage: String?,
        onSelect: @escaping () -> Void,
        onInstall: @escaping () -> Void,
        onPause: @escaping () -> Void,
        onCancel: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ModelLogoView(
                    key: ModelLogoKey.resolve(title: title, engine: repo),
                    fallbackTitle: title,
                    size: 24
                )

                VStack(alignment: .leading, spacing: 5) {
                    Text(title)
                        .font(.callout.weight(.semibold))
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        modelMetaPill(text: sizeText, systemImage: "externaldrive")
                        modelInstallStatusPill(
                            text: localInstallStatusText(isInstalled: isInstalled, status: status),
                            isInstalled: isInstalled,
                            isActive: status != nil
                        )
                        modelMetaPill(text: ratingText, systemImage: "star.fill")
                    }
                }

                Spacer(minLength: 6)

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.accentColor)
                }

                if status != nil {
                    Button(AppLocalization.localizedString("Cancel"), action: onCancel)
                        .buttonStyle(SettingsCompactActionButtonStyle())
                } else if isInstalled {
                    Button(isSelected ? AppLocalization.localizedString("Selected") : AppLocalization.localizedString("Select"), action: onSelect)
                        .buttonStyle(SettingsCompactActionButtonStyle())
                        .disabled(isSelected)
                } else {
                    Button(AppLocalization.localizedString("Install"), action: onInstall)
                        .buttonStyle(SettingsCompactActionButtonStyle())
                }
            }

            if let status {
                ModelDownloadStatusView(status: status)
                Button(AppLocalization.localizedString(isPaused ? "Resume" : "Pause"), action: isPaused ? onInstall : onPause)
                    .buttonStyle(SettingsPillButtonStyle())
            }

            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(isSelected ? Color.accentColor.opacity(0.08) : SettingsUIStyle.controlFillColor)
        )
    }

    func moreListButton(
        isExpanded: Bool,
        expandedCount: Int,
        collapsedCount: Int,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Text(isExpanded ? AppLocalization.localizedString("Less") : AppLocalization.localizedString("More"))
                    .font(.caption.weight(.semibold))
                Text("\(isExpanded ? collapsedCount : expandedCount)")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
                Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(SettingsCompactActionButtonStyle(height: 26, horizontalPadding: 8))
    }

    private func localInstallStatusText(isInstalled: Bool, status: ModelDownloadStatusSnapshot?) -> String {
        if let status {
            return status.titleText
        }
        return isInstalled ? AppLocalization.localizedString("Installed") : AppLocalization.localizedString("Not installed")
    }

    func onboardingRemoteLLMProviderTitle(_ provider: RemoteLLMProvider) -> String {
        switch provider {
        case .volcengine:
            return AppLocalization.localizedString("Doubao")
        default:
            return provider.title
        }
    }

    private func modelMetaPill(text: String, systemImage: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: systemImage)
                .font(.system(size: 8, weight: .semibold))
            Text(text)
                .font(.caption2.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.78)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(
            Capsule(style: .continuous)
                .fill(SettingsUIStyle.panelFillColor)
        )
    }

    private func modelInstallStatusPill(text: String, isInstalled: Bool, isActive: Bool) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .lineLimit(1)
            .minimumScaleFactor(0.72)
            .foregroundStyle(isInstalled ? Color.green : (isActive ? Color.accentColor : .secondary))
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(
                Capsule(style: .continuous)
                    .fill((isInstalled ? Color.green : (isActive ? Color.accentColor : Color.secondary)).opacity(0.12))
            )
    }

    func remoteProviderRow(
        title: String,
        isSelected: Bool,
        isConfigured: Bool,
        onSelect: @escaping () -> Void,
        onConfigure: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(alignment: .top, spacing: 8) {
                ModelLogoView(
                    key: ModelLogoKey.resolve(title: title, engine: "remote"),
                    fallbackTitle: title,
                    size: 22
                )
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.callout.weight(.semibold))
                        .lineLimit(1)
                    Text(isConfigured ? AppLocalization.localizedString("Configured") : AppLocalization.localizedString("Not configured"))
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(isConfigured ? Color.green : .secondary)
                        .lineLimit(1)
                }
                Spacer()

                HStack(spacing: 6) {
                    Button(isSelected ? AppLocalization.localizedString("Selected") : AppLocalization.localizedString("Select")) {
                        onSelect()
                    }
                    .buttonStyle(SettingsCompactActionButtonStyle(height: 24, horizontalPadding: 8))
                    .disabled(isSelected)

                    Button(AppLocalization.localizedString("Configure")) {
                        onConfigure()
                    }
                    .buttonStyle(SettingsCompactActionButtonStyle(height: 24, horizontalPadding: 8))
                }
            }
        }
        .padding(9)
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(isSelected ? Color.accentColor.opacity(0.08) : SettingsUIStyle.controlFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(isSelected ? Color.accentColor.opacity(0.26) : SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
    }
}
