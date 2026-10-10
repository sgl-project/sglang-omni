// AboutSettingsView.swift
// Provides About Settings View for settings screens.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

private func localizedKey(_ key: String) -> LocalizedStringKey {
    LocalizedStringKey(localized(key))
}

struct AboutSettingsView: View {
    let appUpdateManager: AppUpdateManager
    let navigationRequest: SettingsNavigationRequest?
    @AppStorage(AppPreferenceKey.betaUpdatesEnabled) private var betaUpdatesEnabled = false

    private var appVersionText: String {
        let bundle = Bundle.main
        let shortVersion = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        let buildVersion = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String

        if let shortVersion, let buildVersion, !buildVersion.isEmpty {
            return "\(shortVersion) (\(buildVersion))"
        }
        if let shortVersion, !shortVersion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return shortVersion
        }
        if let buildVersion, !buildVersion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return buildVersion
        }
        return localized("Version metadata missing")
    }

    private var isBetaVersion: Bool {
        let bundle = Bundle.main
        let shortVersion = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? ""
        let buildVersion = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? ""
        let lowercasedVersionMetadata = "\(shortVersion) \(buildVersion)".lowercased()
        if lowercasedVersionMetadata.contains("beta") {
            return true
        }

        let versionComponents = shortVersion
            .split(separator: ".")
            .compactMap { Int($0) }
        guard versionComponents.count == 3,
              let major = versionComponents[safe: 0],
              let minor = versionComponents[safe: 1],
              let patch = versionComponents[safe: 2],
              let buildNumber = Int(buildVersion)
        else {
            return false
        }

        let releaseBuildBase = major * 100_000_000 + minor * 100_000 + patch * 100
        let releaseBuildSuffix = buildNumber - releaseBuildBase
        return (1...98).contains(releaseBuildSuffix)
    }

    private let websiteURL = URL(string: "https://voxt.actnow.dev")!
    private let githubURL = URL(string: "https://github.com/hehehai/voxt")!
    private let feedbackURL = URL(string: "https://github.com/hehehai/voxt/issues/new/choose")!
    private let authorURL = URL(string: "https://www.hehehai.cn/")!
    private let sponsorURL = URL(string: "https://voxt.actnow.dev/#pricing")!

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .center, spacing: 12) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(localizedKey("Version"))
                            .font(.body.weight(.semibold))
                            .foregroundStyle(.primary.opacity(0.92))
                        HStack(spacing: 8) {
                            Text(appVersionText)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(.secondary)

                            if isBetaVersion {
                                Text(localizedKey("Beta"))
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundStyle(Color.accentColor)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(
                                        Capsule(style: .continuous)
                                            .fill(Color.accentColor.opacity(0.11))
                                    )
                            }
                        }
                    }

                    Spacer(minLength: 0)

                    Button(localized("Check for Updates")) {
                            appUpdateManager.checkForUpdatesWithUserInterface()
                    }
                    .disabled(appUpdateManager.shouldDisableInteractiveUpdateTrigger)
                    .buttonStyle(SettingsPillButtonStyle())
                }

                GeneralToggleRow(
                    title: localizedKey("Beta Updates"),
                    description: localizedKey("Check beta appcast updates when Voxt checks for app updates."),
                    isOn: $betaUpdatesEnabled
                )
                .font(.caption)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .settingsNavigationAnchor(.aboutVoxt)

            HStack(alignment: .top, spacing: 14) {
                AboutInfoCard(title: localizedKey("Project")) {
                    AboutExternalLink(title: localized("Website"), destination: websiteURL)
                    AboutExternalLink(title: localized("GitHub"), destination: githubURL)
                    AboutExternalLink(title: localized("Feedback"), destination: feedbackURL)
                }
                .settingsNavigationAnchor(.aboutProject)

                AboutInfoCard(title: localizedKey("Author")) {
                    AboutExternalLink(title: "hehehai", destination: authorURL)
                    AboutExternalLink(title: localized("Sponsor"), destination: sponsorURL)
                }
                .settingsNavigationAnchor(.aboutAuthor)
            }

        }
        .onChange(of: betaUpdatesEnabled) { _, _ in
            appUpdateManager.betaUpdatesPreferenceDidChange()
        }
    }
}

private struct AboutInfoCard<Content: View>: View {
    let title: LocalizedStringKey
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.body.weight(.semibold))
                .foregroundStyle(.primary.opacity(0.92))

            VStack(alignment: .leading, spacing: 7) {
                content()
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, minHeight: 96, alignment: .topLeading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                .fill(SettingsUIStyle.controlFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
    }
}

private struct AboutExternalLink: View {
    let title: String
    let destination: URL

    var body: some View {
        Link(destination: destination) {
            HStack(spacing: 5) {
                Text(title)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: "arrow.up.forward")
                    .font(.system(size: 10, weight: .semibold))
            }
            .font(.system(size: 12.5, weight: .semibold))
            .foregroundStyle(Color.primary.opacity(0.84))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private extension Collection {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
