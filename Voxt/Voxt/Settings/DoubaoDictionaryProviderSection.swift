// DoubaoDictionaryProviderSection.swift
// Provides Doubao Dictionary Provider Section for settings screens.

import SwiftUI

struct RemoteProviderDoubaoDictionarySection: View {
    @Binding var mode: String
    @Binding var enableRequestHotwords: Bool
    @Binding var enableRequestCorrections: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(AppLocalization.localizedString("Dictionary Boosting"))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            SettingsMenuPicker(
                selection: $mode,
                options: DoubaoDictionaryMode.allCases.map {
                    SettingsMenuOption(value: $0.rawValue, title: modeTitle($0))
                },
                selectedTitle: modeTitle(resolvedMode),
                width: 240
            )

            switch resolvedMode {
            case .off:
                Text(AppLocalization.localizedString("Do not send Voxt dictionary terms to Doubao."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

            case .requestScoped:
                Toggle(AppLocalization.localizedString("Send hotwords with request"), isOn: $enableRequestHotwords)
                    .toggleStyle(.switch)
                Toggle(AppLocalization.localizedString("Send corrections with request"), isOn: $enableRequestCorrections)
                    .toggleStyle(.switch)
                Text(AppLocalization.localizedString("Current active dictionary terms are sent with each Doubao ASR request."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var resolvedMode: DoubaoDictionaryMode {
        DoubaoDictionaryMode(rawValue: mode) ?? .requestScoped
    }

    private func modeTitle(_ mode: DoubaoDictionaryMode) -> String {
        switch mode {
        case .off:
            return AppLocalization.localizedString("Off")
        case .requestScoped:
            return AppLocalization.localizedString("Request Scoped")
        }
    }
}
