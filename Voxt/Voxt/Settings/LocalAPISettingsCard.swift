// LocalAPISettingsCard.swift
// Provides the General settings card for the Local API.

import AppKit
import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

private func localizedKey(_ key: String) -> LocalizedStringKey {
    LocalizedStringKey(localized(key))
}

struct LocalAPISettingsCard: View {
    @AppStorage(AppPreferenceKey.localAPIServerEnabled) private var isEnabled = false
    @AppStorage(AppPreferenceKey.localAPIServerPort) private var port = LocalAPISettings.defaultPort
    @AppStorage(AppPreferenceKey.historySyncEnabled) private var historySyncEnabled = false
    @State private var portText = ""
    @State private var token = ""
    @State private var status: LocalAPIServerStatus = .off

    var body: some View {
        GeneralSettingsCard(title: localizedKey("Local API")) {
            GeneralToggleRow(
                title: localizedKey("Allow other apps to dictate through Voxt"),
                description: localizedKey("Scripts, shortcuts and tools on this Mac send audio over HTTP. Voxt transcribes it with your current model, enhancement and dictionary, and returns the text."),
                isOn: $isEnabled
            )

            if isEnabled {
                GeneralToggleRow(
                    title: localizedKey("Sync history with Voxt Mobile"),
                    description: localizedKey("Paired phones exchange text history when this and History are both on. Audio and meeting transcripts stay on this Mac."),
                    isOn: $historySyncEnabled
                )

                Text(statusText)
                    .font(.caption)
                    .foregroundStyle(statusIsError ? .red : .secondary)

                GeneralFieldRow(
                    title: localizedKey("Port"),
                    description: localizedKey("Voxt listens on 127.0.0.1 only.")
                ) {
                    TextField(String(LocalAPISettings.defaultPort), text: $portText)
                        .textFieldStyle(.plain)
                        .settingsFieldSurface(width: 90)
                        .onSubmit(applyPort)
                }

                GeneralFieldRow(
                    title: localizedKey("Access token"),
                    description: localizedKey("Send it as a Bearer token with every request. Regenerating it disconnects existing clients.")
                ) {
                    Text(maskedToken)
                        .font(.system(.caption, design: .monospaced))
                    Button(localized("Copy")) { copy(token) }
                        .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 28))
                    Button(localized("Regenerate")) {
                        token = LocalAPISettings.regenerateToken()
                        notifySettingsChanged()
                    }
                    .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 28))
                }
            }
        }
        .onAppear {
            portText = String(port)
            token = LocalAPISettings.token()
            refreshStatus()
        }
        .onChange(of: isEnabled) { _, _ in notifySettingsChanged() }
        .onReceive(NotificationCenter.default.publisher(for: .voxtLocalAPIStatusDidChange)) { _ in
            refreshStatus()
        }
    }

    private var maskedToken: String {
        guard token.count > 8 else { return token }
        return "\(token.prefix(4))…\(token.suffix(4))"
    }

    private var statusText: String {
        switch status {
        case .off: localized("Stopped")
        case .starting: localized("Starting…")
        case let .listening(port): String(format: localized("Listening on 127.0.0.1:%d"), port)
        case let .failed(reason): String(format: localized("Couldn't start: %@"), reason)
        }
    }

    private var statusIsError: Bool {
        if case .failed = status { return true }
        return false
    }

    private func applyPort() {
        guard let value = Int(portText.trimmingCharacters(in: .whitespaces)), (1024...65534).contains(value) else {
            portText = String(port)
            return
        }
        port = value
        notifySettingsChanged()
    }

    private func refreshStatus() {
        status = AppDelegate.shared?.localAPIServerStatus ?? .off
    }

    private func notifySettingsChanged() {
        NotificationCenter.default.post(name: .voxtLocalAPISettingsDidChange, object: nil)
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
