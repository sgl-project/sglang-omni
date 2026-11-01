import SwiftUI
import CoreImage.CIFilterBuiltins
import ServiceManagement
import VoxtConnectionKit
import os

@main
struct VoxtConnectorApp: App {
    @NSApplicationDelegateAdaptor private var delegate: ConnectorAppDelegate
    @State private var model = ConnectorAppModel(
        runtimeURL: Bundle.main.bundleURL.appending(path: "Contents/Helpers/voxt-tailcat-host"),
        store: FileConnectorConfigurationStore(url: Self.storeURL),
        authorizer: Self.authorizer,
        preferences: Self.preferences)

    /// One instance, shared by the model and the Dock setting.
    static let preferences: any ConnectorPreferences = {
        #if DEBUG
        // UI tests start as a Connector Voxt has never approved, and leave no flag behind.
        if ProcessInfo.processInfo.environment["VOXT_CONNECTOR_TEST_STORE"] != nil { return SessionPreferences() }
        #endif
        return UserDefaultsConnectorPreferences()
    }()

    private static var authorizer: ConnectorLink {
        #if DEBUG
        // UI tests point the automatic request at their own socket path, never at a running Voxt.
        if let path = ProcessInfo.processInfo.environment["VOXT_CONNECTOR_TEST_SOCKET"] {
            return ConnectorLink(connect: { try ConnectorLink.connectToVoxt(path: path) })
        }
        #endif
        return ConnectorLink()
    }

    private static var storeURL: URL {
        #if DEBUG
        // UI tests keep their phone list out of the user's real one.
        if let path = ProcessInfo.processInfo.environment["VOXT_CONNECTOR_TEST_STORE"] {
            return URL(fileURLWithPath: path)
        }
        #endif
        return FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/Application Support/Voxt Connector/connections.json")
    }

    var body: some Scene {
        MenuBarExtra {
            ConnectorMenu(model: model, dock: delegate.dock)
        } label: {
            ConnectorMenuLabel(model: model, dock: delegate.dock, delegate: delegate)
        }
    }
}

#if DEBUG
private final class SessionPreferences: ConnectorPreferences, @unchecked Sendable {
    var wasApproved = false
    var showsInDock = true
}
#endif

/// Info.plist keeps LSUIElement, so the app starts without a Dock icon; the saved choice
/// adds it before launch finishes.
@MainActor private final class ConnectorAppDelegate: NSObject, NSApplicationDelegate {
    let dock = ConnectorDock(preferences: VoxtConnectorApp.preferences)
    var testHooks: [any NSObjectProtocol]?

    func applicationWillFinishLaunching(_ notification: Notification) { dock.applyAtLaunch() }

    func applicationDidBecomeActive(_ notification: Notification) {
        Logger(subsystem: ConnectorIdentity.bundleIdentifier, category: "window").notice("app became active")
    }

    /// The Settings window, made once at launch: SwiftUI's openWindow would build it on the click path,
    /// where the first open takes about 1 s. Showing it only orders it front.
    private var settingsWindow: PreparedWindow?

    func prepareSettings(model: ConnectorAppModel) {
        guard settingsWindow == nil else { return }
        let dock = self.dock
        let prepared = PreparedWindow(make: {
            let controller = NSHostingController(rootView: ConnectorSettings(model: model, dock: dock))
            controller.sizingOptions = [.preferredContentSize]
            let window = NSWindow(contentViewController: controller)
            window.title = "Voxt Connector"
            window.styleMask = [.titled, .closable, .miniaturizable]
            window.isReleasedWhenClosed = false
            window.center()
            return window
        })
        prepared.prepare()
        settingsWindow = prepared
        dock.openSettingsWindow = { prepared.show() }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        dock.reopen(hasVisibleWindows: hasVisibleWindows)
    }
}

/// The status item label is the only view SwiftUI renders at launch. A Connector Voxt has
/// approved before connects quietly; otherwise, or if that fails, Settings opens.
private struct ConnectorMenuLabel: View {
    let model: ConnectorAppModel
    let dock: ConnectorDock
    let delegate: ConnectorAppDelegate
    @State private var checkedLaunch = false

    var body: some View {
        Image(systemName: "waveform")
            .accessibilityLabel("Voxt Connector")
            .task {
                guard !checkedLaunch else { return }
                checkedLaunch = true
                delegate.prepareSettings(model: model)
                model.onPhoneRequest = { [dock] in dock.showSettings() }
                // Only with VOXT_CONNECTOR_TEST_HOOKS=1: on-screen tests drive the same handlers as
                // the menu item, the Show in Dock toggle, the Dock icon and a phone on the same Wi-Fi.
                delegate.testHooks = ConnectorTestHooks.install { [model, dock, delegate] command in
                    Logger(subsystem: ConnectorIdentity.bundleIdentifier, category: "window").notice("test hook: \(command.rawValue, privacy: .public)")
                    switch command {
                    case .openSettingsFromMenu: dock.showSettings()
                    case .showInDockOn: dock.showsInDock = true
                    case .showInDockOff: dock.showsInDock = false
                    case .dockIconClicked:
                        _ = delegate.applicationShouldHandleReopen(NSApplication.shared, hasVisibleWindows: false)
                    case .simulatePhoneRequest:
                        Task { _ = await model.askAboutPhone(code: "042917", deviceName: "Test iPhone") }
                    case .answerPhoneRequestDeny: model.answerPhoneRequest(false)
                    }
                }
                if !model.opensSettingsAtLaunch, await model.connectAtLaunch() { return }
                guard !model.isConnected else { return }
                dock.showSettings()
            }
    }
}

private struct ConnectorMenu: View {
    let model: ConnectorAppModel
    let dock: ConnectorDock

    var body: some View {
        Text(model.statusText)
        Divider()
        Button("Settings…") { dock.showSettings() }
        Button("Quit Voxt Connector") {
            model.disconnect()
            NSApplication.shared.terminate(nil)
        }
    }
}

private struct ConnectorSettings: View {
    @Bindable var model: ConnectorAppModel
    @Bindable var dock: ConnectorDock

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(model.statusText)
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .accessibilityIdentifier("connectorStatus")
            Form {
                Section("Voxt on this Mac") {
                    Text("In Voxt Settings › General › Local API, turn on “Allow other apps to dictate through Voxt”. Then click Connect and choose Allow in Voxt. To connect by hand, enter the port and Access token shown there.")
                        .foregroundStyle(.secondary)
                    TextField("Port", text: $model.port)
                        .accessibilityIdentifier("connectorPort")
                    SecureField("Access token", text: $model.token)
                        .accessibilityIdentifier("connectorToken")
                    HStack(spacing: 10) {
                        Button("Connect") { Task { await model.connect() } }
                            .disabled(model.isConnecting)
                            .keyboardShortcut(.defaultAction)
                            .accessibilityIdentifier("connectorConnect")
                        ConnectFeedbackView(feedback: model.connectFeedback)
                        if model.isAuthorizing {
                            Button("Cancel") { model.cancelAuthorization() }
                                .accessibilityIdentifier("connectorCancelAuthorization")
                        }
                    }
                }
                Section("iPhone") {
                    if let code = model.pairingCode {
                        PairingQRCode(code: code)
                        Text("Scan this QR code with the iPhone Camera, then open the link in Voxt. An iPhone on the same Wi-Fi can also connect from Voxt.")
                        Button("Cancel pairing") { model.cancelPairing() }
                            .accessibilityIdentifier("connectorCancelPairing")
                    } else {
                        HStack(spacing: 10) {
                            Button(model.isPreparingPairing ? "Preparing…" : "Add iPhone") {
                                Task { await model.beginPairing() }
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(!model.canPair)
                            .accessibilityIdentifier("connectorAddPhone")
                            if !model.canPair, !model.isPreparingPairing {
                                Text("Connect first.").foregroundStyle(.secondary)
                            }
                        }
                    }
                    if let error = model.pairingError { Text(error).foregroundStyle(.red) }
                }
                Section {
                    Toggle("Open at login", isOn: Binding(get: { model.opensAtLogin },
                                                          set: { model.setOpensAtLogin($0) }))
                        .toggleStyle(.switch)
                        .accessibilityIdentifier("connectorOpenAtLogin")
                    if let message = model.loginItemMessage {
                        Text(message).foregroundStyle(.secondary)
                        Button("Open Login Items") { SMAppService.openSystemSettingsLoginItems() }
                    }
                    Toggle("Show in Dock", isOn: $dock.showsInDock)
                        .toggleStyle(.switch)
                        .accessibilityIdentifier("connectorShowInDock")
                    if !dock.showsInDock {
                        Text("Open Voxt Connector from its icon in the menu bar.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            .onAppear {
                Logger(subsystem: ConnectorIdentity.bundleIdentifier, category: "window").notice("settings window appeared")
            }
            .task { await model.refreshLoginItem() }
            .sheet(item: Binding(get: { model.phoneRequest },
                                 set: { if $0 == nil { model.answerPhoneRequest(false) } })) { request in
                PhoneRequestSheet(request: request) { model.answerPhoneRequest($0) }
            }
        }
        .frame(width: 420)
    }
}

/// Shown when an iPhone on the same Wi-Fi asks to pair.
private struct PhoneRequestSheet: View {
    let request: ConnectorPhoneRequest
    let answer: (Bool) -> Void

    var body: some View {
        VStack(spacing: 14) {
            Text("Allow “\(request.deviceName)” to connect?")
                .font(.headline)
            Text(request.code)
                .font(.system(size: 34, weight: .semibold, design: .monospaced))
                .accessibilityIdentifier("connectorPhoneCode")
            Text("Allow only if your iPhone shows the same code.")
                .foregroundStyle(.secondary)
            HStack {
                Button("Don't Allow", role: .cancel) { answer(false) }
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("connectorPhoneDeny")
                Button("Allow") { answer(true) }
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("connectorPhoneAllow")
            }
        }
        .padding(24)
        .frame(width: 340)
    }
}

private struct ConnectFeedbackView: View {
    let feedback: ConnectorAppModel.ConnectFeedback

    var body: some View {
        Group {
            switch feedback {
            case .idle:
                EmptyView()
            case .waitingForApproval:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Click Allow in Voxt…")
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Click Allow in Voxt…")
            case .connecting:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Connecting…")
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Connecting…")
            case let .connected(text):
                Label(text, systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(text)
            case let .failed(text):
                Label(text, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(text)
            }
        }
        .accessibilityIdentifier("connectorConnectFeedback")
    }
}

private struct PairingQRCode: View {
    let code: ConnectorPairingCode
    var body: some View {
        if let qrImage {
            Image(nsImage: qrImage)
                .interpolation(.none)
                .resizable()
                .frame(width: 220, height: 220)
                .padding(12)
                .background(.white)
                .accessibilityLabel("iPhone pairing QR code")
                .accessibilityIdentifier("connectorPairingQR")
        } else {
            Text("Could not draw the QR code. Cancel and try again.")
        }
    }
    private var qrImage: NSImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(code.url.absoluteString.utf8)
        filter.correctionLevel = "M"
        guard let output = filter.outputImage,
              let image = CIContext().createCGImage(output, from: output.extent) else { return nil }
        return NSImage(cgImage: image, size: NSSize(width: output.extent.width, height: output.extent.height))
    }
}
