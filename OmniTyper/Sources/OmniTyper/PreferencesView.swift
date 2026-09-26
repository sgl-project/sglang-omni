// SPDX-License-Identifier: Apache-2.0
import AppKit
import Carbon
import ServiceManagement
import SwiftUI

struct PreferencesView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    @ObservedObject var shortcut: GlobalShortcut
    @ViewState private var microphones: [MicrophoneDevice] = []
    @ViewState private var captureMonitor: Any?
    @ViewState private var hotKeyMode: UnsafeMutableRawPointer?
    @ViewState private var capturing = false
    @ViewState private var capture = ShortcutCapture()
    @ViewState private var login = false
    @ViewState private var shortcutError: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            pageTitle(L("settings.title"), detail: L("settings.subtitle"))
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label(L("settings.keyboardAudio"), systemImage: "keyboard").font(.headline)
                    HStack {
                        Text(L("settings.shortcut")); Spacer()
                        Button(capturing ? L("settings.pressCombo") : model.shortcutLabel) { captureShortcut() }
                            .font(.system(.body, design: .monospaced)).disabled(model.isBusy)
                        Button(L("action.reset")) { saveShortcut(keyCode: 49, modifiers: 786432) }.disabled(model.isBusy)
                    }
                    if let message = shortcutError ?? shortcut.errorCode.map({ L($0) }) {
                        Label(message, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.red)
                    }
                    Text(L("settings.shortcutCheckNote")).font(.caption).foregroundStyle(.secondary)
                    Toggle(L("settings.holdToTalk"), isOn: $store.preferences.holdToTalk)
                    Text(L("settings.holdNote")).font(.caption).foregroundStyle(.secondary)
                    Divider()
                    Picker(L("settings.microphone"), selection: $store.preferences.microphoneUID) {
                        Text(L("settings.systemDefault")).tag("")
                        ForEach(microphones) { Text($0.name).tag($0.id) }
                    }
                    Toggle(L("settings.sounds"), isOn: $store.preferences.sounds)
                    Toggle(L("settings.autoPaste"), isOn: $store.preferences.autoPaste)
                    Text(L("settings.autoPasteNote")).font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label(L("settings.languages"), systemImage: "globe").font(.headline)
                    Picker(L("settings.speechLanguage"), selection: $store.preferences.language) {
                        Text(L("settings.detectAutomatically")).tag("")
                        ForEach(languages, id: \.self) { Text(L("language." + $0)).tag($0) }
                    }
                    Picker(L("settings.translateInto"), selection: $store.preferences.targetLanguage) { ForEach(languages, id: \.self) { Text(L("language." + $0)).tag($0) } }
                    Divider()
                    Picker(L("settings.interfaceLanguage"), selection: $store.preferences.uiLanguage) {
                        Text(L("settings.followSystem")).tag(String?.none)
                        ForEach(L10n.supported, id: \.self) { Text(L10n.displayName($0)).tag(String?.some($0)) }
                    }
                    Text(L("settings.interfaceLanguageNote")).font(.caption).foregroundStyle(.secondary)
                    Text(L("settings.languageNote")).font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label(L("settings.localModel"), systemImage: "cpu").font(.headline)
                    Text("Qwen3-ASR · 0.6B · MLX 4-bit").font(.subheadline)
                    Picker(L("settings.modelSource"), selection: $store.preferences.asrModel) {
                        Text("Hugging Face").tag("mlx-community/Qwen3-ASR-0.6B-4bit")
                        Text(L("settings.modelScope")).tag("aufklarer/Qwen3-ASR-0.6B-MLX-4bit")
                    }.disabled(model.isBusy)
                    Text(L("settings.modelNote")).font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button(L("settings.prepareASR")) { model.prepareModels() }.buttonStyle(.borderedProminent).disabled(model.isBusy)
                        Button(L("settings.unloadASR")) { model.releaseModels() }.disabled(model.isBusy)
                    }
                    if model.phase == .preparing { PreparationCard(model: model, worker: model.worker) }
                    DisclosureGroup(L("settings.runtime")) {
                        TextField(L("settings.python"), text: $store.preferences.pythonExecutable).textFieldStyle(.roundedBorder).padding(.top, 8)
                        Text(L("settings.runtimeNote")).font(.caption).foregroundStyle(.secondary)
                    }
                    Text(L("settings.downloadNote")).font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label(L("settings.textAPI"), systemImage: "network").font(.headline)
                    Text(L("settings.textAPINote")).font(.caption).foregroundStyle(.secondary)
                    TextField(L("settings.baseURL"), text: $store.preferences.textSettings.baseURL)
                        .textFieldStyle(.roundedBorder).accessibilityLabel(L("settings.baseURLLabel"))
                    HStack {
                        TextField(L("settings.modelName"), text: $store.preferences.textSettings.model)
                            .textFieldStyle(.roundedBorder).accessibilityLabel(L("settings.modelLabel"))
                        if !model.textModels.isEmpty {
                            Menu(L("settings.chooseModel")) {
                                ForEach(model.textModels, id: \.self) { name in
                                    Button(name) { store.preferences.textSettings.model = name }
                                }
                            }
                        }
                    }
                    SecureField(L("settings.apiKey"), text: $model.textAPIKey).textFieldStyle(.roundedBorder)
                    Text(L("settings.apiKeyNote")).font(.caption).foregroundStyle(.secondary)
                    Button(L("settings.connect")) { model.loadTextModels() }.disabled(model.isBusy)
                    DisclosureGroup(L("settings.requestOptions")) {
                        TextEditor(text: $store.preferences.textSettings.optionsJSON)
                            .font(.system(.caption, design: .monospaced)).frame(height: 90)
                            .accessibilityLabel(L("settings.requestOptionsLabel"))
                        Text(L("settings.requestOptionsNote")).font(.caption).foregroundStyle(.secondary)
                    }
                    Text(L("settings.privacyNote")).font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label(L("settings.privacyHistory"), systemImage: "lock.shield").font(.headline)
                    Toggle(L("settings.keepHistory"), isOn: $store.preferences.saveHistory)
                    Picker(L("settings.keepFor"), selection: $store.preferences.historyDays) {
                        Text(L("settings.retain24h")).tag(1); Text(L("settings.retain7d")).tag(7); Text(L("settings.retain30d")).tag(30)
                        Text(L("settings.retain1y")).tag(365); Text(L("settings.retainForever")).tag(0)
                    }.disabled(!store.preferences.saveHistory)
                    Toggle(L("settings.keepAudio"), isOn: $store.preferences.keepAudio).disabled(!store.preferences.saveHistory)
                    Text(L("settings.historyNote")).font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button(L("settings.openDataFolder")) { NSWorkspace.shared.open(store.directory) }
                        if let logs = Diagnostics.directory {
                            Button(L("settings.openLogs")) { NSWorkspace.shared.open(logs) }
                        }
                    }
                    Text(L("settings.logsNote")).font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label(L("settings.general"), systemImage: "gearshape").font(.headline)
                    Picker(L("settings.appearance"), selection: $store.preferences.appearance) { Text(L("appearance.system")).tag("system"); Text(L("appearance.light")).tag("light"); Text(L("appearance.dark")).tag("dark") }
                    Toggle(L("settings.openAtLogin"), isOn: Binding(get: { login }, set: { value in
                        do {
                            if value { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
                            login = SMAppService.mainApp.status == .enabled
                        } catch { model.error = L("settings.loginError", error.localizedDescription) }
                    }))
                    HStack {
                        Button(L("settings.micSettings")) { NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!) }
                        Button(L("settings.axSettings")) { model.requestAccessibility() }
                    }
                    Text(L("app.about")).font(.caption).foregroundStyle(.secondary)
                }
            }
        }.onAppear {
            microphones = AudioRecorder.devices()
            // Note (Jiaxin Deng): Query synchronous XPC status on appearance, not on every body pass.
            login = SMAppService.mainApp.status == .enabled
        }
            .onDisappear { endCapture() }
            .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in endCapture() }
    }
    private func captureShortcut() {
        endCapture(); capturing = true; shortcutError = nil
        model.beginShortcutCapture()
        // Note (Codex): Let the recorder receive reserved combinations so it can explain the conflict.
        hotKeyMode = PushSymbolicHotKeyMode(OptionBits(kHIHotKeyModeAllDisabledExceptUniversalAccess))
        captureMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { event in
            let modifierChange = event.type == .flagsChanged
            let outcome = modifierChange
                ? capture.flagsChanged(keyCode: event.keyCode, flags: event.modifierFlags)
                : capture.keyDown(keyCode: event.keyCode, flags: event.modifierFlags)
            switch outcome {
            case .pending: break
            case .cancel: endCapture()
            case .reject: shortcutError = L("shortcut.needsModifier")
            case let .record(keyCode, modifiers):
                saveShortcut(keyCode: keyCode, modifiers: modifiers)
            }
            // Note (Yifei Leng): Pass modifier changes through so AppKit keeps an accurate modifier state.
            return modifierChange ? event : nil
        }
    }
    private func saveShortcut(keyCode: UInt16, modifiers: UInt64) {
        do {
            try GlobalShortcut.validate(keyCode: keyCode, modifiers: modifiers)
            var preferences = store.preferences
            preferences.shortcutKeyCode = keyCode
            preferences.shortcutModifiers = modifiers
            store.preferences = preferences
            shortcutError = nil
            endCapture()
        } catch { shortcutError = error.localizedDescription }
    }
    private func endCapture() {
        if let captureMonitor { NSEvent.removeMonitor(captureMonitor) }
        if let hotKeyMode { PopSymbolicHotKeyMode(hotKeyMode) }
        hotKeyMode = nil
        captureMonitor = nil; capturing = false; capture = ShortcutCapture()
        model.endShortcutCapture()
    }
}
