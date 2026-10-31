import SwiftUI
import UIKit
import VoxtMobileKit

/// Everyday settings only: Your Mac, Keyboard, Translation, Writing,
/// Microphone, Privacy, App, About. Developer mode and language variants sit under About › Advanced.
struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var showsDictionary = false
    @State private var showsPersonalStyle = false
    @State private var showsTextKeyboards = false
    @State private var confirmClear = false
    @State private var disableHistory = false
    @State private var confirmUnpair = false
    @State private var layouts = TextKeyboardState()
    @State private var pinyinFuzzy = false
    @State private var hasLearnedPinyin = false
    @AppStorage("appearance", store: UserDefaults(suiteName: DictationChannel.appGroup)) private var appearance = "system"
    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    private var interfaceLanguageName: String {
        switch interfaceLanguage {
        case "en": "English"
        case "zh-Hans": "简体中文"
        case "zh-Hant": "繁體中文"
        case "ja": "日本語"
        case "es": "Español"
        case "ko": "한국어"
        default: VoxtL10n.shared.string("System")
        }
    }

    private var appearanceName: String {
        VoxtL10n.shared.string(appearance == "dark" ? "Dark" : appearance == "light" ? "Light" : "System")
    }

    var body: some View {
        let _ = interfaceLanguage
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    yourMac
                    keyboard
                    translation
                    writing
                    microphone
                    privacy
                    app
                    about
                }
                .padding(16)
            }
            .background(VoxtTheme.Colors.screenBackground)
            .navigationTitle(VoxtL10n.shared.string("Settings"))
            .onAppear {
                model.refreshStoredState()
                layouts = TextKeyboardState.load(from: UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard)
                refreshPinyin()
            }
            .sheet(isPresented: $showsDictionary) { DictionaryView().environment(model) }
            .sheet(isPresented: $showsPersonalStyle) { PersonalStyleView() }
            .sheet(isPresented: $showsTextKeyboards, onDismiss: {
                layouts = TextKeyboardState.load(from: UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard)
                refreshPinyin()
            }) { KeyboardLayoutsView() }
            .confirmationDialog(VoxtL10n.shared.string(model.historySyncEnabled && !disableHistory ? "Delete these records from this iPhone and the paired Mac?" : "Clear saved history on this iPhone?"), isPresented: $confirmClear) {
                Button(VoxtL10n.shared.string(disableHistory ? "Turn off and clear" : "Clear history"), role: .destructive) {
                    if disableHistory { model.setHistoryRetention(.off) }
                    else { model.clearHistory() }
                }
            }
            .confirmationDialog(VoxtL10n.shared.string("Unpair this Mac?"), isPresented: $confirmUnpair, titleVisibility: .visible) {
                Button(VoxtL10n.shared.string("Unpair"), role: .destructive) {
                    model.unpair()
                    dismiss()
                }
                .accessibilityIdentifier("confirmUnpair")
            }
            .toolbar {
                Button(VoxtL10n.shared.string("Done")) { dismiss() }
            }
        }
        .tint(VoxtTheme.Colors.ink)
    }

    // MARK: Sections

    private var yourMac: some View {
        SettingsSection("Your Mac", id: "yourMac") {
            if let server = model.server {
                MacStatusCardView(card: MacStatusCard(macName: server.displayName,
                                                      state: AppModel.statusCardState(for: model.connection) ?? .connecting),
                                  onRetry: { Task { await model.checkConnection() } }, identifier: "pairedMacName", framed: false)
            }
            Button(VoxtL10n.shared.string("Pair again")) {
                model.unpair()
                dismiss()
            }
            .foregroundStyle(VoxtTheme.Colors.accent)
            .accessibilityIdentifier("pairAgain")
            Divider()
            Button(VoxtL10n.shared.string("Unpair"), role: .destructive) { confirmUnpair = true }
                .foregroundStyle(.red)
        }
    }

    private var keyboard: some View {
        SettingsSection("Keyboard", id: "keyboard") {
            Button { showsTextKeyboards = true } label: {
                SettingsRow(VoxtL10n.shared.string("Layouts"), value: layouts.enabled.map(\.shortLabel).joined(separator: ", "), accessory: .chevron)
            }
            .accessibilityIdentifier("Text keyboards")
            Divider()
            Toggle(VoxtL10n.shared.string("Haptic feedback"), isOn: Binding(
                get: { model.keyboardSettings.hapticFeedback }, set: { model.setHapticFeedback($0) }))
                .accessibilityIdentifier("hapticFeedback")
            // Only with 简体拼音 on; the keyboard reads both on its next conversion.
            if layouts.enabled.contains(.pinyin) {
                Divider()
                Toggle(VoxtL10n.shared.string("Fuzzy pinyin (z/zh, c/ch, s/sh)"), isOn: Binding(
                    get: { pinyinFuzzy },
                    set: { pinyinFuzzy = $0; PinyinSettings.setFuzzy($0, in: Self.groupDefaults) }))
                    .accessibilityIdentifier("pinyinFuzzy")
                Divider()
                Button(VoxtL10n.shared.string("Clear learned words")) {
                    PinyinSettings.clearLearnedWords(in: Self.groupContainer)
                    hasLearnedPinyin = false
                }
                .foregroundStyle(hasLearnedPinyin ? Color.red : Color.secondary)
                .disabled(!hasLearnedPinyin)
                .accessibilityIdentifier("clearPinyinLearnedWords")
            }
        }
    }

    private static var groupDefaults: UserDefaults { UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard }
    private static var groupContainer: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: DictationChannel.appGroup)
    }

    private func refreshPinyin() {
        pinyinFuzzy = PinyinSettings.fuzzy(in: Self.groupDefaults)
        hasLearnedPinyin = PinyinSettings.hasLearnedWords(in: Self.groupContainer)
    }

    private var translation: some View {
        SettingsSection("Translation", id: "translation") {
            // Destination links, not navigationDestination(for:): with the latter, Personal
            // dictionary's Add word alert closes without saving on iOS 26.3
            // (PreferencesUITests.testDictionaryAddSearchAndDeletePersists).
            NavigationLink { TranslationLanguagesView() } label: {
                SettingsRow(VoxtL10n.shared.string("Languages"),
                            value: model.keyboardSettings.translationTargets.map(TranslationLanguages.nativeName(for:)).joined(separator: ", "),
                            accessory: .chevron)
            }
            .accessibilityIdentifier("translationLanguages")
        }
    }

    private var writing: some View {
        SettingsSection("Writing", id: "writing") {
            VStack(alignment: .leading, spacing: 6) {
                Toggle(VoxtL10n.shared.string("Clean up my words"), isOn: Binding(
                    get: { model.cleansUpWords }, set: { model.setCleansUpWords($0) }))
                    .accessibilityIdentifier("cleanUpWords")
                SettingsCaption("Off: inserts your words as spoken, without cleanup. Faster.")
            }
            Menu {
                ForEach(CompositionOptions.Style.allCases, id: \.self) { style in
                    Button(style.title) { model.setStyle(style) }
                }
            } label: {
                SettingsRow(VoxtL10n.shared.string("Writing style"), value: model.preferences.style.title, accessory: .menu)
            }
            .accessibilityIdentifier("writingStyle")
            Divider()
            Button { showsDictionary = true } label: {
                SettingsRow(VoxtL10n.shared.string("Personal dictionary"), accessory: .chevron)
            }
            Divider()
            Button { showsPersonalStyle = true } label: {
                SettingsRow(VoxtL10n.shared.string("Learn my style"), accessory: .chevron)
            }
            .accessibilityIdentifier("Personal style")
        }
    }

    private var microphone: some View {
        SettingsSection("Microphone", id: "microphone") {
            Menu {
                ForEach(MicrophoneReadiness.allCases, id: \.self) { readiness in
                    Button(readiness.title) { model.setMicrophoneReadiness(readiness) }
                }
            } label: {
                SettingsRow(VoxtL10n.shared.string(MicrophoneReadiness.settingTitleKey), value: model.microphoneReadiness.title, accessory: .menu)
            }
            .accessibilityIdentifier("microphoneReadiness")
            .accessibilityLabel(VoxtL10n.shared.format("Keep the mic on after dictating: %@", model.microphoneReadiness.title))
            SettingsCaption(MicrophoneReadiness.settingCaptionKey)
            if model.sessionOpen {
                Button(VoxtL10n.shared.string("Turn off now")) { model.endSession() }
                    .foregroundStyle(VoxtTheme.Colors.accent)
            } else {
                SettingsCaption("Microphone is off.")
            }
        }
    }

    private var privacy: some View {
        SettingsSection("Privacy", id: "privacy") {
            VStack(alignment: .leading, spacing: 6) {
                Toggle(VoxtL10n.shared.string("Allow cloud models"), isOn: Binding(
                    get: { model.preferences.allowCloudModels }, set: { model.setAllowCloudModels($0) }))
                    .accessibilityIdentifier("allowCloudModels")
                SettingsCaption(model.preferences.allowCloudModels
                                ? "On: your Mac may send audio and text to its selected cloud providers."
                                : "Off: your voice and text stay on this iPhone and your Mac.")
            }
            Menu {
                ForEach(TranscriptHistory.Retention.allCases, id: \.self) { retention in
                    Button(retention.title) {
                        if retention == .off {
                            disableHistory = true
                            confirmClear = true
                        } else { model.setHistoryRetention(retention) }
                    }
                }
            } label: {
                SettingsRow(VoxtL10n.shared.string("Save history"), value: model.historyRetention.title, accessory: .menu)
            }
            .accessibilityIdentifier("historyRetention")
            .accessibilityLabel(VoxtL10n.shared.format("Save history: %@", model.historyRetention.title))
            if model.macServes(.historySync) {
                Divider()
                Toggle(VoxtL10n.shared.string("Sync history with Mac"), isOn: Binding(
                    get: { model.historySyncEnabled }, set: { model.setHistorySyncEnabled($0) }))
                    .disabled(model.historyRetention == .off)
                    .accessibilityIdentifier("historySyncToggle")
            }
            if model.historySyncEnabled && model.macServes(.historySync) {
                SettingsCaption("Enable history sync in Voxt Settings on your Mac too. Saved text merges between devices. Deleting a record removes it from both; automatic retention only clears this device.")
                Button(VoxtL10n.shared.string(model.isSyncingHistory ? "Syncing…" : "Sync now")) { Task { await model.syncHistory() } }
                    .disabled(model.isSyncingHistory)
                    .foregroundStyle(VoxtTheme.Colors.accent)
                    .accessibilityIdentifier("historySyncNow")
                if let message = model.historySyncMessage {
                    Text(VoxtL10n.shared.string(message)).font(VoxtTheme.Typography.caption).foregroundStyle(.secondary)
                        .accessibilityIdentifier("historySyncStatus")
                }
            }
            Divider()
            Button(VoxtL10n.shared.string(model.historySyncEnabled ? "Clear synced history" : "Clear phone history"), role: .destructive) {
                disableHistory = false
                confirmClear = true
            }
            .foregroundStyle(.red)
        }
    }

    private var app: some View {
        SettingsSection("App", id: "app") {
            Menu {
                Button(VoxtL10n.shared.string("System")) { interfaceLanguage = "" }
                Button("English") { interfaceLanguage = "en" }
                Button("简体中文") { interfaceLanguage = "zh-Hans" }
                Button("繁體中文") { interfaceLanguage = "zh-Hant" }
                Button("日本語") { interfaceLanguage = "ja" }
                Button("Español") { interfaceLanguage = "es" }
                Button("한국어") { interfaceLanguage = "ko" }
            } label: {
                SettingsRow(VoxtL10n.shared.string("Language"), value: interfaceLanguageName, accessory: .menu)
            }
            .accessibilityIdentifier("interfaceLanguage")
            Divider()
            Menu {
                Button(VoxtL10n.shared.string("System")) { appearance = "system" }
                Button(VoxtL10n.shared.string("Light")) { appearance = "light" }
                Button(VoxtL10n.shared.string("Dark")) { appearance = "dark" }
            } label: {
                SettingsRow(VoxtL10n.shared.string("Appearance"), value: appearanceName, accessory: .menu)
            }
            .accessibilityIdentifier("appearancePicker")
        }
    }

    private var about: some View {
        SettingsSection("About", id: "about") {
            SettingsRow(VoxtL10n.shared.string("Version"), value: Self.version, accessory: .none)
            Divider()
            sourceCode
            Divider()
            Button {
                model.setUpAgain()
                dismiss()
            } label: {
                SettingsRow(VoxtL10n.shared.string("Set up again"), accessory: .chevron)
            }
            .accessibilityIdentifier("setUpAgain")
            Divider()
            NavigationLink { AdvancedSettingsView() } label: {
                SettingsRow(VoxtL10n.shared.string("Advanced"), accessory: .chevron)
            }
            .accessibilityIdentifier("settingsAdvanced")
        }
    }

    /// The public commit a release was built from, opened on GitHub.
    @ViewBuilder private var sourceCode: some View {
        if let revision = SourceRevision(info: Bundle.main.infoDictionary) {
            Link(destination: revision.url) {
                SettingsRow(VoxtL10n.shared.string("Source code"), value: revision.short, accessory: .chevron)
            }
            .accessibilityIdentifier("sourceCode")
        } else {
            SettingsRow(VoxtL10n.shared.string("Source code"), value: VoxtL10n.shared.string("Development build"), accessory: .none)
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("sourceCode")
        }
    }

    private static var version: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "\(short) (\(build))"
    }
}

/// About › Advanced: language variants and Developer mode (still confirmed before it turns on).
private struct AdvancedSettingsView: View {
    @Environment(AppModel.self) private var model
    @State private var showsLanguageVariants = false
    @State private var confirmDeveloperMode = false
    @State private var showsDeveloperPrompts = false
    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        ScrollView {
            VStack(spacing: 16) {
                Button { showsLanguageVariants = true } label: {
                    SettingsRow(VoxtL10n.shared.string("Language variants"), accessory: .chevron).voxtCard()
                }
                .accessibilityIdentifier("variants")
                VStack(alignment: .leading, spacing: 8) {
                    Toggle(VoxtL10n.shared.string("Developer mode"), isOn: Binding(
                        get: { model.developerModeEnabled },
                        set: { enabled in
                            if enabled { confirmDeveloperMode = true } else { model.setDeveloperMode(false) }
                        }))
                        .accessibilityIdentifier("developerModeToggle")
                    if model.developerModeEnabled && model.macServes(.composePrompts) {
                        Button(VoxtL10n.shared.string("Prompts")) { showsDeveloperPrompts = true }
                            .foregroundStyle(VoxtTheme.Colors.accent)
                            .accessibilityIdentifier("developerPrompts")
                    }
                    SettingsCaption(model.developerModeEnabled
                                    ? "Your custom prompts are sent with each request. Custom prompts can lower quality."
                                    : "Lets you edit the prompts Voxt sends to your Mac. Asks to confirm before it turns on.")
                }
                .voxtCard()
            }
            .padding(16)
        }
        .foregroundStyle(VoxtTheme.Colors.ink)
        .background(VoxtTheme.Colors.screenBackground)
        .navigationTitle(VoxtL10n.shared.string("Advanced"))
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showsLanguageVariants) { LanguageVariantsView().environment(model) }
        .sheet(isPresented: $showsDeveloperPrompts) { DeveloperPromptsView().environment(model) }
        .alert(VoxtL10n.shared.string("Turn on Developer mode?"), isPresented: $confirmDeveloperMode) {
            Button(VoxtL10n.shared.string("Cancel"), role: .cancel) {}
            Button(VoxtL10n.shared.string("Turn On")) { model.setDeveloperMode(true) }
        } message: {
            Text(VoxtL10n.shared.string("Developer mode lets you edit the prompts Voxt uses. Custom prompts can lower quality."))
        }
    }
}

/// One card: a title, then rows.
private struct SettingsSection<Content: View>: View {
    let title: String
    let id: String
    @ViewBuilder var content: Content

    init(_ title: String, id: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.id = id
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(VoxtL10n.shared.string(title))
                .font(VoxtTheme.Typography.cardTitle)
                .accessibilityIdentifier("settingsSection-\(id)")
            content
        }
        .tint(Color.green)
        .foregroundStyle(VoxtTheme.Colors.ink)
        .frame(maxWidth: .infinity, alignment: .leading)
        .voxtCard()
    }
}

private struct SettingsRow: View {
    enum Accessory { case none, chevron, menu }
    let title: String
    var value: String?
    var accessory: Accessory

    init(_ title: String, value: String? = nil, accessory: Accessory) {
        self.title = title
        self.value = value
        self.accessory = accessory
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(title).foregroundStyle(VoxtTheme.Colors.ink)
            Spacer(minLength: 12)
            if let value, !value.isEmpty {
                // Two lines for long values ("Hasta que lo elimine" on an iPhone SE).
                Text(value).foregroundStyle(.secondary).lineLimit(2).multilineTextAlignment(.trailing)
            }
            switch accessory {
            case .none: EmptyView()
            case .chevron:
                Image(systemName: "chevron.right").font(.system(size: 13, weight: .semibold)).foregroundStyle(.tertiary)
            case .menu:
                Image(systemName: "chevron.up.chevron.down").font(.system(size: 12, weight: .semibold)).foregroundStyle(.secondary)
            }
        }
        .contentShape(Rectangle())
    }
}

private struct SettingsCaption: View {
    let key: String
    init(_ key: String) { self.key = key }
    var body: some View {
        Text(VoxtL10n.shared.string(key))
            .font(VoxtTheme.Typography.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

private struct KeyboardLayoutsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var state = TextKeyboardState()
    private let defaults = UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        NavigationStack {
            List {
                Section(VoxtL10n.shared.string("Enabled")) {
                    ForEach(Array(state.enabled.enumerated()), id: \.element) { index, layout in
                        HStack {
                            Text(layout.title)
                            Spacer()
                            Button { state.moveEnabled(from: index, to: index - 1); save() } label: { Image(systemName: "chevron.up") }
                                .disabled(index == 0)
                                .accessibilityLabel(VoxtL10n.shared.format("Move %@ up", layout.title))
                            Button { state.moveEnabled(from: index, to: index + 1); save() } label: { Image(systemName: "chevron.down") }
                                .disabled(index == state.enabled.count - 1)
                                .accessibilityLabel(VoxtL10n.shared.format("Move %@ down", layout.title))
                            Button { try? state.disable(layout); save() } label: { Image(systemName: "minus.circle") }
                                .disabled(state.enabled.count == 1)
                                .accessibilityLabel(VoxtL10n.shared.format("Remove %@", layout.title))
                                .accessibilityIdentifier("removeLayout-\(layout.rawValue)")
                        }
                        .buttonStyle(.borderless)
                    }
                }
                let available = TextKeyboardLayoutID.offered.filter { !state.enabled.contains($0) }
                if !available.isEmpty {
                    Section(VoxtL10n.shared.string("Add keyboard")) {
                        ForEach(available, id: \.self) { layout in
                            Button(layout.title, systemImage: "plus.circle") {
                                try? state.enable(rawValue: layout.rawValue)
                                save()
                            }
                            .accessibilityIdentifier("addLayout-\(layout.rawValue)")
                        }
                    }
                }
                Section {
                    Text(VoxtL10n.shared.string("The keyboard shows only these languages. Swipe the language row to switch layouts or return to voice. Hold a letter for accents."))
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle(VoxtL10n.shared.string("Keyboard languages"))
            .toolbar { Button(VoxtL10n.shared.string("Done")) { dismiss() } }
            .onAppear { state = TextKeyboardState.load(from: defaults) }
        }
        .tint(VoxtTheme.Colors.ink)
    }

    private func save() { state.save(to: defaults) }
}

private struct LanguageVariantsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var variants: [String] = []
    private let languages = ["en", "zh", "es", "fr", "pt"]

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        NavigationStack {
            Form {
                Section {
                    ForEach(languages, id: \.self) { base in
                        let selected = variants.first { $0.hasPrefix(base + "-") }
                        Menu {
                            Button(VoxtL10n.shared.string("Automatic")) { choose(nil, for: base) }
                            ForEach(CompositionOptions.supportedLanguageVariants.filter { $0.hasPrefix(base + "-") }, id: \.self) { code in
                                Button(CompositionPreferences.title(for: code)) { choose(code, for: base) }
                            }
                        } label: {
                            HStack {
                                Text(VoxtL10n.shared.locale.localizedString(forLanguageCode: base) ?? base)
                                Spacer()
                                Text(selected.map(CompositionPreferences.title(for:)) ?? VoxtL10n.shared.string("Automatic"))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .accessibilityIdentifier("variant-\(base)")
                    }
                } footer: {
                    Text(VoxtL10n.shared.string("Use your preferred spelling and Chinese script when writing in these languages. Translation uses the target language you choose."))
                }
            }
            .navigationTitle(VoxtL10n.shared.string("Language variants"))
            .toolbar { Button(VoxtL10n.shared.string("Done")) { dismiss() } }
            .onAppear { variants = model.languageVariants }
        }
        .tint(VoxtTheme.Colors.ink)
    }

    private func choose(_ value: String?, for base: String) {
        model.setLanguageVariant(value, for: base)
        variants = model.languageVariants
    }
}
