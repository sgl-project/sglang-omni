import SwiftUI
import VoxtKit

/// Settings: the models first (what the features need), then the
/// features, the languages and the dictionary.
struct SettingsView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Form {
            Section(L("Models")) {
                NavigationLink(L("Language models")) { LanguageModelSettings(model: model) }
                    .accessibilityIdentifier("settings.languageModels")
                NavigationLink(L("Speech models")) { SpeechSettings(model: model) }
                    .accessibilityIdentifier("settings.speechModels")
            }
            Section(L("Features")) {
                NavigationLink(L("Transcribe")) {
                    FeatureSettingsView(title: L("Transcribe"), feature: $model.settings.transcribe, structure: true)
                }
                NavigationLink(L("Translate")) { TranslateSettings(model: model) }
                NavigationLink(L("Rewrite")) {
                    FeatureSettingsView(title: L("Rewrite"), feature: $model.settings.rewrite, structure: false)
                }
            }
            Section(L("Languages")) {
                LanguagePicker(model: model, title: L("Main language"), code: $model.settings.general.mainLanguage, automatic: true)
                NavigationLink(L("Other languages I speak")) { OtherLanguages(model: model) }
            }
            Section(L("Dictionary")) {
                NavigationLink(L("Dictionary")) { DictionarySettings(model: model) }
                NavigationLink(L("Snippets")) { SnippetSettings(model: model) }
            }
            Section {
                Text(L("Your speech is recognised on this iPhone unless you choose a remote speech model. Text goes only to the language model you set up; API keys stay in this iPhone's Keychain."))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle(L("Settings"))
    }
}

struct LanguageModelSettings: View {
    @ObservedObject var model: AppModel
    @State private var testing = false
    @State private var testResult: String?

    var body: some View {
        let lm = $model.settings.languageModel
        let preset = model.preset
        Form {
            Section {
                Picker(L("Service"), selection: lm.preset) {
                    ForEach(model.presets) { Text($0.name).tag($0.id) }
                }
                .accessibilityIdentifier("lm.service")
                TextField(preset?.baseURL ?? L("API address, https://…/v1"), text: lm.baseURL)
                    .plainInput()
                    .accessibilityIdentifier("lm.baseURL")
                TextField(preset?.defaultModel ?? L("Model"), text: lm.model)
                    .plainInput()
                    .accessibilityIdentifier("lm.model")
                if let models = preset?.models, !models.isEmpty {
                    Picker(L("Suggested models"), selection: lm.model) {
                        Text(L("Default")).tag("")
                        ForEach(models, id: \.self) { Text($0).tag($0) }
                    }
                }
            } footer: {
                Text(L("Leave the address and the model empty to use the service's defaults."))
            }
            Section {
                KeyField(model: model, title: preset?.keyOptional == true ? L("API key (optional)") : L("API key"),
                         account: KeyAccount.languageModel(model.settings.languageModel.preset), identifier: "lm.key")
                if let url = preset?.keyURL.flatMap(URL.init(string:)) {
                    Link(L("Get an API key"), destination: url)
                }
            }
            Section {
                Button(testing ? L("Testing…") : L("Test connection")) { Task { await test() } }
                    .disabled(testing)
                    .accessibilityIdentifier("lm.test")
                if let testResult {
                    Text(testResult).font(.callout).foregroundStyle(.secondary)
                        .accessibilityIdentifier("lm.result")
                }
            }
        }
        .navigationTitle(L("Language models"))
        .onChange(of: model.settings.languageModel.preset) { _, _ in
            model.settings.languageModel.baseURL = ""
            model.settings.languageModel.model = ""
            testResult = nil
        }
    }

    private func test() async {
        testing = true
        defer { testing = false }
        do {
            let result = try await model.core.testProvider(model.settings.languageModel.provider(keys: model.keys))
            testResult = L("Connected in %d ms.", result.latencyMillis)
        } catch {
            testResult = L("Not connected: %@", error.localizedDescription)
        }
    }
}

/// An API key field. The Keychain is read when the field appears (or its
/// account changes) and written as the key changes; a key the Keychain
/// refuses says so under the field instead of just vanishing.
struct KeyField: View {
    var model: AppModel
    var title: String
    var account: String
    var identifier: String
    @State private var key = ""
    @State private var saved = ""
    @State private var refused: OSStatus?

    var body: some View {
        SecureField(title, text: $key)
            .plainInput()
            .accessibilityIdentifier(identifier)
            .task(id: account) {
                key = model.keys.key(account) ?? ""
                saved = key
                refused = nil
            }
            .onChange(of: key) { _, value in
                guard value != saved else {
                    refused = nil // back at the key the Keychain holds
                    return
                }
                refused = model.saveKey(value, for: account)
                if refused == nil {
                    saved = value
                }
            }
        if let refused {
            Text(L("The key could not be saved in this iPhone's Keychain (error %d).", Int(refused)))
                .font(.footnote)
                .foregroundStyle(.orange)
        }
    }
}

struct SpeechSettings: View {
    @ObservedObject var model: AppModel

    var body: some View {
        let speech = $model.settings.speech
        Form {
            Section {
                Picker(L("Speech model"), selection: speech.engine) {
                    Text(L("On this iPhone")).tag(SpeechEngineKind.onDevice)
                        .accessibilityIdentifier("speech.onDevice")
                    Text(L("Remote speech service")).tag(SpeechEngineKind.remote)
                        .accessibilityIdentifier("speech.remote")
                }
                .pickerStyle(.inline)
                .labelsHidden()
            }
            switch model.settings.speech.engine {
            case .onDevice:
                Section {
                    Toggle(L("Only on this iPhone"), isOn: speech.requireOffline)
                } footer: {
                    Text(L("Uses Apple's speech recognition. With \"Only on this iPhone\" the audio never leaves the device; a language the iPhone cannot recognise offline then fails instead. Voxt's own speech models on the iPhone come in a later version."))
                }
            case .remote:
                Section {
                    TextField(L("Address, default https://api.openai.com/v1"), text: speech.endpoint)
                        .plainInput()
                        .accessibilityIdentifier("speech.endpoint")
                    TextField(L("Model, default whisper-1"), text: speech.model)
                        .plainInput()
                    KeyField(model: model, title: L("API key"), account: KeyAccount.speech, identifier: "speech.key")
                } footer: {
                    Text(L("Any OpenAI-compatible transcription API. The recording goes only to this service."))
                }
            }
        }
        .navigationTitle(L("Speech models"))
    }
}

struct FeatureSettingsView: View {
    var title: String
    @Binding var feature: AppSettings.FeatureSettings
    var structure: Bool

    var body: some View {
        Form {
            Section {
                Toggle(L("Clean up speech"), isOn: $feature.cleanup)
                if structure {
                    Toggle(L("Lists and paragraphs"), isOn: $feature.structure)
                        .disabled(!feature.cleanup)
                }
            } footer: {
                Text(L("The language model removes fillers and false starts and fixes punctuation, without changing what you said."))
            }
            PromptSection(prompt: $feature.prompt)
        }
        .navigationTitle(title)
    }
}

struct TranslateSettings: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Form {
            Section {
                LanguagePicker(model: model, title: L("Translate into"), code: $model.settings.translate.targetLanguage, automatic: false)
                Toggle(L("Clean up speech"), isOn: $model.settings.translate.cleanup)
            }
            PromptSection(prompt: $model.settings.translate.prompt)
        }
        .navigationTitle(L("Translate"))
    }
}

struct PromptSection: View {
    @Binding var prompt: String

    var body: some View {
        Section {
            TextEditor(text: $prompt).frame(minHeight: 100)
        } header: {
            Text(L("Your rules"))
        } footer: {
            Text(L("Added to Voxt's rules for this feature, for example \"Write numbers as digits\"."))
        }
    }
}

struct LanguagePicker: View {
    @ObservedObject var model: AppModel
    var title: String
    @Binding var code: String
    var automatic: Bool

    var body: some View {
        Picker(title, selection: $code) {
            if automatic {
                Text(L("Same as this iPhone")).tag("")
            }
            ForEach(model.languages) { Text($0.displayName(chinese: showsChinese)).tag($0.code) }
        }
    }
}

struct OtherLanguages: View {
    @ObservedObject var model: AppModel
    @State private var search = ""

    var body: some View {
        let chosen = Set(model.settings.general.otherLanguages)
        List(model.languages.filter { search.isEmpty || $0.displayName(chinese: showsChinese).localizedCaseInsensitiveContains(search) }) { language in
            Button {
                if chosen.contains(language.code) {
                    model.settings.general.otherLanguages.removeAll { $0 == language.code }
                } else {
                    model.settings.general.otherLanguages.append(language.code)
                }
            } label: {
                HStack {
                    Text(language.displayName(chinese: showsChinese)).foregroundStyle(.primary)
                    Spacer()
                    if chosen.contains(language.code) {
                        Image(systemName: "checkmark").foregroundStyle(Color.accentColor)
                    }
                }
            }
        }
        .searchable(text: $search)
        .navigationTitle(L("Other languages I speak"))
    }
}
