// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct HistoryView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    @ViewState private var query = ""
    @ViewState private var filter = "all"
    @ViewState private var editing: HistoryEntry?
    @ViewState private var confirmDelete = false
    private var entries: [HistoryEntry] {
        store.history.filter { (filter == "all" || $0.mode.rawValue == filter) && (query.isEmpty || $0.text.localizedCaseInsensitiveContains(query) || $0.appName.localizedCaseInsensitiveContains(query)) }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            pageTitle(L("history.title"), detail: L("history.subtitle"))
            HStack {
                TextField(L("history.search"), text: $query).textFieldStyle(.roundedBorder)
                Picker(L("history.filter"), selection: $filter) {
                    Text(L("history.allModes")).tag("all")
                    ForEach(VoiceMode.allCases) { Text($0.title).tag($0.rawValue) }
                }.labelsHidden().frame(width: 135)
                Menu {
                    Button(L("history.export")) { FileActions.exportHistory(store.history) }
                    Button(L("history.deleteAll"), role: .destructive) { confirmDelete = true }
                } label: { Image(systemName: "ellipsis.circle") }.menuStyle(.borderlessButton).frame(width: 28)
            }
            if entries.isEmpty {
                ContentUnavailableView(query.isEmpty ? L("history.emptyTitle") : L("history.emptySearch"), systemImage: "text.bubble", description: Text(query.isEmpty ? L("history.emptyBody") : L("history.emptySearchBody")))
                    .frame(maxWidth: .infinity, minHeight: 230)
            }
            LazyVStack(spacing: 14) {
                ForEach(entries) { entry in
                    Card {
                        VStack(alignment: .leading, spacing: 13) {
                            HStack {
                                Label(entry.mode.title, systemImage: entry.mode.icon).font(.system(size: 11, weight: .semibold)).foregroundStyle(accent)
                                Text(L("history.appSuffix", entry.appName)).font(.system(size: 11)).foregroundStyle(.secondary)
                                Spacer()
                                Text(entry.date, format: .dateTime.month(.abbreviated).day().hour().minute()).font(.system(size: 10)).foregroundStyle(.secondary)
                            }
                            Text(entry.text).font(.system(size: 14)).lineSpacing(4).textSelection(.enabled)
                            if let warning = entry.warning, !warning.isEmpty { Text(warning).font(.system(size: 11)).foregroundStyle(.orange) }
                            HStack {
                                Text(L("history.meta", String(entry.units), String(Int(entry.duration)))).font(.system(size: 10)).foregroundStyle(.tertiary)
                                Spacer()
                                Button { TextInsertion.copy(entry.text) } label: { Label(L("action.copy"), systemImage: "doc.on.doc") }
                                Menu {
                                    Button(L("history.correct")) { editing = entry }
                                    if store.audioURL(for: entry) != nil {
                                        Button(L("history.retryAudio")) { model.retry(entry) }.disabled(model.isBusy)
                                        Button(L("history.exportAudio")) { if let url = store.audioURL(for: entry) { FileActions.exportAudio(url) } }
                                    }
                                    Button(L("action.delete"), role: .destructive) { store.delete([entry.id]) }
                                } label: { Image(systemName: "ellipsis") }.menuStyle(.borderlessButton).frame(width: 20)
                            }.controlSize(.small)
                            if entry.rawText != entry.text {
                                DisclosureGroup(L("home.originalTranscript")) { Text(entry.rawText).textSelection(.enabled).padding(.top, 6) }.font(.system(size: 11)).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }.sheet(item: $editing) { entry in CorrectionView(store: store, entry: entry) }
            .confirmationDialog(L("history.confirmDelete"), isPresented: $confirmDelete) {
                Button(L("history.deleteAllConfirm"), role: .destructive) { store.delete(Set(store.history.map(\.id))) }
            }
    }
}

private struct CorrectionView: View {
    @ObservedObject var store: AppStore
    let entry: HistoryEntry
    @Environment(\.dismiss) var dismiss
    @ViewState private var corrected = ""
    @ViewState private var spoken = ""
    @ViewState private var written = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(L("correct.title")).font(.title2.bold())
            TextEditor(text: $corrected).font(.body).frame(height: 150).border(.secondary.opacity(0.2))
            Text(L("correct.remember")).font(.subheadline)
            HStack { TextField(L("correct.heard"), text: $spoken); Image(systemName: "arrow.right"); TextField(L("correct.written"), text: $written) }.textFieldStyle(.roundedBorder)
            Text(L("correct.note")).font(.caption).foregroundStyle(.secondary)
            HStack {
                Spacer(); Button(L("action.cancel")) { dismiss() }
                Button(L("correct.save")) {
                    if let index = store.history.firstIndex(where: { $0.id == entry.id }) { store.history[index].text = corrected }
                    store.addWord(spoken: spoken, written: written, learned: true); dismiss()
                }.buttonStyle(.borderedProminent).disabled(corrected.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }.padding(28).frame(width: 520).onAppear { corrected = entry.text }
    }
}

struct DictionaryView: View {
    @ObservedObject var store: AppStore
    @ViewState private var spoken = ""
    @ViewState private var written = ""
    @ViewState private var query = ""
    @ViewState private var importMessage = ""
    @ViewState private var editingID: UUID?
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            pageTitle(L("dict.title"), detail: L("dict.subtitle"))
            Card {
                VStack(alignment: .leading, spacing: 14) {
                    Text(editingID == nil ? L("dict.add") : L("dict.editEntry")).font(.headline)
                    HStack {
                        TextField(L("dict.spoken"), text: $spoken)
                        Image(systemName: "arrow.right").foregroundStyle(.secondary)
                        TextField(L("dict.written"), text: $written)
                        Button(editingID == nil ? L("dict.addWord") : L("dict.saveWord")) {
                            guard store.addWord(spoken: spoken, written: written.isEmpty ? spoken : written, replacing: editingID) else {
                                importMessage = L("dict.saveError")
                                return
                            }
                            spoken = ""; written = ""; editingID = nil; importMessage = ""
                        }.buttonStyle(.borderedProminent)
                            .disabled(!DictionaryEntry.isValidPhrase(spoken) || (!written.isEmpty && !DictionaryEntry.isValidPhrase(written)))
                        if editingID != nil { Button(L("action.cancel")) { spoken = ""; written = ""; editingID = nil } }
                    }.textFieldStyle(.roundedBorder)
                    Text(L("dict.hint")).font(.system(size: 11)).foregroundStyle(.secondary)
                }
            }
            HStack {
                TextField(L("dict.find"), text: $query).textFieldStyle(.roundedBorder)
                Button(L("dict.import")) {
                    do { if let text = try FileActions.importText() { importMessage = L("dict.imported", String(try store.importWords(text))) } }
                    catch { importMessage = error.localizedDescription }
                }
                Button(L("dict.exportAction")) { FileActions.exportDictionary(store.dictionary) }
            }
            if !importMessage.isEmpty { Text(importMessage).font(.caption).foregroundStyle(.secondary) }
            if store.dictionary.isEmpty {
                ContentUnavailableView(L("dict.emptyTitle"), systemImage: "book.closed", description: Text(L("dict.emptyBody")))
                    .frame(maxWidth: .infinity, minHeight: 180)
            }
            LazyVStack(spacing: 8) {
                ForEach(store.dictionary.filter { query.isEmpty || $0.spoken.localizedCaseInsensitiveContains(query) || $0.written.localizedCaseInsensitiveContains(query) }) { entry in
                    HStack(spacing: 12) {
                        Text(entry.spoken).frame(maxWidth: .infinity, alignment: .leading)
                        Image(systemName: "arrow.right").foregroundStyle(.tertiary)
                        Text(entry.written).frame(maxWidth: .infinity, alignment: .leading)
                        if !entry.isValid { Image(systemName: "exclamationmark.circle").foregroundStyle(.orange).help(L("dict.invalidEntry")) }
                        if entry.learned { Image(systemName: "sparkle").foregroundStyle(accent).help(L("dict.fromCorrection")) }
                        Button(L("action.edit")) { editingID = entry.id; spoken = entry.spoken; written = entry.written }
                        Button {
                            store.dictionary.removeAll { $0.id == entry.id }
                            if editingID == entry.id { editingID = nil; spoken = ""; written = "" }
                        } label: { Image(systemName: "trash").foregroundStyle(.secondary) }.buttonStyle(.plain).accessibilityLabel(L("dict.deleteWord"))
                    }.textFieldStyle(.plain).padding(15).background(cardBackground, in: RoundedRectangle(cornerRadius: 10))
                }
            }
        }
    }
}

struct RulesView: View {
    @ObservedObject var store: AppStore
    @ViewState private var bundleID = ""
    @ViewState private var appName = ""
    @ViewState private var selectedStyle = "clean"
    @ViewState private var instructions = ""
    @ViewState private var apps: [NSRunningApplication] = []
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            pageTitle(L("rules.title"), detail: L("rules.subtitle"))
            Card {
                VStack(alignment: .leading, spacing: 16) {
                    Text(L("rules.everywhere")).font(.headline)
                    Picker(L("rules.defaultStyle"), selection: $store.preferences.style) { ForEach(styles, id: \.self) { Text(L("style." + $0)).tag($0) } }
                    TextField(L("rules.instructions"), text: $store.preferences.instructions, axis: .vertical)
                        .textFieldStyle(.roundedBorder).lineLimit(3...5)
                    Text(L("rules.charCount", String(store.preferences.instructions.unicodeScalars.count)))
                        .font(.caption).foregroundStyle(store.preferences.instructions.unicodeScalars.count > 1000 ? .orange : .secondary)
                    if let error = instructionError(store.preferences.instructions, "") { Text(error).font(.caption).foregroundStyle(.orange) }
                    Text(L("rules.styleNote")).font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 14) {
                    Text(L("rules.addApp")).font(.headline)
                    Picker(L("rules.runningApp"), selection: $bundleID) {
                        Text(L("rules.chooseApp")).tag("")
                        ForEach(apps, id: \.processIdentifier) { app in Text(app.localizedName ?? app.bundleIdentifier ?? "App").tag(app.bundleIdentifier ?? "") }
                    }.onChange(of: bundleID) { _, value in appName = apps.first { $0.bundleIdentifier == value }?.localizedName ?? value }
                    HStack { TextField(L("rules.bundleID"), text: $bundleID); TextField(L("rules.displayName"), text: $appName) }.textFieldStyle(.roundedBorder)
                    Picker(L("rules.style"), selection: $selectedStyle) { ForEach(styles, id: \.self) { Text(L("style." + $0)).tag($0) } }
                    TextField(L("rules.appInstructions"), text: $instructions, axis: .vertical).textFieldStyle(.roundedBorder)
                    Text(L("rules.charCountCombined", String(instructions.unicodeScalars.count)))
                        .font(.caption).foregroundStyle(instructions.unicodeScalars.count > 1000 ? .orange : .secondary)
                    if let error = instructionError(store.preferences.instructions, instructions) { Text(error).font(.caption).foregroundStyle(.orange) }
                    Button(L("rules.saveApp")) {
                        store.rules.removeAll { $0.bundleID == bundleID }
                        store.rules.append(AppRule(bundleID: bundleID, name: appName.isEmpty ? bundleID : appName, style: selectedStyle, instructions: instructions))
                        bundleID = ""; appName = ""; instructions = ""
                    }.disabled(bundleID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || instructionError(store.preferences.instructions, instructions) != nil).buttonStyle(.borderedProminent)
                }
            }
            ForEach(store.rules) { rule in
                Card {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(rule.name).font(.headline)
                            Text(L("rules.appSummary", L("style." + rule.style), rule.bundleID)).font(.caption).foregroundStyle(.secondary)
                            if !rule.instructions.isEmpty { Text(rule.instructions).font(.subheadline) }
                        }
                        Spacer()
                        Button(L("action.edit")) { bundleID = rule.bundleID; appName = rule.name; selectedStyle = rule.style; instructions = rule.instructions }
                        Button { store.rules.removeAll { $0.id == rule.id } } label: { Image(systemName: "trash") }.accessibilityLabel(L("rules.deleteApp"))
                    }
                }
            }
        }.onAppear { apps = NSWorkspace.shared.runningApplications.filter { $0.activationPolicy == .regular && $0.bundleIdentifier != nil }.sorted { ($0.localizedName ?? "") < ($1.localizedName ?? "") } }
    }
    private func instructionError(_ defaults: String, _ app: String) -> String? {
        do { _ = try Preferences.combinedInstructions(defaults, app); return nil }
        catch { return error.localizedDescription }
    }
}

@MainActor
private enum FileActions {
    static func importText() throws -> String? {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.commaSeparatedText, .plainText]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size <= 1_000_000 else { throw Failure("error.dictionaryTooLarge") }
        return try String(contentsOf: url, encoding: .utf8)
    }
    static func exportHistory(_ entries: [HistoryEntry]) {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
        save(name: "OmniTyper-history.json", type: .json) { try encoder.encode(entries) }
    }
    static func exportDictionary(_ entries: [DictionaryEntry]) {
        func escape(_ text: String) -> String { "\"" + text.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        let text = "spoken,written\n" + entries.map { "\(escape($0.spoken)),\(escape($0.written))" }.joined(separator: "\n")
        save(name: "OmniTyper-dictionary.csv", type: .commaSeparatedText) { Data(text.utf8) }
    }
    static func exportAudio(_ url: URL) { save(name: "OmniTyper-recording.wav", type: .wav) { try Data(contentsOf: url) } }
    private static func save(name: String, type: UTType, data: () throws -> Data) {
        let panel = NSSavePanel(); panel.nameFieldStringValue = name; panel.allowedContentTypes = [type]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try data().write(to: url, options: .atomic) }
        catch { let alert = NSAlert(error: error); alert.runModal() }
    }
}
