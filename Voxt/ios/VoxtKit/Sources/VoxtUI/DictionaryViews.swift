import SwiftUI
import VoxtKit

/// The dictionary: notes for the language model and the terms speech
/// recognition should get right.
struct DictionarySettings: View {
    @ObservedObject var model: AppModel
    @State private var term = ""
    @State private var aliases = ""

    var body: some View {
        let dict = $model.settings.dictionary
        Form {
            Section {
                Toggle(L("Use the dictionary"), isOn: dict.enabled)
            }
            Section {
                TextEditor(text: dict.notes).frame(minHeight: 100)
            } header: {
                Text(L("Notes"))
            } footer: {
                Text(L("Who you are, the names and words you use: the language model reads these with every request."))
            }
            Section {
                TextField(L("Term, as it must be written"), text: $term)
                TextField(L("Heard as (comma-separated, optional)"), text: $aliases)
                Button(L("Add term")) {
                    let heard = aliases.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                    model.settings.dictionary.terms.append(Term(term: term.trimmingCharacters(in: .whitespaces), aliases: heard))
                    term = ""
                    aliases = ""
                }
                .disabled(term.trimmingCharacters(in: .whitespaces).isEmpty)
            } header: {
                Text(L("Terms"))
            }
            Section {
                ForEach(model.settings.dictionary.terms) { t in
                    VStack(alignment: .leading) {
                        Text(t.term)
                        if !t.aliases.isEmpty {
                            Text(t.aliases.joined(separator: ", ")).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { model.settings.dictionary.terms.remove(atOffsets: $0) }
            }
        }
        .navigationTitle(L("Dictionary"))
    }
}

/// Snippets: a spoken trigger that Transcribe replaces with its text.
struct SnippetSettings: View {
    @ObservedObject var model: AppModel
    @State private var trigger = ""
    @State private var text = ""

    var body: some View {
        Form {
            Section {
                TextField(L("Trigger, for example \"my address\""), text: $trigger)
                TextField(L("Text"), text: $text, axis: .vertical)
                    .lineLimit(2...6)
                Button(L("Add snippet")) {
                    model.settings.snippets.append(Snippet(trigger: trigger.trimmingCharacters(in: .whitespaces), text: text))
                    trigger = ""
                    text = ""
                }
                .disabled(trigger.trimmingCharacters(in: .whitespaces).isEmpty || text.isEmpty)
            } footer: {
                Text(L("When you say a trigger in Transcribe, Voxt puts in its text as you saved it."))
            }
            Section {
                ForEach(model.settings.snippets) { s in
                    VStack(alignment: .leading) {
                        Text(s.trigger)
                        Text(s.text).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    }
                }
                .onDelete { model.settings.snippets.remove(atOffsets: $0) }
            }
        }
        .navigationTitle(L("Snippets"))
    }
}
