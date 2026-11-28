import SwiftUI
import VoxtKit

/// The finished sessions, newest first.
struct HistoryView: View {
    @ObservedObject var model: AppModel
    @State private var confirmClear = false

    var body: some View {
        List {
            if model.history.isEmpty {
                Text(L("Your transcripts appear here."))
                    .foregroundStyle(.secondary)
            }
            ForEach(model.history) { entry in
                NavigationLink {
                    HistoryDetail(entry: entry)
                } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Label(entry.feature.label, systemImage: entry.feature.symbol)
                            Spacer()
                            Text(entry.date, style: .relative)
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        Text(entry.title.isEmpty ? entry.result : entry.title)
                            .lineLimit(3)
                    }
                }
            }
            .onDelete { offsets in
                model.deleteHistory(Set(offsets.map { model.history[$0].id }))
            }
        }
        .navigationTitle(L("History"))
        .toolbar {
            if !model.history.isEmpty {
                Button(L("Clear"), role: .destructive) { confirmClear = true }
            }
        }
        .confirmationDialog(L("Delete the whole history?"), isPresented: $confirmClear, titleVisibility: .visible) {
            Button(L("Delete all"), role: .destructive) { model.clearHistory() }
        }
        .onAppear { model.reloadHistory() }
    }
}

struct HistoryDetail: View {
    var entry: HistoryEntry

    var body: some View {
        List {
            Section {
                if !entry.title.isEmpty {
                    Text(entry.title).font(.headline)
                }
                Text(entry.result).textSelection(.enabled)
                HStack {
                    Button(L("Copy"), systemImage: "doc.on.doc") { Clipboard.copy(entry.result) }
                    ShareLink(item: entry.result) { Label(L("Share"), systemImage: "square.and.arrow.up") }
                }
                .buttonStyle(.borderless)
            }
            if !entry.warning.isEmpty {
                Section(L("Kept as recognised")) { Text(entry.warning).font(.callout) }
            }
            if !entry.source.isEmpty {
                Section(entry.feature == .rewrite ? L("Text to rewrite") : L("Text to translate")) {
                    Text(entry.source).textSelection(.enabled)
                }
            }
            if !entry.speech.isEmpty {
                Section(entry.feature == .rewrite ? L("Instruction") : L("What was recognised")) {
                    Text(entry.speech).textSelection(.enabled)
                }
            }
            Section {
                LabeledContent(L("Feature"), value: entry.feature.label)
                LabeledContent(L("Date"), value: entry.date.formatted(date: .abbreviated, time: .shortened))
                LabeledContent(L("Duration"), value: L("%.1f s", entry.seconds))
            }
        }
        .navigationTitle(entry.feature.label)
        .inlineTitle()
    }
}
