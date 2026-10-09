// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

struct NotesView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var taker: NoteTaker
    @ObservedObject var notes: NoteStore
    @ObservedObject var recorder: AudioRecorder
    @ViewState private var expanded: UUID?
    @ViewState private var deleting: Note?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            pageTitle(L("notes.title"), detail: L("notes.subtitle"))
            if !notes.storageError.isEmpty { Text(notes.storageError).font(.system(size: 12)).foregroundStyle(.orange) }
            recorderCard
            if notes.notes.isEmpty {
                ContentUnavailableView(L("notes.emptyTitle"), systemImage: "note.text", description: Text(L("notes.emptyBody")))
                    .frame(maxWidth: .infinity, minHeight: 200)
            }
            LazyVStack(spacing: 14) {
                ForEach(notes.notes) { note in noteCard(note) }
            }
        }
        .confirmationDialog(L("notes.deleteConfirm"), isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            Button(L("action.delete"), role: .destructive) { if let deleting { notes.delete(deleting.id) } }
        }
    }

    private var active: Note? { taker.activeID.flatMap { notes.note($0) } }

    private var recorderCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .center, spacing: 14) {
                    Image(systemName: taker.phase == .recording ? "record.circle" : "note.text")
                        .font(.system(size: 26)).foregroundStyle(taker.phase == .recording ? .red : accent)
                        .frame(width: 54, height: 54).background(accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 16))
                    VStack(alignment: .leading, spacing: 5) {
                        Text(taker.isActive ? (active?.title ?? L("notes.idleTitle")) : L("notes.idleTitle"))
                            .font(.system(size: 17, weight: .semibold)).lineLimit(1)
                        Text(statusText).font(.system(size: 12)).foregroundStyle(.secondary).lineLimit(2)
                    }
                    Spacer()
                    if taker.phase == .recording {
                        Text(clock(recorder.elapsed)).font(.system(size: 15, weight: .medium, design: .monospaced))
                    }
                    actionButton
                }
                if taker.phase == .recording {
                    ProgressView(value: min(1, recorder.level)).tint(accent)
                }
                if taker.phase == .recording || taker.phase == .finishing, let active {
                    ScrollViewReader { proxy in
                        ScrollView {
                            Text(active.transcript.isEmpty ? L("notes.liveEmpty") : active.transcript)
                                .font(.system(size: 13)).lineSpacing(4).textSelection(.enabled)
                                .foregroundStyle(active.transcript.isEmpty ? .secondary : .primary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Color.clear.frame(height: 1).id("end")
                        }
                        .frame(height: 150).padding(12)
                        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
                        .onChange(of: active.segments.count) { _, _ in proxy.scrollTo("end", anchor: .bottom) }
                    }
                }
                if !taker.error.isEmpty { Text(taker.error).font(.system(size: 11)).foregroundStyle(.orange).textSelection(.enabled) }
            }
        }
    }

    private var statusText: String {
        switch taker.phase {
        case .idle: return L("notes.idleBody")
        case .starting: return model.worker.status
        case .recording: return taker.status
        case .finishing: return model.worker.isRunning && !model.worker.status.isEmpty ? model.worker.status : taker.status
        }
    }

    @ViewBuilder private var actionButton: some View {
        switch taker.phase {
        case .idle:
            Button { taker.start() } label: { Label(L("notes.start"), systemImage: "mic.fill") }
                .buttonStyle(.borderedProminent).disabled(model.phase != .idle || !model.microphoneAllowed)
        case .starting, .finishing:
            ProgressView().controlSize(.small)
        case .recording:
            Button { taker.stop() } label: { Label(L("notes.stop"), systemImage: "stop.fill") }
                .buttonStyle(.borderedProminent).tint(.red)
        }
    }

    private func noteCard(_ note: Note) -> some View {
        let isOpen = expanded == note.id
        return Card {
            VStack(alignment: .leading, spacing: 12) {
                Button { expanded = isOpen ? nil : note.id } label: {
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(note.title).font(.system(size: 15, weight: .semibold)).lineLimit(1)
                            Text("\(note.date.formatted(date: .abbreviated, time: .shortened)) · \(clock(note.duration))")
                                .font(.system(size: 11)).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(L("notes.state.\(note.status.rawValue)")).font(.system(size: 10, weight: .semibold))
                            .padding(.horizontal, 8).padding(.vertical, 3)
                            .background((note.status == .failed || note.status == .interrupted ? Color.orange : accent).opacity(0.12), in: Capsule())
                        Image(systemName: isOpen ? "chevron.up" : "chevron.down").foregroundStyle(.secondary)
                    }.contentShape(Rectangle())
                }.buttonStyle(.plain)
                if let warning = note.warning, !warning.isEmpty {
                    Text(warning).font(.system(size: 11)).foregroundStyle(.orange).textSelection(.enabled)
                }
                if isOpen { detail(note) }
            }
        }
    }

    @ViewBuilder private func detail(_ note: Note) -> some View {
        TextField(L("notes.titleField"), text: Binding(get: { note.title }, set: { value in
            notes.update(note.id) { $0.title = String(value.prefix(120)) }
        })).textFieldStyle(.roundedBorder)
        if !note.notes.isEmpty { NoteMarkdownView(markdown: note.notes) }
        if !note.transcript.isEmpty {
            DisclosureGroup(L("notes.transcript")) {
                Text(note.transcript).font(.system(size: 12)).lineSpacing(4).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 6)
            }.font(.system(size: 12)).foregroundStyle(.secondary)
        }
        HStack {
            if !note.notes.isEmpty {
                Button { TextInsertion.copy(note.notes) } label: { Label(L("notes.copyNotes"), systemImage: "doc.on.doc") }
            }
            if !note.transcript.isEmpty {
                Button { TextInsertion.copy(note.transcript) } label: { Label(L("notes.copyTranscript"), systemImage: "text.quote") }
            }
            Spacer()
            if !note.transcript.isEmpty && !note.isUnfinished {
                Button(note.notes.isEmpty ? L("notes.write") : L("notes.rewrite")) { taker.rewrite(note.id) }
                    .disabled(model.isBusy)
            }
            Menu {
                Button(L("notes.export")) { FileActions.exportMarkdown(note.markdown, name: fileName(note)) }
                Button(L("action.delete"), role: .destructive) { deleting = note }.disabled(taker.activeID == note.id)
            } label: { Image(systemName: "ellipsis") }.menuStyle(.borderlessButton).frame(width: 20)
        }.controlSize(.small)
    }

    private func fileName(_ note: Note) -> String {
        let safe = note.title.components(separatedBy: CharacterSet(charactersIn: "/:\\?%*|\"<>")).joined(separator: "-")
        return (safe.isEmpty ? "Notes" : safe) + ".md"
    }

    private func clock(_ seconds: Double) -> String {
        let total = Int(seconds)
        return total >= 3600 ? String(format: "%d:%02d:%02d", total / 3600, total / 60 % 60, total % 60)
            : String(format: "%d:%02d", total / 60, total % 60)
    }
}

struct NoteMarkdownView: View {
    let markdown: String

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ForEach(Array(markdown.components(separatedBy: "\n").enumerated()), id: \.offset) { _, line in
                row(line.trimmingCharacters(in: .whitespaces))
            }
        }.textSelection(.enabled)
    }

    @ViewBuilder private func row(_ line: String) -> some View {
        if line.hasPrefix("# ") {
            EmptyView()
        } else if line.hasPrefix("## ") || line.hasPrefix("### ") {
            inline(String(line.drop { $0 == "#" }.dropFirst())).font(.system(size: 13, weight: .semibold))
                .foregroundStyle(accent).padding(.top, 6)
        } else if let task = ["- [ ] ", "- [x] ", "- [X] "].first(where: line.hasPrefix) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: task == "- [ ] " ? "square" : "checkmark.square").font(.system(size: 11)).foregroundStyle(.secondary)
                inline(String(line.dropFirst(task.count)))
            }
        } else if line.hasPrefix("- ") || line.hasPrefix("* ") {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("•").foregroundStyle(.secondary)
                inline(String(line.dropFirst(2)))
            }
        } else if !line.isEmpty {
            inline(line)
        }
    }

    private func inline(_ text: String) -> Text {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        let attributed = (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
        return Text(attributed).font(.system(size: 13))
    }
}
