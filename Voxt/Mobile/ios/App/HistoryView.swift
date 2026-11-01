import SwiftUI
import UIKit
import VoxtMobileKit

/// Transcripts made on this phone, newest first, grouped by day. Tap a card to copy it.
struct HistoryView: View {
    @Environment(AppModel.self) private var model
    var onEditLatest: () -> Void = {}
    @State private var copiedID: UUID?
    @State private var query = ""
    @State private var showsSettings = false

    private struct DaySection: Identifiable {
        var id: String { title }
        let title: String
        let items: [HistoryItem]
    }

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 14) {
                        Button { showsSettings = true } label: {
                            HStack {
                                Image(systemName: "externaldrive")
                                Text(VoxtL10n.shared.string("Save history"))
                                Spacer()
                                Text(model.historyRetention.title).foregroundStyle(.secondary)
                                Image(systemName: "chevron.right").font(.caption)
                            }
                        }
                        .buttonStyle(.plain)
                        Divider()
                        Text(VoxtL10n.shared.string("Uses your Mac’s selected models. If a result doesn’t land in the other app, tap it here to copy."))
                            .font(VoxtTheme.Typography.caption)
                            .foregroundStyle(.secondary)
                        if model.historySyncEnabled {
                            Text(VoxtL10n.shared.string("Sync is enabled. Deletions reach your Mac when both devices can sync."))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .padding(16)
                    .background(VoxtTheme.Colors.panel, in: RoundedRectangle(cornerRadius: 18))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                }
                .listSectionSeparator(.hidden)

                if !model.unsentRecordings.isEmpty {
                    Section {
                        ForEach(model.unsentRecordings) { recording in
                            UnsentRecordingRow(recording: recording)
                                .voxtCard()
                                .listRowBackground(Color.clear)
                                .listRowSeparator(.hidden)
                        }
                        .onDelete { offsets in
                            for offset in offsets { model.deleteUnsentRecording(model.unsentRecordings[offset].id) }
                        }
                    } header: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(VoxtL10n.shared.string("Not sent yet"))
                                .font(.system(size: 20, weight: .bold))
                                .foregroundStyle(VoxtTheme.Colors.ink)
                            Text(VoxtL10n.shared.string("Your Mac hasn't turned these into text. They stay on this iPhone until they're sent or you delete them."))
                                .font(VoxtTheme.Typography.caption)
                                .foregroundStyle(.secondary)
                        }
                        .textCase(nil)
                    }
                    .listSectionSeparator(.hidden)
                }

                if daySections.isEmpty {
                    Section {
                        Text(VoxtL10n.shared.string(query.isEmpty ? (model.historyRetention == .off ? "History is off. Your latest result is available on Home until you close the app." : "Nothing here yet. What you dictate shows up here.") : "No matching dictations."))
                            .font(VoxtTheme.Typography.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity)
                            .multilineTextAlignment(.center)
                    }
                    .listRowBackground(Color.clear)
                } else {
                    ForEach(daySections) { section in
                        Section {
                            ForEach(section.items) { item in
                                HStack(alignment: .top) {
                                Button {
                                    UIPasteboard.general.string = item.text
                                    copiedID = item.id
                                } label: {
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text(copiedID == item.id ? VoxtL10n.shared.string("Copied") : item.date.formatted(.dateTime.hour().minute().locale(VoxtL10n.shared.locale)))
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(item.text)
                                            .font(VoxtTheme.Typography.body)
                                            .foregroundStyle(.primary)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                            .accessibilityIdentifier("historyText")
                                    }
                                }
                                .buttonStyle(.plain)
                                if item.id == model.history.first?.id {
                                    Button(action: onEditLatest) { Image(systemName: "pencil").frame(width: 44, height: 44) }
                                        .buttonStyle(.plain)
                                        .accessibilityLabel(VoxtL10n.shared.string("Edit latest result"))
                                        .accessibilityIdentifier("editLatestResult")
                                        .disabled(model.phase == .recording || model.phase == .processing)
                                }
                                }
                                .voxtCard()
                                .listRowBackground(Color.clear)
                                .listRowSeparator(.hidden)
                            }
                            .onDelete { offsets in
                                delete(offsets: offsets, in: section)
                            }
                        } header: {
                            Text(section.title)
                                .font(.system(size: 20, weight: .bold))
                                .foregroundStyle(VoxtTheme.Colors.ink)
                                .textCase(nil)
                        }
                        .listSectionSeparator(.hidden)
                    }
                }
            }
            .listStyle(.plain)
            .listRowSeparator(.hidden)
            .listRowBackground(Color.clear)
            .listRowInsets(EdgeInsets(top: 0, leading: 16, bottom: 12, trailing: 16))
            .scrollContentBackground(.hidden)
            .background(VoxtTheme.Colors.screenBackground)
            .navigationTitle(VoxtL10n.shared.string("History"))
            .searchable(text: $query, prompt: VoxtL10n.shared.string("Search history"))
            .sheet(isPresented: $showsSettings) { SettingsView() }
        }
    }

    private var daySections: [DaySection] {
        let matching = model.history.filter { query.isEmpty || $0.text.localizedStandardContains(query) }
        let grouped = Dictionary(grouping: matching) { Calendar.current.startOfDay(for: $0.date) }
        return grouped.keys.sorted(by: >).map { day in
            let title: String
            if Calendar.current.isDateInToday(day) {
                title = VoxtL10n.shared.string("Today")
            } else if Calendar.current.isDateInYesterday(day) {
                title = VoxtL10n.shared.string("Yesterday")
            } else {
                title = day.formatted(.dateTime.year().month().day().locale(VoxtL10n.shared.locale))
            }
            return DaySection(title: title, items: grouped[day]?.sorted { $0.date > $1.date } ?? [])
        }
    }

    private func delete(offsets: IndexSet, in section: DaySection) {
        let global = IndexSet(offsets.compactMap { offset in
            model.history.firstIndex(where: { $0.id == section.items[offset].id })
        })
        model.deleteHistory(at: global)
    }
}

/// A recording the Mac never answered: when, how long, why, and a button to send it again.
private struct UnsentRecordingRow: View {
    @Environment(AppModel.self) private var model
    let recording: SavedRecording

    private var reason: String {
        switch recording.status {
        case .cancelled: VoxtL10n.shared.string("Cancelled")
        case .interrupted: VoxtL10n.shared.string("Interrupted")
        default: VoxtL10n.shared.string("Couldn't send")
        }
    }

    private var when: String {
        recording.startedAt.formatted(.dateTime.month().day().hour().minute().locale(VoxtL10n.shared.locale))
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: "waveform")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 4) {
                Text(Duration.seconds(recording.duration.rounded()).formatted(.time(pattern: .minuteSecond)))
                    .font(VoxtTheme.Typography.body)
                    .monospacedDigit()
                Text([when, reason].joined(separator: " · "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button(VoxtL10n.shared.string("Send again")) {
                Task { await model.sendAgain(recording.id) }
            }
            .buttonStyle(.bordered)
            .disabled(model.phase == .recording || model.phase == .processing)
            .accessibilityIdentifier("sendRecordingAgain")
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("unsentRecording")
    }
}
