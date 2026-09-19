// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

// Note (Codex): Some Command Line Tools SDKs expose an unavailable State macro.
typealias ViewState<Value> = SwiftUI.State<Value>

private let brandGreen = Color(red: 0.13, green: 0.46, blue: 0.36)
let accent = Color(nsColor: NSColor(name: nil) { appearance in
    appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        ? NSColor(red: 0.40, green: 0.78, blue: 0.64, alpha: 1)
        : NSColor(red: 0.13, green: 0.46, blue: 0.36, alpha: 1)
})
let cardBackground = Color(nsColor: .controlBackgroundColor)
let styles = ["clean", "verbatim", "casual", "formal", "concise"]
let languages = ["English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Italian", "Russian", "Arabic", "Hindi", "Cantonese"]

enum Page: String, CaseIterable {
    case home = "Home", history = "History", dictionary = "Dictionary", rules = "Writing style", settings = "Settings"
    var title: String { L("nav." + rawValue.replacingOccurrences(of: " ", with: "")) }
    var icon: String {
        switch self { case .home: return "square.grid.2x2"; case .history: return "clock.arrow.circlepath"
        case .dictionary: return "book.closed"; case .rules: return "slider.horizontal.3"; case .settings: return "gearshape" }
    }
}

struct RootView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    @ViewState private var page: Page = .home
    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            VStack(spacing: 0) {
                HStack {
                    Text(page.title).font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Label(L("app.badge"), systemImage: "lock.shield")
                        .font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundStyle(accent)
                    Circle().fill(accent).frame(width: 6, height: 6)
                }.padding(.horizontal, 32).frame(height: 60)
                Divider().opacity(0.5)
                ScrollView {
                    VStack(alignment: .leading, spacing: 22) {
                        if !store.storageError.isEmpty { message(store.storageError, error: true) }
                        if !model.error.isEmpty {
                            message(model.error, error: true)
                            if model.canRetry { Button(L("app.retryLast")) { model.retryLast() } }
                        }
                        if !model.notice.isEmpty { message(model.notice, error: false) }
                        if model.phase == .preparing { PreparationCard(model: model, worker: model.worker) }
                        switch page {
                        case .home: HomeView(model: model, store: store)
                        case .history: HistoryView(model: model, store: store)
                        case .dictionary: DictionaryView(store: store)
                        case .rules: RulesView(store: store)
                        case .settings: PreferencesView(model: model, store: store, shortcut: model.shortcut)
                        }
                    }.padding(32).frame(maxWidth: 940, alignment: .leading).frame(maxWidth: .infinity)
                }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .tint(accent)
        .preferredColorScheme(store.preferences.appearance == "light" ? .light : store.preferences.appearance == "dark" ? .dark : nil)
        .onChange(of: model.resultText) { _, _ in page = .home }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 30) {
            HStack(spacing: 10) {
                Image(systemName: "waveform").font(.system(size: 23, weight: .medium))
                    .foregroundStyle(.white).frame(width: 40, height: 40)
                    .background(brandGreen, in: RoundedRectangle(cornerRadius: 13))
                VStack(alignment: .leading, spacing: 2) {
                    Text("OmniTyper").font(.system(size: 16, weight: .semibold))
                    Text(L("app.tagline")).font(.system(size: 10)).foregroundStyle(.secondary)
                }
            }.padding(.top, 35).padding(.horizontal, 18)
            VStack(spacing: 6) {
                ForEach(Page.allCases, id: \.self) { item in
                    Button { page = item } label: {
                        HStack(spacing: 12) {
                            Image(systemName: item.icon).frame(width: 20)
                            Text(item.title).font(.system(size: 13, weight: page == item ? .semibold : .regular))
                            Spacer()
                            if item == .history && !store.history.isEmpty {
                                Text("\(store.history.count)").font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
                            }
                        }.padding(.horizontal, 14).padding(.vertical, 12)
                            .background(page == item ? accent.opacity(0.1) : .clear, in: RoundedRectangle(cornerRadius: 10))
                            .foregroundStyle(page == item ? accent : .primary)
                            .contentShape(Rectangle())
                    }.buttonStyle(.plain)
                }
            }.padding(.horizontal, 12)
            Spacer()
            VStack(alignment: .leading, spacing: 12) {
                Label(L("app.privacyTitle"), systemImage: "desktopcomputer").font(.system(size: 11, weight: .medium))
                Text(L("app.privacyBody"))
                    .font(.system(size: 11)).foregroundStyle(.secondary).lineSpacing(4)
                HStack {
                    Text("SGLang-Omni + MLX").font(.system(size: 9, weight: .medium, design: .monospaced))
                    Spacer()
                    Text("0.1").font(.system(size: 9, design: .monospaced))
                }.foregroundStyle(.tertiary)
            }.padding(18)
        }.frame(width: 218).background(cardBackground.opacity(0.48))
    }

    private func message(_ text: String, error: Bool) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: error ? "exclamationmark.circle" : "checkmark.circle")
                .foregroundStyle(error ? .orange : accent)
            Text(text).font(.system(size: 12)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
            Button { if error { model.error = "" } else { model.notice = "" } } label: { Image(systemName: "xmark") }
                .buttonStyle(.plain).accessibilityLabel(L("app.dismiss"))
        }.padding(14).background((error ? Color.orange : accent).opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }
}

struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content.padding(22).frame(maxWidth: .infinity, alignment: .leading)
            .background(cardBackground, in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(.primary.opacity(0.055), lineWidth: 1))
    }
}

struct HomeView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            VStack(alignment: .leading, spacing: 9) {
                Text(L("home.eyebrow")).font(.system(size: 10, weight: .semibold, design: .monospaced)).tracking(2).foregroundStyle(accent)
                Text(L("home.title")).font(.system(size: 34, weight: .semibold, design: .rounded))
                Text(L("home.subtitle")).font(.system(size: 14)).foregroundStyle(.secondary)
            }.padding(.bottom, 4)
            if !model.microphoneAllowed || !model.accessibilityAllowed {
                Card {
                    VStack(alignment: .leading, spacing: 16) {
                        Label(L("home.permissions.title"), systemImage: "hand.wave").font(.system(size: 16, weight: .semibold))
                        Text(L("home.permissions.body")).font(.system(size: 12)).foregroundStyle(.secondary)
                        permission(L("settings.microphone"), subtitle: L("home.permissions.mic"), ready: model.microphoneAllowed, action: model.requestMicrophone)
                        permission(L("settings.accessibility"), subtitle: L("home.permissions.ax"), ready: model.accessibilityAllowed, action: model.requestAccessibility)
                        if model.accessibilityGrantStale {
                            Text(L("home.permissions.axStale")).font(.system(size: 11)).foregroundStyle(.orange)
                                .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 20) {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 7) {
                            Text(model.mode.title).font(.system(size: 23, weight: .semibold))
                            Text(model.mode.detail).font(.system(size: 13)).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: model.mode.icon).font(.system(size: 30)).foregroundStyle(accent)
                            .frame(width: 64, height: 64).background(accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 18))
                    }
                    Picker(L("home.voiceMode"), selection: $model.mode) {
                        ForEach(VoiceMode.allCases) { Label($0.title, systemImage: $0.icon).tag($0) }
                    }.pickerStyle(.segmented).labelsHidden().disabled(model.isBusy)
                    HStack(spacing: 12) {
                        Button { model.toggle() } label: {
                            Label(model.phase == .recording ? L("home.finish") : model.phase == .processing ? L("home.working") : model.phase == .starting ? L("home.loadingSpeech") : L("home.start"),
                                  systemImage: model.phase == .recording ? "stop.fill" : "mic.fill")
                                .frame(minWidth: 142).padding(.vertical, 6)
                        }.buttonStyle(.borderedProminent).controlSize(.large)
                            .disabled(model.isBusy && model.phase != .recording)
                        if model.isBusy { Button(L("action.cancel"), role: .cancel) { model.cancel() } }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 5) {
                            Text(model.shortcutLabel).font(.system(size: 12, weight: .medium, design: .monospaced))
                                .padding(.horizontal, 10).padding(.vertical, 6).background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 7))
                            Text(store.preferences.holdToTalk ? L("home.holdHint") : L("home.pressHint")).font(.system(size: 10)).foregroundStyle(.secondary)
                        }
                    }
                    if model.mode == .translate {
                        HStack {
                            Text(L("home.writeIn")).foregroundStyle(.secondary)
                            Picker(L("home.translationLanguage"), selection: $store.preferences.targetLanguage) { ForEach(languages, id: \.self) { Text(L("language." + $0)).tag($0) } }
                                .labelsHidden().frame(width: 160)
                        }.font(.system(size: 12))
                    }
                    if model.mode == .ask {
                        Text(L("home.askNote")).font(.system(size: 11)).foregroundStyle(.secondary)
                    }
                }
            }
            if model.phase == .starting || model.phase == .recording || model.phase == .processing {
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        Label(L("home.liveTranscript"), systemImage: "waveform").font(.system(size: 13, weight: .semibold))
                        Text(model.liveStatus).font(.system(size: 11)).foregroundStyle(.secondary)
                        Text(model.liveText.isEmpty ? L("home.livePlaceholder") : model.liveText)
                            .font(.system(size: 15)).lineSpacing(4).textSelection(.enabled)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            if !model.resultText.isEmpty {
                Card {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Label(L("home.resultTitle"), systemImage: "text.alignleft").font(.system(size: 13, weight: .semibold))
                            Spacer()
                            Button { model.copyResult() } label: { Label(L("action.copy"), systemImage: "doc.on.doc") }
                        }
                        Text(model.resultText).font(.system(size: 15)).lineSpacing(5).textSelection(.enabled)
                        if model.rawText != model.resultText && !model.rawText.isEmpty {
                            DisclosureGroup(L("home.originalTranscript")) { Text(model.rawText).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(.enabled).padding(.top, 6) }
                                .font(.system(size: 11))
                        }
                    }
                }
            }
            HStack(spacing: 14) {
                stat(L("home.stat.words"), value: "\(store.history.reduce(0) { $0 + $1.units })", icon: "text.word.spacing")
                stat(L("home.stat.time"), value: L("home.stat.minutes", String(Int(store.history.reduce(0) { $0 + $1.duration } / 60))), icon: "waveform")
                stat(L("home.stat.vocabulary"), value: L("home.stat.wordCount", String(store.dictionary.count)), icon: "book.closed")
            }
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "lightbulb").foregroundStyle(accent)
                Text(L("home.tip"))
                    .font(.system(size: 12)).foregroundStyle(.secondary).lineSpacing(4)
            }.padding(.horizontal, 4)
        }
    }
    private func permission(_ title: String, subtitle: String, ready: Bool, action: @escaping () -> Void) -> some View {
        HStack {
            Image(systemName: ready ? "checkmark.circle.fill" : "circle").foregroundStyle(ready ? accent : .secondary)
            VStack(alignment: .leading, spacing: 3) { Text(title).font(.system(size: 12, weight: .medium)); Text(subtitle).font(.system(size: 11)).foregroundStyle(.secondary) }
            Spacer()
            if ready { Text(L("action.ready")).font(.system(size: 11)).foregroundStyle(accent) }
            else { Button(L("action.allow"), action: action).controlSize(.small) }
        }
    }
    private func stat(_ title: String, value: String, icon: String) -> some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                Image(systemName: icon).foregroundStyle(accent)
                Text(value).font(.system(size: 21, weight: .semibold, design: .rounded))
                Text(title).font(.system(size: 10)).foregroundStyle(.secondary)
            }
        }
    }
}

private struct PreparationCard: View {
    @ObservedObject var model: AppModel
    @ObservedObject var worker: WorkerClient
    var body: some View {
        HStack(spacing: 14) {
            ProgressView().controlSize(.small)
            VStack(alignment: .leading, spacing: 5) {
                Text(L("prepare.title")).font(.system(size: 13, weight: .semibold))
                Text(worker.status.isEmpty ? L("prepare.body") : worker.status).font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer(); Button(L("action.cancel")) { model.cancel() }
        }.padding(20).background(accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
    }
}

struct VoicePanel: View {
    @ObservedObject var model: AppModel
    @ObservedObject var recorder: AudioRecorder
    @ObservedObject var worker: WorkerClient
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 14) {
                if model.phase == .recording {
                    HStack(alignment: .center, spacing: 3) {
                        ForEach(0..<9) { index in
                            Capsule().fill(accent).frame(width: 3, height: 5 + 30 * recorder.level * (index % 2 == 0 ? 1 : 0.55))
                        }
                    }.frame(width: 45, height: 38).animation(.easeOut(duration: 0.1), value: recorder.level)
                } else { ProgressView().controlSize(.small).frame(width: 45) }
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.phase == .recording ? L("panel.listening", model.mode.title) : model.phase == .starting ? L("status.loadingModel") : model.liveStatus)
                        .font(.system(size: 12, weight: .semibold)).lineLimit(1)
                    Text(model.phase == .recording ? String(format: L("panel.elapsed"), Int(recorder.elapsed) / 60, Int(recorder.elapsed) % 60) : worker.status)
                        .font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer(minLength: 0)
                if model.phase == .recording {
                    Button { model.finish() } label: { Image(systemName: "stop.fill").foregroundStyle(accent) }.buttonStyle(.plain).accessibilityLabel(L("home.finish"))
                }
                Button { model.cancel() } label: { Image(systemName: "xmark").foregroundStyle(.secondary) }.buttonStyle(.plain).accessibilityLabel(L("panel.cancelRecording"))
            }
            Divider()
            Text(model.liveText.isEmpty ? (model.phase == .starting ? L("panel.waitListening") : L("panel.placeholder")) : String(model.liveText.suffix(600)))
                .font(.system(size: 14)).lineSpacing(3).lineLimit(3).truncationMode(.head)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Text(model.phase == .recording ? model.liveStatus : L("panel.insertNote"))
                .font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
        }.padding(18).frame(width: 460, height: 190).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22))
            .overlay(RoundedRectangle(cornerRadius: 22).stroke(.white.opacity(0.2), lineWidth: 1))
    }
}

func pageTitle(_ title: String, detail: String) -> some View {
    VStack(alignment: .leading, spacing: 9) { Text(title).font(.system(size: 27, weight: .semibold, design: .rounded)); Text(detail).font(.system(size: 13)).foregroundStyle(.secondary) }
}
