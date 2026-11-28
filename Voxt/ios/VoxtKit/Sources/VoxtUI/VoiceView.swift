import SwiftUI
import VoxtKit

/// The voice screen: pick a feature, hold nothing, tap to record and tap
/// again to stop; the result comes below, to copy or share.
struct VoiceView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var session: VoiceSession
    /// The source editor's keyboard: it goes when a session starts, so
    /// that the result below is not hidden behind it.
    @FocusState private var editing: Bool

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                Picker(L("Feature"), selection: $session.feature) {
                    ForEach(Feature.allCases) { Label($0.label, systemImage: $0.symbol).tag($0) }
                }
                .pickerStyle(.segmented)
                .accessibilityIdentifier("voice.feature")
                .disabled(session.isBusy)
                .onChange(of: session.feature) { _, feature in
                    model.settings.lastFeature = feature
                    session.clearConversation()
                }
                hint
                if session.feature != .transcribe {
                    sourceEditor
                }
                RecordButton(state: session.state, level: session.level) {
                    editing = false
                    Task {
                        if session.state == .recording {
                            session.stop()
                        } else if !session.isBusy {
                            await session.start()
                        }
                    }
                }
                status
                ResultCard(session: session)
            }
            .padding()
        }
        .scrollDismissesKeyboard(.interactively)
        .navigationTitle("Voxt")
        .inlineTitle()
        .toolbar {
            if session.isBusy {
                Button(L("Cancel"), role: .cancel) { session.cancel() }
            }
        }
    }

    private var hint: some View {
        Group {
            switch session.feature {
            case .transcribe:
                Text(L("Speak, and Voxt writes it down, cleaned up by your language model."))
            case .translate:
                Text(L("Speak, or type below, and Voxt translates it into %@.", model.languageName(model.settings.translate.targetLanguage)))
            case .rewrite:
                Text(session.turns.isEmpty
                     ? L("Paste a text and say what to do with it, or just say what to write.")
                     : L("Say what to change in the answer."))
            }
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var sourceEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            if session.feature == .translate || session.turns.isEmpty {
                TextEditor(text: $session.source)
                    .frame(minHeight: 90, maxHeight: 160)
                    .padding(6)
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(.quaternary))
                    .overlay(alignment: .topLeading) {
                        if session.source.isEmpty {
                            Text(session.feature == .translate ? L("Text to translate (optional)") : L("Text to rewrite (optional)"))
                                .foregroundStyle(.tertiary)
                                .padding(12)
                                .allowsHitTesting(false)
                        }
                    }
                    .disabled(session.isBusy)
                    .focused($editing)
                    .accessibilityIdentifier("voice.source")
            }
            HStack {
                if session.feature == .translate || session.turns.isEmpty {
                    Button(L("Paste"), systemImage: "doc.on.clipboard") {
                        session.source = Clipboard.text ?? session.source
                    }
                }
                if session.feature == .translate {
                    Button(L("Translate text"), systemImage: "globe") {
                        editing = false
                        session.translateTyped()
                    }
                        .disabled(session.isBusy || session.source.isEmpty)
                        .accessibilityIdentifier("voice.translate")
                }
                Spacer()
                if session.feature == .rewrite && !session.turns.isEmpty {
                    Button(L("New conversation"), systemImage: "arrow.counterclockwise") { session.clearConversation() }
                        .disabled(session.isBusy)
                }
            }
            .buttonStyle(.bordered)
            .font(.callout)
        }
    }

    private var status: some View {
        Group {
            switch session.state {
            case .idle, .done: EmptyView()
            case .recording: Text(L("Listening… tap to stop"))
            case .recognizing: Label(L("Recognising speech…"), systemImage: "waveform")
            case .composing: Label(L("Asking the language model…"), systemImage: "sparkles")
            case .failed(let problem):
                Label(problem.message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
        }
        .font(.callout)
        .multilineTextAlignment(.center)
        .accessibilityIdentifier("voice.status")
    }
}

/// The round record button, with the input's level around it.
struct RecordButton: View {
    var state: SessionState
    var level: Float
    var action: () -> Void

    var body: some View {
        let recording = state == .recording
        let busy = state == .recognizing || state == .composing
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(recording ? Color.red.opacity(0.18) : Color.accentColor.opacity(0.12))
                    .scaleEffect(recording ? 1 + CGFloat(level) * 0.35 : 1)
                    .animation(.easeOut(duration: 0.1), value: level)
                Circle()
                    .fill(recording ? Color.red : Color.accentColor)
                    .padding(18)
                if busy {
                    ProgressView().tint(.white)
                } else {
                    Image(systemName: recording ? "stop.fill" : "mic.fill")
                        .font(.system(size: 34, weight: .semibold))
                        .foregroundStyle(.white)
                }
            }
            .frame(width: 128, height: 128)
        }
        .buttonStyle(.plain)
        .disabled(busy)
        .accessibilityLabel(recording ? L("Stop recording") : L("Start recording"))
        .accessibilityIdentifier("voice.record")
    }
}

/// The session's result: the title of a written answer, the text, and
/// what was recognised.
struct ResultCard: View {
    @ObservedObject var session: VoiceSession

    var body: some View {
        if session.state == .done, !session.result.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                if !session.title.isEmpty {
                    Text(session.title).font(.headline)
                }
                Text(session.result)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("voice.result")
                if !session.warning.isEmpty {
                    Label(L("Kept as recognised: %@", session.warning), systemImage: "info.circle")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }
                HStack {
                    Button(L("Copy"), systemImage: "doc.on.doc") { Clipboard.copy(session.result) }
                    ShareLink(item: session.result) { Label(L("Share"), systemImage: "square.and.arrow.up") }
                }
                .buttonStyle(.bordered)
                if !session.speech.isEmpty && session.speech != session.result {
                    DisclosureGroup(L("What was recognised")) {
                        Text(session.speech)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.callout)
                }
            }
            .padding()
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14))
        }
    }
}
