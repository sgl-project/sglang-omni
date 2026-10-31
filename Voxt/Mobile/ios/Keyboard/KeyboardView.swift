import SwiftUI
import UIKit
import VoxtMobileKit

struct KeyboardView: View {
    @AppStorage("appearance", store: UserDefaults(suiteName: DictationChannel.appGroup)) private var appearance = "system"
    let model: KeyboardModel
    let showsNextKeyboard: Bool
    let inputController: UIInputViewController

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        if model.styleCandidate != nil {
            VStack(spacing: 0) {
                learningNotice
                Spacer(minLength: 0)
            }
            .background(VoxtTheme.Colors.panel)
            .preferredColorScheme(appearance == "dark" ? .dark : appearance == "light" ? .light : nil)
        } else if model.showsTextKeyboard {
            VStack(spacing: 0) {
                learningNotice
                if model.activeLayoutSurface == "japaneseFlick" && model.textKeyboard.plane == .letters {
                    JapaneseKeyboardView(model: model)
                } else if model.activeLayoutSurface == "zhuyin" && model.textKeyboard.plane == .letters {
                    ZhuyinKeyboardView(model: model)
                } else {
                    TextKeyboardView(
                    state: model.textKeyboard,
                    letterRows: model.activeKeyRows,
                    candidates: model.activeCandidates,
                    reservesCandidateBar: model.hasCandidateBar,
                    composing: model.isComposingCandidates,
                    spaceCaption: model.spaceCaption,
                    onCandidate: model.selectCandidate,
                    suggestions: model.suggestions,
                    onPress: model.pressTextKey,
                    onAlternate: model.insertAlternate,
                    onSuggestion: model.insertSuggestion,
                    onSelectLayout: model.selectTextLayout,
                    onCycleLayout: model.cycleTextLayout
                    )
                }
                HStack {
                    if showsNextKeyboard {
                        NextKeyboardKey(inputController: inputController).frame(width: 40, height: 36)
                    }
                    Spacer()
                }.padding(.horizontal, 12)
                Spacer(minLength: 0)
            }
            .background(VoxtTheme.Colors.panel)
            .preferredColorScheme(appearance == "dark" ? .dark : appearance == "light" ? .light : nil)
        } else {
            voiceKeyboard
        }
    }

    private var voiceKeyboard: some View {
        VStack(spacing: 0) {
            learningNotice
            // While recording, the top row is only the X that cancels.
            if isBusy {
                HStack {
                    Spacer()
                    Button(action: model.cancelTapped) {
                        Image(systemName: "xmark").font(.system(size: 15, weight: .semibold))
                            .frame(width: 36, height: 36)
                            .background(VoxtTheme.Colors.keyGray, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(VoxtTheme.Colors.ink)
                    .accessibilityLabel(VoxtL10n.shared.string("Cancel"))
                    .accessibilityIdentifier("keyboardCancelComposition")
                    .keyFeedback(.action)
                }
                .padding(.horizontal, 16)
                .frame(height: 38)
            } else {
            HStack {
                // ↶ takes the title's place while the latest insert can be undone.
                if model.canUndo {
                    Button(action: model.undoTapped) {
                        Image(systemName: "arrow.uturn.backward").font(.system(size: 16, weight: .semibold))
                            .frame(width: 36, height: 36)
                            .background(VoxtTheme.Colors.keyGray, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(VoxtL10n.shared.string("Undo"))
                    .accessibilityIdentifier("keyboardUndo")
                    .keyFeedback(.action)
                } else {
                    // Opens Voxt on Settings.
                    Button(action: model.openVoxtSettings) {
                        HStack(spacing: 4) {
                            Text("Voxt").font(.system(size: 17, weight: .bold))
                            Image(systemName: "chevron.right").font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(VoxtTheme.Colors.caption)
                        }
                        .frame(minHeight: 36)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(VoxtL10n.shared.string("Open Voxt settings"))
                    .accessibilityIdentifier("keyboardOpenVoxt")
                    .keyFeedback(.action)
                }
                Spacer()
                HStack(spacing: 10) {
                    HStack(spacing: 0) {
                        ForEach(model.textKeyboard.enabled, id: \.self) { id in
                            Button { model.showTyping(); model.selectTextLayout(id) } label: {
                                Text(id.shortLabel).font(.system(size: 13))
                                    .lineLimit(1).minimumScaleFactor(0.65)
                                    .frame(maxWidth: .infinity, minHeight: 32)
                            }
                            .accessibilityLabel(id == .englishQWERTY ? VoxtL10n.shared.string("Show text keyboard") : id.title)
                            .accessibilityIdentifier("voiceLayout-\(id.rawValue)")
                            .keyFeedback(.mode)
                        }
                    }
                    .frame(maxWidth: CGFloat(model.textKeyboard.enabled.count) * 44)
                    .padding(3)
                    .background(VoxtTheme.Colors.card, in: Capsule())
                    .accessibilityElement(children: .contain)
                    .accessibilityIdentifier("voiceLayoutToolbar")
                }
            }
            .foregroundStyle(VoxtTheme.Colors.ink)
            .padding(.horizontal, 16)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 30).onEnded { value in
                if abs(value.translation.width) > abs(value.translation.height) { model.showTyping() }
            })
            }

            Spacer(minLength: 10)
            HStack(spacing: 10) {
                Text(statusText.isEmpty ? caption : statusText)
                    .font(.system(size: 13))
                    .foregroundStyle(model.phase == .failed ? .red : VoxtTheme.Colors.caption)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                // Copy and Retry sit in the line that explains why they are needed.
                if !isBusy, model.recoveryText != nil {
                    lineButton("Copy", id: "keyboardCopyResult", action: model.copyRecoveredText)
                } else if !isBusy, model.canRetry {
                    lineButton("Retry", id: "keyboardRetryComposition", action: model.retryTapped)
                }
            }
            .padding(.horizontal, 16)
            .frame(height: 32)
            .padding(.bottom, 10)

            ZStack(alignment: .trailing) {
                DictationMicButton(
                    phase: model.phase,
                    sessionActive: model.sessionAlive,
                    level: model.audioLevel,
                    size: .keyboard,
                    startLabel: VoxtL10n.shared.string("Start dictation"),
                    stopLabel: VoxtL10n.shared.string("Stop dictation"),
                    action: { model.micTapped() }
                )
                // Touches go to the hold-and-slide area; VoiceOver still activates the button.
                .allowsHitTesting(false)
                .overlay { MicTouchArea(model: model) }
                .overlay { TranslationPills(pills: model.translationPills, entered: model.enteredPill) }
                .frame(maxWidth: .infinity)
                if !isBusy {
                    CircleKey(systemImage: "delete.left", action: model.backspace)
                        .keyFeedback(.key(.delete))
                        .padding(.trailing, 16)
                        .offset(y: 27)
                }
            }
            .frame(height: 92)

            HStack {
              if isBusy {
                // The pen, Return and @ come back when recording ends.
                Color.clear.frame(height: 42)
              } else {
                Button(action: model.penTapped) {
                    Image(systemName: "pencil.and.outline")
                        .font(.system(size: 20))
                        .foregroundStyle(model.penIntent == nil ? VoxtTheme.Colors.ink : VoxtTheme.Colors.onInk)
                        .frame(width: 42, height: 42)
                        .background(model.penIntent == nil ? VoxtTheme.Colors.card : VoxtTheme.Colors.ink, in: Circle())
                }
                .accessibilityLabel(VoxtL10n.shared.string("Edit or write"))
                .accessibilityValue(model.penIntent.map { VoxtL10n.shared.string($0.line) } ?? "")
                .accessibilityAddTraits(model.penIntent == nil ? [] : .isSelected)
                .accessibilityIdentifier("keyboardPen")
                .keyFeedback(.action)
                .disabled(model.phase == .recording || model.phase == .processing)
                Spacer()
                Button(VoxtL10n.shared.string("Return"), action: model.returnKey)
                    .font(.system(size: 16))
                    .frame(width: 120, height: 40)
                    .background(VoxtTheme.Colors.card, in: Capsule())
                    .accessibilityLabel(VoxtL10n.shared.string("Return"))
                    .keyFeedback(.key(.return))
                Spacer()
                CircleKey(systemImage: "at") { model.insertAlternate("@") }
                    .keyFeedback(.key(.character))
              }
            }
            .buttonStyle(.plain)
            .foregroundStyle(VoxtTheme.Colors.ink)
            .padding(.horizontal, 16)
            .padding(.top, 12)

            HStack {
                if showsNextKeyboard {
                    NextKeyboardKey(inputController: inputController).frame(width: 40, height: 40)
                }
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.top, 12)
        }
        .dictationFeedback(phase: model.phase, includeSuccess: false)
        .sensoryFeedback(.success, trigger: model.insertionFeedbackCount)
        .padding(.top, 12)
        .padding(.bottom, 6)
        .background(VoxtTheme.Colors.panel)
        .preferredColorScheme(appearance == "dark" ? .dark : appearance == "light" ? .light : nil)
    }

    @ViewBuilder private var learningNotice: some View {
        if let example = model.styleCandidate {
            VStack(alignment: .leading, spacing: 6) {
                Text(VoxtL10n.shared.string("Keep this wording?")).font(.system(size: 14, weight: .semibold))
                Text(VoxtL10n.shared.string("Keep only if the meaning is unchanged."))
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(VoxtL10n.shared.string("Before")).foregroundStyle(.secondary)
                        Text(example.before).accessibilityIdentifier("styleReviewBefore")
                        Text(VoxtL10n.shared.string("After")).foregroundStyle(.secondary)
                        Text(example.after).accessibilityIdentifier("styleReviewAfter")
                    }
                    .font(.system(size: 13))
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                HStack {
                    Button(VoxtL10n.shared.string("Discard"), action: model.discardPersonalStyle)
                        .accessibilityIdentifier("discardPersonalStyle")
                    Spacer()
                    Button(VoxtL10n.shared.string("Keep"), action: model.keepPersonalStyle)
                        .buttonStyle(.borderedProminent)
                        .tint(VoxtTheme.Colors.ink)
                        .foregroundStyle(VoxtTheme.Colors.onInk)
                        .accessibilityIdentifier("keepPersonalStyle")
                }
            }
            .padding(10)
            .frame(height: 230)
            .background(VoxtTheme.Colors.card)
        } else if let notice = model.styleNotice {
            Text(VoxtL10n.shared.string(notice))
                .font(.system(size: 12))
                .foregroundStyle(VoxtTheme.Colors.ink)
                .lineLimit(2)
                .padding(.horizontal, 14)
                .frame(height: 40)
                .accessibilityIdentifier("personalStyleNotice")
        } else if let record = model.learnedRecord {
            HStack {
                Text(VoxtL10n.shared.format("Learned “%@”", record.correction.replacement))
                    .accessibilityIdentifier("learnedWordNotice")
                    .lineLimit(1)
                Spacer()
                Button(VoxtL10n.shared.string("Undo"), action: model.undoLearnedWord)
                    .accessibilityLabel(VoxtL10n.shared.string("Undo learned word"))
            }
            .font(.system(size: 13))
            .foregroundStyle(VoxtTheme.Colors.ink)
            .padding(.horizontal, 14)
            .frame(height: 40)
        } else if model.learningError != nil {
            Text(VoxtL10n.shared.string("Couldn’t learn this correction. You can add it in Personal dictionary."))
                .font(.system(size: 12))
                .foregroundStyle(VoxtTheme.Colors.caption)
                .lineLimit(2)
                .padding(.horizontal, 14)
                .frame(height: 40)
                .accessibilityIdentifier("learningError")
        }
    }

    private var isBusy: Bool { model.phase == .recording || model.phase == .processing }

    private func lineButton(_ title: String, id: String, action: @escaping () -> Void) -> some View {
        Button(VoxtL10n.shared.string(title), action: action)
            .font(.system(size: 13, weight: .semibold))
            .padding(.horizontal, 12).frame(height: 28)
            .background(VoxtTheme.Colors.card, in: Capsule())
            .buttonStyle(.plain)
            .accessibilityIdentifier(id)
            .keyFeedback(.action)
    }

    /// Errors and one-off messages; per-phase guidance lives in the caption above the mic.
    private var statusText: String {
        if let message = model.message { return VoxtL10n.shared.string(message) }
        if model.phase == .failed { return VoxtL10n.shared.string("Something went wrong.") }
        return ""
    }

    private var caption: String {
        switch model.phase {
        case .recording:
            if let language = model.translatingTo {
                return VoxtL10n.shared.format("Translating to %@", TranslationLanguages.nativeName(for: language))
            }
            switch model.editScope {
            case .lastResult: return VoxtL10n.shared.string("Editing your last result. Tap again to finish.")
            case .selection: return VoxtL10n.shared.string("Editing selection. Tap again to finish.")
            case nil: return VoxtL10n.shared.string("Tap again to finish")
            }
        case .processing: return VoxtL10n.shared.string("Writing it up…")
        case .idle, .done, .failed:
            return VoxtL10n.shared.string(model.penIntent?.line ?? "Tap to speak")
        }
    }
}

/// Tap = dictate (or stop); hold ~0.3 s and slide onto a pill = translate.
private struct MicTouchArea: View {
    let model: KeyboardModel
    @State private var touching = false

    var body: some View {
        Color.clear
            .contentShape(Circle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { value in
                    if !touching { touching = true; model.micDown() }
                    model.micMoved(.init(dx: value.translation.width, dy: value.translation.height))
                }
                .onEnded { value in
                    touching = false
                    model.micUp(.init(dx: value.translation.width, dy: value.translation.height))
                })
            .accessibilityHidden(true)
    }
}

/// Up to three language pills fanned out above the mic while it is held.
private struct TranslationPills: View {
    let pills: [MicHoldGesture.Pill]
    let entered: Int?

    var body: some View {
        let centers = MicHoldGesture.pillCenters(count: pills.count)
        ZStack {
            ForEach(Array(pills.enumerated()), id: \.offset) { index, pill in
                let label: String = {
                    switch pill {
                    case let .language(code): TranslationLanguages.nativeName(for: code)
                    case .chooseLanguages: VoxtL10n.shared.string("Choose translation languages")
                    }
                }()
                let id: String = {
                    switch pill {
                    case let .language(code): "translationPill-\(code)"
                    case .chooseLanguages: "translationPill-choose"
                    }
                }()
                Text(label)
                    .font(.system(size: 15, weight: .semibold))
                    .lineLimit(1)
                    .fixedSize()
                    .padding(.horizontal, 14).frame(height: 40)
                    .foregroundStyle(index == entered ? VoxtTheme.Colors.onInk : VoxtTheme.Colors.ink)
                    .background(index == entered ? VoxtTheme.Colors.ink : VoxtTheme.Colors.card, in: Capsule())
                    .shadow(color: .black.opacity(0.15), radius: 4, y: 1)
                    .offset(x: centers[index].dx, y: centers[index].dy)
                    .accessibilityIdentifier(id)
                    .accessibilityAddTraits(index == entered ? .isSelected : [])
            }
        }
        .allowsHitTesting(false)
    }
}

/// Circular light-gray side key (backspace).
private struct CircleKey: View {
    let systemImage: String
    let action: () -> Void

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 18))
                .foregroundStyle(VoxtTheme.Colors.ink)
                .frame(width: 44, height: 44)
                .background(VoxtTheme.Colors.keyGray, in: Circle())
        }
        .buttonStyle(.plain)
    }
}

/// The globe key has to be a real UIControl so iOS can show the keyboard picker on long press.
private struct NextKeyboardKey: UIViewRepresentable {
    let inputController: UIInputViewController

    func makeUIView(context: Context) -> UIButton {
        let button = UIButton(type: .system)
        button.setImage(UIImage(systemName: "globe"), for: .normal)
        button.accessibilityLabel = VoxtL10n.shared.string("Next keyboard")
        button.tintColor = UIColor(VoxtTheme.Colors.ink)
        button.addTarget(
            inputController,
            action: #selector(UIInputViewController.handleInputModeList(from:with:)),
            for: .allTouchEvents
        )
        return button
    }

    func updateUIView(_ uiView: UIButton, context: Context) {
        uiView.accessibilityLabel = VoxtL10n.shared.string("Next keyboard")
    }
}

/// Plays KeyFeedback: a prepared light impact and the system input click (which follows the
/// user's Keyboard Clicks setting). Both need Full Access in a keyboard extension.
@MainActor
final class KeyFeedbackPlayer {
    static let shared = KeyFeedbackPlayer()
    private let impact = UIImpactFeedbackGenerator(style: .light)
    var hasFullAccess = false {
        didSet { if hasFullAccess { impact.prepare() } }
    }
    /// The app's Haptic feedback setting, read when the keyboard appears.
    var hapticFeedback = true

    func play(_ event: KeyFeedback.Event) {
        let outputs = KeyFeedback.outputs(for: event, hasFullAccess: hasFullAccess, hapticFeedback: hapticFeedback)
        if outputs.contains(.haptic) {
            impact.impactOccurred()
            impact.prepare()
        }
        if outputs.contains(.click) { UIDevice.current.playInputClick() }
    }
}

/// The keyboard's root input view. Adopting UIInputViewAudioFeedback lets playInputClick sound.
final class KeyboardInputView: UIInputView, UIInputViewAudioFeedback {
    var enableInputClicksWhenVisible: Bool { true }
}

private struct KeyFeedbackModifier: ViewModifier {
    let event: KeyFeedback.Event
    @State private var pressed = false

    func body(content: Content) -> some View {
        content.simultaneousGesture(DragGesture(minimumDistance: 0)
            .onChanged { _ in
                guard !pressed else { return }
                pressed = true
                KeyFeedbackPlayer.shared.play(event)
            }
            .onEnded { _ in pressed = false })
    }
}

extension View {
    /// Feedback on touch-down, before the button's action runs on touch-up.
    func keyFeedback(_ event: KeyFeedback.Event) -> some View { modifier(KeyFeedbackModifier(event: event)) }
}
