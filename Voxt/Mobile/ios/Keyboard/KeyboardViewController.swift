import SwiftUI
import UIKit
import VoxtMobileKit
import TypingHost
import TypingInput

/// The Voxt keyboard: a mic key that starts and stops dictation in the Voxt app,
/// plus return, delete and the next-keyboard key.
final class KeyboardViewController: UIInputViewController {
    private let model = KeyboardModel()
    private var host: UIHostingController<KeyboardView>?
    private var keyboardHeight: NSLayoutConstraint?
    private var keyboardVisible = false

    override func loadView() {
        // Input clicks only sound when the input view adopts UIInputViewAudioFeedback.
        inputView = KeyboardInputView(frame: .zero, inputViewStyle: .keyboard)
        // The mic on the keyboard is ours. Apple: "If the value of this property is true, the
        // system dictation key is in a disabled state" (UIInputViewController docs). On iOS 26.3
        // setting it in viewDidLoad leaves the key enabled; set here and in viewWillAppear it
        // is disabled (still drawn, not hidden).
        hasDictationKey = true
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        model.insertText = { [weak self] in self?.textDocumentProxy.insertText($0) }
        model.deleteBackward = { [weak self] in self?.textDocumentProxy.deleteBackward() }
        model.setMarkedText = { [weak self] text in
            self?.textDocumentProxy.setMarkedText(text, selectedRange: NSRange(location: text.utf16.count, length: 0))
        }
        model.unmarkText = { [weak self] in self?.textDocumentProxy.unmarkText() }
        model.openApp = { [weak self] url in self?.openContainingApp(url) }
        model.contextProvider = { [weak self] in
            // iOS 26.3: a late textDidChange after dismissal crashes while
            // bridging documentIdentifier to UUID. Do not query a hidden proxy.
            guard let self, self.keyboardVisible, self.viewIfLoaded?.window != nil else { return nil }
            let proxy = self.textDocumentProxy
            return TextProxySnapshot(proxy)
        }
        model.hasFullAccess = hasFullAccess
        KeyFeedbackPlayer.shared.hasFullAccess = hasFullAccess
        model.globeRowHeight = needsInputModeSwitchKey ? 36 : 0
        model.onKeyboardHeightChange = { [weak self] height in self?.keyboardHeight?.constant = height }

        let view = KeyboardView(
            model: model,
            showsNextKeyboard: needsInputModeSwitchKey,
            inputController: self
        )
        let host = UIHostingController(rootView: view)
        host.view.backgroundColor = .clear
        host.view.translatesAutoresizingMaskIntoConstraints = false
        addChild(host)
        self.view.addSubview(host.view)
        let keyboardHeight = host.view.heightAnchor.constraint(equalToConstant: 300)
        self.keyboardHeight = keyboardHeight
        NSLayoutConstraint.activate([
            host.view.leadingAnchor.constraint(equalTo: self.view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: self.view.trailingAnchor),
            host.view.topAnchor.constraint(equalTo: self.view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: self.view.bottomAnchor),
            keyboardHeight,
        ])
        host.didMove(toParent: self)
        self.host = host
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        hasDictationKey = true
        keyboardVisible = true
        model.hasFullAccess = hasFullAccess
        KeyFeedbackPlayer.shared.hasFullAccess = hasFullAccess
        let defaults = UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard
        KeyFeedbackPlayer.shared.hapticFeedback = KeyboardSettings.load(from: defaults).hapticFeedback
        reportFullAccess()
        model.attach()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // Full Access can be switched on in Settings while the keyboard was off screen.
        model.hasFullAccess = hasFullAccess
        KeyFeedbackPlayer.shared.hasFullAccess = hasFullAccess
        reportFullAccess()
    }

    /// Tells the app (Try it) that Full Access is on. Without Full Access the App Group
    /// isn't shared, so only "on" can ever arrive; the app treats silence as off.
    private func reportFullAccess() {
        guard let defaults = UserDefaults(suiteName: DictationChannel.appGroup) else { return }
        KeyboardStatus.reportAppearance(hasFullAccess: hasFullAccess, to: defaults)
    }

    override func viewWillDisappear(_ animated: Bool) {
        model.prepareForDisappearance()
        keyboardVisible = false
        super.viewWillDisappear(animated)
        model.detach()
    }

    override func textDidChange(_ textInput: (any UITextInput)?) {
        super.textDidChange(textInput)
        guard keyboardVisible else { return }
        model.refreshTextSuggestions()
    }

    override func selectionDidChange(_ textInput: (any UITextInput)?) {
        super.selectionDidChange(textInput)
        guard keyboardVisible else { return }
        model.refreshTextSuggestions()
    }

    /// Extensions can't call UIApplication.shared; the app object is still reachable
    /// through the responder chain.
    private func openContainingApp(_ url: URL) {
        var responder: UIResponder? = self
        while let current = responder {
            if let application = current as? UIApplication {
                application.open(url, options: [:], completionHandler: nil)
                return
            }
            responder = current.next
        }
    }
}

@MainActor
@Observable
final class KeyboardModel {
    var insertionFeedbackCount = 0
    var audioLevel: Float = 0
    private var levelTimer: Timer?
    var phase: DictationChannel.Phase = .idle
    var message: String?
    var recoveryText: String?
    var hasFullAccess = false
    var sessionAlive = false
    var showsTextKeyboard = false { didSet { updateKeyboardHeight() } }
    var textKeyboard = TextKeyboardState()
    /// The marked text in the field and its candidate bar, for any composing layout.
    var composition = MarkedComposition()
    var conversionError: String?
    /// The active composing layout's engine. Nil for direct layouts from the kit catalog.
    private var engine: (any InputEngine)?
    private var engineLayoutID: String?
    private var candidateSources: [String: any CandidateSource] = [:]
    private var conversionTask: Task<Void, Never>?
    private var requestedConversion: ConversionRequest?
    private enum CompositionAction: Sendable {
        case key(InputKey), candidate(String, reading: String), voice, numbers
        /// Mark the next piece after its predecessor was committed (Hangul).
        case continueMark(String)
        case layout(TextKeyboardLayoutID), cycle(Int)
    }
    private var compositionActions = CompositionWriteQueue<CompositionAction>()
    private var performingCompositionAction = false
    private var pendingFieldWrite: PendingFieldWrite?
    private var fieldWriteWait: Task<Void, Never>?
    private var compositionFieldID: UUID?
    private var writingMarkedText = false
    var setMarkedText: (String) -> Void = { _ in }
    var unmarkText: () -> Void = {}
    var learnedRecord: CorrectionLearningStore.Record? { didSet { updateKeyboardHeight() } }
    var learningError: String? { didSet { updateKeyboardHeight() } }
    var styleCandidate: PersonalStyleExample? { didSet { updateKeyboardHeight() } }
    var styleNotice: String? { didSet { updateKeyboardHeight() } }
    private var styleStore: PersonalStyleStore?
    private var pendingStyle: CorrectionLearningRequest?
    private var styleConsentRevision: String?
    private var learningConsentRevision: String?
    private var learningSession: CorrectionLearningSession?
    private var learningEdits: CorrectionEditTracker?
    private var pendingLearning: (id: UUID, request: CorrectionLearningRequest)?
    private var learningStore: CorrectionLearningStore?
    private var learningChannel: CorrectionLearningChannel?
    var suggestions: [String] = [] { didSet { updateKeyboardHeight() } }
    var onKeyboardHeightChange: (CGFloat) -> Void = { _ in }

    var canRetry = false
    /// What the in-flight Edit changes, shown while listening.
    var editScope: EditScope?
    /// ↶ is offered while the latest insert (dictation or edit) is still at the cursor.
    var canUndo = false
    var insertText: (String) -> Void = { _ in }
    var deleteBackward: () -> Void = {}
    var openApp: (URL) -> Void = { _ in }
    var contextProvider: @MainActor () -> CompositionTextContext? = { nil }

    private var channel: DictationChannel?
    private var composer = KeyboardComposer(defaults: .standard)
    private var timer: Timer?
    private var observing = false

    /// The mic's hold-and-slide state while a finger is on it.
    private var micGesture: MicHoldGesture?
    private var micHoldTimer: Task<Void, Never>?
    private var micOffset = MicHoldGesture.Offset.zero
    private(set) var translationPills: [MicHoldGesture.Pill] = []
    private(set) var enteredPill: Int?
    /// The language the current recording translates into (shown while recording).
    private(set) var translatingTo: String?

    func micDown() {
        let busy = phase == .recording || phase == .processing
        let defaults = UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard
        // While recording, the mic only stops; no pills.
        micGesture = MicHoldGesture(targets: busy ? [] : KeyboardSettings.load(from: defaults).translationTargets)
        micGesture?.down(at: Date())
        micOffset = .zero
        guard !busy else { return }
        micHoldTimer?.cancel()
        micHoldTimer = Task { [weak self] in
            try? await Task.sleep(for: .seconds(MicHoldGesture.holdDelay + 0.02))
            guard !Task.isCancelled else { return }
            self?.micMoved(self?.micOffset ?? .zero)
        }
    }

    func micMoved(_ offset: MicHoldGesture.Offset) {
        micOffset = offset
        guard var gesture = micGesture, phase != .recording, phase != .processing else { return }
        let feedback = gesture.update(now: Date(), offset: offset)
        micGesture = gesture
        translationPills = gesture.pills
        enteredPill = gesture.entered
        if !feedback.isEmpty { KeyFeedbackPlayer.shared.play(.mode) }
    }

    func micUp(_ offset: MicHoldGesture.Offset) {
        micHoldTimer?.cancel()
        guard var gesture = micGesture else { micTapped(); return }
        let busy = phase == .recording || phase == .processing
        let outcome = busy ? .dictate : gesture.up(now: Date(), offset: offset)
        micGesture = nil
        translationPills = []
        enteredPill = nil
        switch outcome {
        case .dictate:
            KeyFeedbackPlayer.shared.play(.mic)
            micTapped()
        case let .translate(code):
            KeyFeedbackPlayer.shared.play(.mic)
            micTapped(translateTo: code)
        case .chooseLanguages: openApp(URL(string: "voxtmobile://translation")!)
        case .nothing: break
        }
    }

    /// What the pen armed for the next recording; nil is plain dictation.
    private(set) var penIntent: PenIntent?

    /// The pen: one tap arms editing the selection, editing our last result, or writing
    /// from a description, decided by the field. A second tap disarms it.
    func penTapped() {
        guard phase != .recording, phase != .processing else { return }
        if penIntent != nil { penIntent = nil; return }
        guard let context = contextProvider() else {
            message = "Switch to a text field and try again."
            return
        }
        message = nil
        penIntent = composer.penIntent(context: context)
    }

    /// The selection or the cursor moved: the armed pen follows what is now in the field.
    private func refreshPen() {
        guard penIntent != nil, let context = contextProvider() else { return }
        penIntent = composer.penIntent(context: context)
    }

    func showTyping() {
        textKeyboard.reloadEnabledLayouts(from: UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard)
        showsTextKeyboard = true
        updateAutoShift()
        refreshTextSuggestions()
    }

    func pressTextKey(_ key: TextKeyboardKey) {
        // While composing, the numbers plane's characters go to the engine too
        // (pinyin: "，" commits the first candidate before itself).
        if textKeyboard.plane == .numbers, !composition.reading.isEmpty, activeLayout != nil {
            switch key.kind {
            case .character: press(.text(key.text ?? key.label)); updateKeyboardHeight(); return
            case .space: press(.space); updateKeyboardHeight(); return
            case .return: press(.return); updateKeyboardHeight(); return
            case .delete: press(.delete); updateKeyboardHeight(); return
            default: break
            }
        }
        guard textKeyboard.plane == .letters, activeKeyRows != nil else {
            applyTextAction(textKeyboard.press(key))
            // Back on letters after ¿ or a full stop on the numbers plane.
            if key.kind == .letters { updateAutoShift() }
            updateKeyboardHeight()
            return
        }
        // A layout drawn from its package: letters go through its engine.
        switch key.kind {
        case .character:
            textKeyboard.shifted = false
            press(.text(key.text ?? key.label))
            updateAutoShift()
        case .space: press(.space); updateAutoShift()
        case .return: press(.return); updateAutoShift()
        case .delete: press(.delete); updateAutoShift()
        case .numbers:
            // Keep composing: the next character on the numbers plane goes to the engine,
            // which commits its candidate first (pinyin "xiexie" + "，" → "谢谢，").
            applyTextAction(textKeyboard.press(key))
        case .voice:
            finishComposition()
            applyTextAction(textKeyboard.press(key))
        case .shift, .letters:
            _ = textKeyboard.press(key)
        }
        updateKeyboardHeight()
    }

    func insertAlternate(_ text: String) {
        guard textKeyboard.plane == .letters, activeKeyRows != nil else {
            applyTextAction(textKeyboard.insertAlternate(text))
            return
        }
        let upper = text.uppercased()
        let value = textKeyboard.shifted && upper.count == 1 ? upper : text
        textKeyboard.shifted = false
        press(.text(value))
        updateAutoShift()
    }

    /// A key-grid layout that converts (pinyin) keeps a candidate bar, so the keyboard
    /// doesn't change height while candidates come and go.
    var hasCandidateBar: Bool {
        guard case .keys? = activeLayout?.surface else { return false }
        return activeLayout?.makeCandidateSource != nil && textKeyboard.plane == .letters
    }

    /// The reading itself isn't offered: Return enters the letters.
    var activeCandidates: [String] {
        hasCandidateBar ? composition.candidates.filter { $0 != composition.reading } : []
    }

    /// A converting layout with letters in the field: the candidate bar replaces the toolbar.
    var isComposingCandidates: Bool { hasCandidateBar && !composition.reading.isEmpty }

    /// Pinyin's space bar shows only its small 拼.
    var spaceCaption: String? { hasCandidateBar ? activeLayout?.shortLabel : nil }

    /// Letter rows of the active layout when its package draws them with the shared key grid.
    var activeKeyRows: [[TextKeyboardKey]]? {
        guard case let .keys(rows)? = activeLayout?.surface else { return nil }
        return TypingLayouts.keys(rows, shifted: textKeyboard.shifted)
    }

    /// Shift on where the layout capitalizes (Spanish: start of a sentence, also after ¿ ¡).
    private func updateAutoShift() {
        guard textKeyboard.plane == .letters, composition.reading.isEmpty,
              let engine = activeEngine(), let context = contextProvider() else { return }
        if engine.capitalizesNextKey(context: InputContext(before: context.documentContextBeforeInput ?? "")) {
            textKeyboard.shifted = true
        }
    }

    func selectTextLayout(_ id: TextKeyboardLayoutID) {
        if deferCompositionAction(.layout(id)) { return }
        finishComposition()
        try? textKeyboard.select(id)
        updateAutoShift()
        saveTextPreferences()
        updateKeyboardHeight()
    }

    func cycleTextLayout(_ direction: Int) {
        if deferCompositionAction(.cycle(direction)) { return }
        finishComposition()
        let at = textKeyboard.enabled.firstIndex(of: textKeyboard.active) ?? 0
        if (direction > 0 && at == textKeyboard.enabled.count - 1) || (direction < 0 && at == 0) {
            showsTextKeyboard = false
        } else {
            textKeyboard.cycle(by: direction)
            saveTextPreferences()
        }
        updateKeyboardHeight()
    }

    /// The registered layout for the active saved id, if it composes text.
    private var activeLayout: LayoutDescriptor? { TypingLayouts.registry.layout(id: textKeyboard.active.rawValue) }

    /// Layouts that draw their own key grid (Japanese flick, Zhuyin).
    var activeLayoutSurface: String? {
        guard case let .custom(name)? = activeLayout?.surface else { return nil }
        return name
    }

    private func activeEngine() -> (any InputEngine)? {
        guard let layout = activeLayout else { return nil }
        if engineLayoutID != layout.id || engine == nil {
            engine = layout.makeEngine()
            engineLayoutID = layout.id
        }
        return engine
    }

    private var fieldContext: FieldContext? {
        guard let value = contextProvider() else { return nil }
        return FieldContext(documentID: value.documentIdentifier,
                            before: value.documentContextBeforeInput ?? "",
                            after: value.documentContextAfterInput ?? "",
                            selected: value.selectedText ?? "")
    }

    /// Forgets the composition on both sides: the field's mark and the engine.
    private func resetComposition() {
        composition.reset()
        engine?.reset()
        requestedConversion = nil
    }

    private func validateComposition() {
        guard !writingMarkedText else { return }
        if let pending = pendingFieldWrite {
            if acknowledgeFieldWrite() { return }
            guard let current = fieldContext,
                  current.documentID == pending.origin.documentID,
                  current.after == pending.origin.after, current.selected.isEmpty else {
                if fieldContext?.documentID == pending.origin.documentID { unmarkText() }
                composer.cancel()
                cancelFieldWrites()
                resetComposition()
                compositionFieldID = nil
                conversionTask?.cancel()
                return
            }
            // A write can have several intermediate host callbacks. No next
            // write is sent until its exact final context has been observed.
            return
        }
        guard !writingMarkedText, !composition.reading.isEmpty, !composition.acknowledgeMarkedContext(fieldContext) else { return }
        if fieldContext?.documentID == compositionFieldID { unmarkText() }
        resetComposition()
        compositionFieldID = nil
        conversionTask?.cancel()
        cancelFieldWrites()
    }

    private func deferCompositionAction(_ action: CompositionAction) -> Bool {
        guard !performingCompositionAction else { return false }
        guard compositionActions.enqueue(action) else { return true }
        drainCompositionActions()
        return true
    }

    private func drainCompositionActions() {
        guard !performingCompositionAction else { return }
        performingCompositionAction = true
        defer { performingCompositionAction = false }
        while let action = compositionActions.next() {
            switch action {
            case let .key(key): press(key)
            case let .candidate(candidate, reading):
                if composition.reading == reading { selectCandidate(candidate) }
            case let .continueMark(text): markComposition(text)
            case .voice: compositionVoice()
            case .numbers: compositionNumbers()
            case let .layout(id): selectTextLayout(id)
            case let .cycle(direction): cycleTextLayout(direction)
            }
        }
    }

    private func beginFieldWrite(id: UUID, replacing count: Int, with text: String) {
        guard let origin = fieldContext else { return }
        let write = PendingFieldWrite(id: id, origin: origin, replacing: count, text: text)
        pendingFieldWrite = write
        compositionActions.beginWrite(id: id)
        fieldWriteWait?.cancel()
        fieldWriteWait = Task { [weak self] in
            do {
                try await Task.sleep(for: .milliseconds(80))
                guard let self, self.pendingFieldWrite?.id == id else { return }
                if self.acknowledgeFieldWrite() { return }
                // A host may exclude an active mark from before-context. Commit
                // its mark, then still require the exact text acknowledgement.
                // Elapsed time alone never releases the next queued input.
                if write.canUnmarkForAcknowledgement(self.fieldContext) { self.unmarkText() }
                for _ in 0..<20 {
                    try await Task.sleep(for: .milliseconds(50))
                    guard self.pendingFieldWrite?.id == id else { return }
                    if self.acknowledgeFieldWrite() { return }
                }
                if self.fieldContext?.documentID == origin.documentID { self.unmarkText() }
                self.composer.cancel()
                self.cancelFieldWrites()
                self.resetComposition()
                self.compositionFieldID = nil
                self.conversionTask?.cancel()
                self.conversionError = "Switch to a text field and try again."
            } catch { }
        }
    }

    @discardableResult
    private func acknowledgeFieldWrite() -> Bool {
        guard let pending = pendingFieldWrite, fieldContext == pending.expected else { return false }
        _ = composition.acknowledgeMarkedContext(pending.expected)
        guard compositionActions.acknowledge(id: pending.id) else { return false }
        pendingFieldWrite = nil
        fieldWriteWait?.cancel()
        drainCompositionActions()
        return true
    }

    private func cancelFieldWrites() {
        compositionActions.cancel()
        pendingFieldWrite = nil
        fieldWriteWait?.cancel()
    }

    /// Every key of a composing layout: the engine decides, the model writes.
    func press(_ key: InputKey) {
        if deferCompositionAction(.key(key)) { return }
        validateComposition()
        if composition.reading.isEmpty { engine?.reset() }
        guard var engine = activeEngine() else { return }
        let context = InputContext(before: contextProvider()?.documentContextBeforeInput ?? "")
        let previous = self.engine
        let edit = engine.handle(key, context: context)
        self.engine = engine
        apply(edit, for: key)
        // The field refused the new mark (an unverified host range): undo the key
        // in the engine too, so its reading never runs ahead of the field.
        if case let .mark(text) = edit, !text.isEmpty, composition.reading != text { self.engine = previous }
    }

    private func apply(_ edit: InputEdit, for key: InputKey?) {
        switch edit {
        case .none:
            // The engine changed what it converts without changing the mark
            // (Zhuyin first tone), so ㄊㄚ + Space would skip conversion;
            // refresh the query without another host write.
            if engine?.conversion != requestedConversion,
               let query = composition.update(reading: composition.reading, context: fieldContext) {
                requestCandidates(for: query)
            }
        case let .mark(text):
            markComposition(text)
        case let .insert(text, mark: next):
            if next.isEmpty, text == composition.reading { finishComposition(); return }
            // When the engine already holds the next piece (Hangul ㄱ after 한), keep it.
            commitComposition(text, keepingEngine: !next.isEmpty)
            guard !next.isEmpty else { return }
            continueComposition(with: next)
        case .passThrough:
            switch key {
            case let .text(text)?: applyTextAction(.insert(text))
            case .delete?: applyTextAction(.delete)
            case .space?: applyTextAction(.insert(" "))
            case .return?: applyTextAction(.insert("\n"))
            case .modifier?, nil: break
            }
        }
    }

    private func markComposition(_ reading: String) {
        guard reading != composition.reading else { return }
        // A field switch can leave an old mark without dismissing the extension.
        // Complete any such mark before starting a new reading in the live field.
        if composition.reading.isEmpty { unmarkText() }
        let visibleCount: Int
        if composition.reading.isEmpty { visibleCount = 0 }
        else {
            guard let count = composition.visibleReadingCount(in: fieldContext) else { return }
            visibleCount = count
        }
        guard let query = composition.update(reading: reading, context: fieldContext) else { return }
        clearLearningSession()
        compositionFieldID = fieldContext?.documentID
        beginFieldWrite(id: query.id, replacing: visibleCount, with: reading)
        writingMarkedText = true
        // iOS 26.3 SwiftUI field: a second setMarkedText appends instead of
        // replacing the old kana. Unmark and remove only the exact
        // suffix verified above when the host exposes it in before-context.
        if visibleCount > 0 {
            unmarkText()
            for _ in 0..<visibleCount { deleteBackward() }
        }
        setMarkedText(reading)
        composition.didMark(context: fieldContext, for: query)
        writingMarkedText = false
        acknowledgeFieldWrite()
        if reading.isEmpty {
            unmarkText()
            resetComposition()
            compositionFieldID = nil
        }
        requestCandidates(for: query)
    }

    private func requestCandidates(for query: MarkedComposition.Query) {
        conversionTask?.cancel()
        conversionError = nil
        requestedConversion = engine?.conversion
        guard !query.reading.isEmpty, let request = engine?.conversion, let layout = activeLayout,
              let makeSource = layout.makeCandidateSource else { return }
        conversionTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .milliseconds(40))
                guard let self else { return }
                let source: any CandidateSource
                if let existing = self.candidateSources[layout.id] { source = existing }
                else {
                    source = try makeSource()
                    self.candidateSources[layout.id] = source
                }
                #if DEBUG
                let conversionStart = Date()
                #endif
                let candidates = try await source.candidates(for: request)
                #if DEBUG
                // Count and duration only: never log the user's reading or candidates.
                NSLog("Conversion layout=%@ chars=%d elapsedMs=%.1f", layout.id, query.reading.count,
                      Date().timeIntervalSince(conversionStart) * 1_000)
                #endif
                try Task.checkCancellation()
                self.validateComposition()
                if self.composition.receive(candidates.map(\.text), for: query) {
                    self.engine?.receive(candidates, for: request)
                }
            } catch is CancellationError {
                return
            } catch CandidateSourceError.unconvertible {
                // An unlisted syllable keeps its raw reading editable.
                return
            } catch {
                guard let self, !Task.isCancelled else { return }
                self.validateComposition()
                if self.composition.receive([], for: query) {
                    self.conversionError = layout.unavailableMessage
                }
            }
        }
    }

    func selectCandidate(_ candidate: String) {
        if deferCompositionAction(.candidate(candidate, reading: composition.reading)) { return }
        validateComposition()
        if candidate == composition.reading { finishComposition(); return }
        guard var engine else { return }
        let match = engine.state.candidates.first { $0.text == candidate } ?? Candidate(text: candidate)
        let edit = engine.select(match)
        self.engine = engine
        apply(edit, for: nil)
    }

    /// Marks the next piece after a commit, ahead of keys already waiting.
    /// A mark set in the same main-queue turn as the commit's unmarkText replaces the
    /// committed 한 (the field shows ㄱ, then ㄱㄹ). So the mark waits for the commit's
    /// acknowledgement, and never runs in the commit's own turn.
    private func continueComposition(with next: String) {
        if pendingFieldWrite != nil {
            if !compositionActions.enqueueFirst(.continueMark(next)) { engine?.reset() }
            return
        }
        let turn = UUID()
        compositionActions.beginWrite(id: turn)
        DispatchQueue.main.async { [weak self] in
            guard let self, self.compositionActions.acknowledge(id: turn) else { return }
            self.markComposition(next)
            self.drainCompositionActions()
        }
    }

    /// Replaces the verified mark with `text` for good.
    private func commitComposition(_ text: String, keepingEngine: Bool = false) {
        guard let visibleCount = composition.visibleReadingCount(in: fieldContext) else { return }
        let committed = composition.candidates.contains(text)
            ? composition.commit(text, context: fieldContext)
            : composition.commitText(text, context: fieldContext)
        guard let committed else {
            validateComposition()
            return
        }
        if !keepingEngine { engine?.reset() }
        beginFieldWrite(id: UUID(), replacing: visibleCount, with: committed)
        writingMarkedText = true
        if visibleCount > 0 {
            unmarkText()
            for _ in 0..<visibleCount { deleteBackward() }
        }
        setMarkedText(committed)
        unmarkText()
        writingMarkedText = false
        acknowledgeFieldWrite()
        compositionFieldID = nil
        conversionTask?.cancel()
        conversionError = nil
    }

    func finishComposition() {
        guard !composition.reading.isEmpty else { return }
        validateComposition()
        guard composition.matches(fieldContext) else { return }
        unmarkText()
        resetComposition()
        compositionFieldID = nil
        conversionTask?.cancel()
    }

    func compositionVoice() {
        if deferCompositionAction(.voice) { return }
        finishComposition(); showsTextKeyboard = false
    }

    func compositionNumbers() {
        if deferCompositionAction(.numbers) { return }
        finishComposition()
        textKeyboard.plane = .numbers
        updateKeyboardHeight()
    }

    func refreshTextSuggestions() {
        validateComposition()
        refreshPen()
        guard showsTextKeyboard, let context = contextProvider(), (context.selectedText ?? "").isEmpty else {
            suggestions = []
            return
        }
        let defaults = UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard
        let matches = TextKeyboardSuggestions.matches(
            before: context.documentContextBeforeInput ?? "",
            after: context.documentContextAfterInput ?? "",
            vocabulary: PersonalDictionary(defaults: defaults).vocabulary
        )
        suggestions = Array(matches.prefix(3))
    }

    func insertSuggestion(_ suggestion: String) {
        guard let context = contextProvider(), (context.selectedText ?? "").isEmpty,
              let replacement = TextKeyboardSuggestions.replacement(
                before: context.documentContextBeforeInput ?? "",
                after: context.documentContextAfterInput ?? "",
                suggestion: suggestion
              ) else { return }
        for _ in 0..<replacement.deleteCount { applyTextAction(.delete) }
        applyTextAction(.insert(replacement.text))
        refreshTextSuggestions()
    }

    func backspace() { applyTextAction(.delete) }
    func returnKey() { applyTextAction(.insert("\n")) }

    private func applyTextAction(_ action: TextKeyboardAction?) {
        if let context = contextProvider(), learningEdits != nil {
            let before = context.documentContextBeforeInput ?? ""
            let after = context.documentContextAfterInput ?? ""
            let selected = context.selectedText ?? ""
            let accepted: Bool?
            switch action {
            case .insert(let text): accepted = learningEdits?.replace(before: before, selected: selected, after: after, with: text)
            case .delete: accepted = learningEdits?.delete(before: before, selected: selected, after: after)
            default: accepted = true
            }
            if accepted != true { clearLearningSession() }
        }
        switch action {
        case .insert(let text): insertText(text)
        case .delete: deleteBackward()
        case .voice: showsTextKeyboard = false
        case nil: break
        }
        refreshTextSuggestions()
    }

    private func saveTextPreferences() {
        textKeyboard.save(to: UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard)
    }

    private func updateKeyboardHeight() {
        if styleCandidate != nil { onKeyboardHeightChange(270); return }
        let noticeHeight = learnedRecord == nil && learningError == nil && styleNotice == nil ? 0 : 40
        let textHeight = activeLayoutSurface != nil && textKeyboard.plane == .letters ? 350 : keyGridHeight
        onKeyboardHeightChange(CGFloat((showsTextKeyboard ? textHeight : 300) + noticeHeight))
    }

    /// The globe row under the keys, on iPhones that need one (needsInputModeSwitchKey).
    var globeRowHeight: CGFloat = 0

    /// TextKeyboardView's height: the row above the keys, four key rows, the globe row.
    private var keyGridHeight: Int {
        let top: CGFloat = hasCandidateBar ? 40 : 30 + (suggestions.isEmpty ? 0 : 32 + TextKeyboardGeometry.rowGap)
        let keys = 4 * TextKeyboardGeometry.keyHeight + 3 * TextKeyboardGeometry.rowGap
        return Int((16 + top + TextKeyboardGeometry.rowGap + keys + globeRowHeight).rounded(.up))
    }

    func attach() {
        cancelFieldWrites()
        conversionTask?.cancel()
        conversionError = nil
        // iOS 26.3: a dismissed keyboard can leave its last kana marked in the
        // host after its context has disappeared. Finish that mark on
        // reattachment before a new reading can replace it.
        unmarkText()
        resetComposition()
        compositionFieldID = nil
        let defaults = UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard
        learningStore = CorrectionLearningStore(defaults: defaults)
        styleStore = PersonalStyleStore(defaults: defaults)
        learningChannel = CorrectionLearningChannel(defaults: defaults)
        textKeyboard = TextKeyboardState.load(from: defaults)
        showsTextKeyboard = false
        composer = KeyboardComposer(defaults: defaults)
        channel = hasFullAccess ? DictationChannel(defaults: defaults) : nil
        composer.prepareForAppear(stateRequestID: channel?.state.requestID, stateResultID: channel?.state.resultID)
        refresh()
        if !observing {
            observing = true
            let observer = Unmanaged.passUnretained(self).toOpaque()
            CFNotificationCenterAddObserver(
                CFNotificationCenterGetDarwinNotifyCenter(),
                observer,
                { _, observer, _, _, _ in
                    guard let observer else { return }
                    let model = Unmanaged<KeyboardModel>.fromOpaque(observer).takeUnretainedValue()
                    Task { @MainActor in model.refresh() }
                },
                DictationChannel.stateNotification as CFString,
                nil,
                .deliverImmediately
            )
        }
        levelTimer?.invalidate()
        levelTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let level: Float = self.phase == .recording ? (self.channel?.audioLevel ?? 0) : 0
                if self.audioLevel != level { self.audioLevel = level }
            }
        }
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
    }

    func prepareForDisappearance() {
        // Do not validate/drain pending keys while UIKit is removing the proxy.
        // Persist cancellation before clearing the reading so a late voice result
        // cannot regain permission to insert on the next appearance.
        composer.prepareForTextInputDetach(isComposingText: !composition.reading.isEmpty,
                                           hasPendingTextWrite: pendingFieldWrite != nil)
        cancelFieldWrites()
        if let field = fieldContext?.documentID, field == compositionFieldID { unmarkText() }
        conversionTask?.cancel()
        resetComposition()
        compositionFieldID = nil
    }

    func detach() {
        prepareForDisappearance()
        cancelFieldWrites()
        conversionTask?.cancel()
        conversionError = nil
        resetComposition()
        compositionFieldID = nil
        clearLearningSession()
        learnedRecord = nil
        learningError = nil
        styleNotice = nil
        composer.persist()
        levelTimer?.invalidate()
        levelTimer = nil
        timer?.invalidate()
        timer = nil
        if observing {
            CFNotificationCenterRemoveEveryObserver(
                CFNotificationCenterGetDarwinNotifyCenter(),
                Unmanaged.passUnretained(self).toOpaque()
            )
            observing = false
        }
    }

    func micTapped(translateTo language: String? = nil) {
        guard hasFullAccess, let channel else {
            message = "Turn on Allow Full Access for the Voxt keyboard in Settings."
            return
        }
        switch phase {
        case .recording:
            channel.send(.stop, requestID: composer.gate.origin?.requestID)
        case .processing:
            break
        case .idle, .done, .failed:
            guard let context = contextProvider() else {
                message = "Switch to a text field and try again."
                return
            }
            do {
                // Decide again from the field as it is now, then back to plain dictation.
                let command: CompositionCommand
                if let language {
                    command = try composer.begin(context: context, translateTo: language)
                } else {
                    let intent = penIntent.flatMap { _ in composer.penIntent(context: context) }
                    command = try composer.begin(context: context, intent: intent)
                }
                penIntent = nil
                translatingTo = composer.translatingTo
                clearLearningSession()
                channel.send(.start, options: command.options, requestID: command.requestID)
                if !channel.isSessionAlive() {
                    composer.prepareForHandoff()
                    // The app has to come forward once to start the microphone.
                    // The start command is already in the app group; a cold launch
                    // marks it handled, and the deep link reads those options.
                    message = "Opening Voxt to start the mic. Tap ◀ to come back."
                    openApp(URL(string: "voxtmobile://session")!)
                }
            } catch {
                message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            }
        }
    }

    func cancelTapped() {
        translatingTo = nil
        clearLearningSession()
        composer.cancel()
        channel?.send(.cancel, requestID: composer.gate.origin?.requestID)
    }

    func retryTapped() {
        clearLearningSession()
        guard let channel else { return }
        if channel.state.retryOpensApp(sessionAlive: channel.isSessionAlive()) { reconnectInApp(); return }
        channel.send(.retry, options: nil, requestID: composer.gate.origin?.requestID)
    }

    func openVoxtSettings() { openApp(URL(string: "voxtmobile://settings")!) }

    /// The request whose failure already opened Voxt to reconnect.
    private var reconnectOpenedFor: String?

    private func reconnectInApp() {
        reconnectOpenedFor = channel?.state.requestID
        message = "Opening Voxt to reconnect to your Mac. Tap ◀ to come back."
        openApp(URL(string: "voxtmobile://reconnect")!)
    }

    func refresh() {
        guard let channel else {
            clearLearningSession()
            phase = .idle
            sessionAlive = false
            canRetry = false
            return
        }
        if channel.isStalled() {
            // The app stopped running mid-request, which would leave "Tap again to finish"
            // up forever. Cancel that request so a late result can't insert, and say so; the
            // next mic tap finds no live session and reopens Voxt.
            let stalled = channel.state
            composer.cancel()
            channel.send(.cancel, requestID: stalled.requestID)
            channel.publish(.init(phase: .failed, errorMessage: "Voxt stopped. Tap the mic to open it again.", requestID: stalled.requestID))
        }
        let state = channel.state
        if state.phase == .done, composition.reading.isEmpty && pendingFieldWrite == nil {
            composer.resumeAfterHandoff(current: contextProvider(), stateRequestID: state.requestID)
        }
        recoveryText = nil
        let busy = state.phase == .recording || state.phase == .processing
        canUndo = !busy && composer.canUndo(context: contextProvider())
        editScope = busy ? composer.editScope : nil
        translatingTo = busy ? composer.translatingTo : nil
        phase = state.phase
        // Edits to a previous dictation while recording would produce a style
        // review that hides Stop and Cancel. Observe
        // only when no recording or composition is in progress, including starts
        // initiated by the host app rather than this keyboard.
        if phase == .recording || phase == .processing {
            clearLearningSession()
        } else {
            observeCorrections()
        }
        sessionAlive = channel.isSessionAlive()
        canRetry = state.canRetry ?? false
        switch state.phase {
        case .failed:
            message = state.errorMessage
            // The Mac was out of reach from the background: Voxt reconnects in front and resends.
            if state.opensAppToReconnect(ownRequestID: composer.gate.origin?.requestID, alreadyOpened: reconnectOpenedFor) {
                reconnectInApp()
            }
        case .recording, .processing:
            message = nil
        case .idle, .done:
            if message?.hasPrefix("Opening Voxt") == true, sessionAlive { message = nil }
        }
        guard let insertion = channel.textToInsert(lastInsertedID: composer.gate.lastInsertedResultID) else { return }
        let current = contextProvider()
        #if DEBUG
        // [DEBUG-handoff] Lengths only, to verify the field-changing UI test's proxy.
        UserDefaults(suiteName: DictationChannel.appGroup)?.set([
            "originBefore": composer.gate.origin?.field.contextBefore.count ?? -1,
            "originAfter": composer.gate.origin?.field.contextAfter.count ?? -1,
            "currentBefore": current?.documentContextBeforeInput?.count ?? -1,
            "currentAfter": current?.documentContextAfterInput?.count ?? -1,
            "documentMatches": current?.documentIdentifier == composer.gate.origin?.field.documentID
        ], forKey: "debug.handoffContextLengths")
        #endif
        let decision = composer.receive(
            current: current,
            text: insertion.text,
            resultID: insertion.resultID,
            requestID: insertion.requestID ?? state.requestID,
            isComposingText: !composition.reading.isEmpty,
            hasPendingTextWrite: pendingFieldWrite != nil
        )
        switch decision {
        case let .insert(text):
            clearLearningSession()
            learnedRecord = nil
            learningError = nil
            styleNotice = nil
            insertText(text)
            channel.markInserted(requestID: insertion.requestID ?? state.requestID)
            insertionFeedbackCount += 1
            if learningStore?.isEnabled == true || styleStore?.isEnabled == true, let current, let insertedContext = contextProvider(),
               insertedContext.documentIdentifier == current.documentIdentifier,
               (insertedContext.documentContextBeforeInput ?? "") + (insertedContext.documentContextAfterInput ?? "")
                == (current.documentContextBeforeInput ?? "") + text + (current.documentContextAfterInput ?? "") {
                let before = current.documentContextBeforeInput ?? ""
                let after = current.documentContextAfterInput ?? ""
                learningSession = CorrectionLearningSession(
                    documentID: current.documentIdentifier, before: before, inserted: text, after: after
                )
                learningEdits = CorrectionEditTracker(text: before + text + after)
                styleConsentRevision = styleStore?.consentRevision
                learningConsentRevision = learningStore?.consentRevision
            }
            message = nil
        case .recoverFromHistory:
            recoveryText = insertion.text
            message = "Text is ready. Copy it into your field."
        case let .replace(oldText, newText):
            clearLearningSession()
            // receive compared the complete captured field before authorizing this
            // synchronous replacement. No text is removed while awaiting the model.
            for _ in oldText { deleteBackward() }
            insertText(newText)
            channel.markInserted(requestID: insertion.requestID ?? state.requestID)
            message = nil
        case .drop:
            break
        }
        canUndo = composer.canUndo(context: contextProvider())
    }

    /// One Undo for the latest edit, only while the edited text is still verified at the cursor.
    /// Undoes the latest insert once, only while it is still verified at the cursor.
    func undoTapped() {
        guard case let .replace(oldText, newText)? = composer.undo(context: contextProvider()) else {
            canUndo = false
            return
        }
        clearLearningSession()
        for _ in oldText { deleteBackward() }
        if !newText.isEmpty { insertText(newText) }
        canUndo = false
    }

    func copyRecoveredText() {
        guard let recoveryText else { return }
        UIPasteboard.general.string = recoveryText
    }

    func undoLearnedWord() {
        guard let record = learnedRecord else { return }
        learningStore?.undo(record.id)
        learnedRecord = nil
        refreshTextSuggestions()
    }

    private func clearLearningSession() {
        learningSession = nil
        learningEdits = nil
        pendingLearning = nil
        learningChannel?.clear()
        pendingStyle = nil
        styleCandidate = nil
        styleConsentRevision = nil
        learningConsentRevision = nil
    }

    func discardPersonalStyle() { clearLearningSession() }

    func keepPersonalStyle() {
        guard let request = pendingStyle, let example = styleCandidate, let styleStore,
              styleStore.isEnabled, styleStore.consentRevision == styleConsentRevision,
              let context = contextProvider(),
              learningSession?.accepts(request, documentID: context.documentIdentifier,
                                       before: context.documentContextBeforeInput ?? "",
                                       after: context.documentContextAfterInput ?? "",
                                       selectedText: context.selectedText ?? "") == true else {
            clearLearningSession()
            styleNotice = "This edit can no longer be saved."
            return
        }
        do {
            try styleStore.keep(example, expectedRevision: styleConsentRevision)
            styleNotice = "Style example saved."
        } catch PersonalStyleStore.KeepError.full {
            styleNotice = "You already have 8 style examples. Remove one in Settings first."
        } catch PersonalStyleStore.KeepError.duplicate {
            styleNotice = "This style example is already saved."
        } catch {
            styleNotice = "This edit can no longer be saved."
        }
        clearLearningSession()
    }

    private func observeCorrections() {
        guard hasFullAccess, learningStore?.isEnabled == true || styleStore?.isEnabled == true,
              let context = contextProvider() else {
            clearLearningSession()
            return
        }
        let before = context.documentContextBeforeInput ?? ""
        let after = context.documentContextAfterInput ?? ""
        let selected = context.selectedText ?? ""
        guard styleConsentRevision == styleStore?.consentRevision,
              learningConsentRevision == learningStore?.consentRevision else {
            clearLearningSession()
            return
        }
        guard learningEdits?.matches(before: before, selected: selected, after: after) == true else {
            clearLearningSession()
            return
        }
        if let pendingStyle {
            if learningSession?.accepts(pendingStyle, documentID: context.documentIdentifier,
                                        before: before, after: after, selectedText: selected) != true {
                clearLearningSession()
            }
            return
        }
        if let pending = pendingLearning, let result = learningChannel?.takeResult(for: pending.id) {
            pendingLearning = nil
            if learningSession?.accepts(pending.request, documentID: context.documentIdentifier,
                                        before: before, after: after, selectedText: selected) == true {
                if let correction = pending.request.validated(result.correction) {
                    do {
                        learnedRecord = try learningStore?.learn(correction)
                        learningError = nil
                        refreshTextSuggestions()
                    } catch { learningError = error.localizedDescription }
                } else if let error = result.error {
                    learningError = error
                    clearLearningSession()
                } else { clearLearningSession() }
            }
        }
        // Keep one request in flight. A continued edit will be compared after this
        // response is accepted or discarded, without overwriting its request ID.
        if pendingLearning == nil, let request = learningSession?.observeEdit(documentID: context.documentIdentifier, before: before,
                                                  after: after, selectedText: selected) {
            if styleStore?.isEnabled == true,
               let example = PersonalStyleExample.candidate(original: request.original, edited: request.edited) {
                pendingStyle = request
                styleCandidate = example
            } else if learningStore?.isEnabled == true, request.isVocabularyCandidate, let learningChannel,
                      MacFeatures.serves(.corrections, in: UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard) {
                pendingLearning = (learningChannel.submit(request), request)
            } else {
                clearLearningSession()
            }
        }
    }

}

private struct TextProxySnapshot: CompositionTextContext {
    let documentIdentifier: UUID
    let selectedText: String?
    let documentContextBeforeInput: String?
    let documentContextAfterInput: String?

    @MainActor init?(_ proxy: UITextDocumentProxy) {
        // iOS 26.3: UIKit can return nil during dismissal even while
        // the keyboard view is still attached. Swift's nonoptional UUID bridge traps.
        // KVC reads this public property as nullable Objective-C data so we can
        // discard the detached context instead of inventing an input-field identity.
        guard let object = proxy as? NSObject,
              let identifier = object.value(forKey: "documentIdentifier") as? UUID else { return nil }
        documentIdentifier = identifier
        selectedText = proxy.selectedText
        documentContextBeforeInput = proxy.documentContextBeforeInput
        documentContextAfterInput = proxy.documentContextAfterInput
    }
}
