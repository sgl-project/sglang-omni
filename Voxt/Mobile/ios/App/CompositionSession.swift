import Foundation
import VoxtMobileKit

protocol CompositionTextContext {
    var documentIdentifier: UUID { get }
    var selectedText: String? { get }
    var documentContextBeforeInput: String? { get }
    var documentContextAfterInput: String? { get }
}

struct CapturedField: Codable, Equatable {
    var documentID: UUID
    var selectedText: String
    var contextBefore: String
    var contextAfter: String
    // Optional for gates saved before availability was recorded. Unknown legacy
    // availability cannot authorize a handoff rebind.
    var contextBeforeAvailable: Bool?
    var contextAfterAvailable: Bool?

    func matchesContext(_ other: CapturedField) -> Bool {
        selectedText == other.selectedText && contextBefore == other.contextBefore
            && contextAfter == other.contextAfter
            && contextBeforeAvailable == other.contextBeforeAvailable
            && contextAfterAvailable == other.contextAfterAvailable
    }
}

struct FieldOrigin: Codable, Equatable {
    var field: CapturedField
    var requestID: String
}

enum InsertionDecision: Equatable {
    case insert(String)
    /// One-tap re-edit outcome: the old text is still in the field; the caller
    /// deletes exactly `oldText` and inserts `newText` in its place.
    case replace(oldText: String, newText: String)
    case recoverFromHistory
    case drop
}

/// The keyboard's most recent voice insertion, for "Edit latest result".
struct LatestInsertion: Codable, Equatable {
    var documentID: UUID
    /// Context before the inserted run, excluding it.
    var before: String
    var inserted: String
    /// Context after the inserted run.
    var after: String
    var resultID: String
    /// The space added before `inserted` (part of `before`); Undo removes it too.
    var separator: String? = nil
}

/// An in-flight re-edit of `LatestInsertion`: the original stays in the field
/// until a strict re-check authorizes the replacement on arrival.
struct PendingReEdit: Codable, Equatable {
    /// The post-insertion snapshot the field must still match.
    var field: CapturedField
    var oldText: String
    var requestID: String
}

enum CompositionReEditError: Error, Equatable, LocalizedError {
    case unavailable

    var errorDescription: String? {
        switch self {
        case .unavailable: "The result to edit is no longer at the cursor."
        }
    }
}

enum CompositionEditScopeError: Error, Equatable, LocalizedError {
    case nothingToEdit
    var errorDescription: String? { "Select text to edit, or dictate to add" }
}

/// What an Edit request changes. With no selection only the keyboard's own last result,
/// verified at the cursor, is edited; never the whole field (host context is truncated).
enum EditScope: String, Codable, Equatable { case selection, lastResult }

/// What the keyboard's pen acts on, for one recording.
enum PenIntent: Equatable {
    case editSelection, editLastResult, write

    /// Shown above the mic before the user speaks.
    var line: String {
        switch self {
        case .editSelection: "Editing your selection"
        case .editLastResult: "Editing what you just said"
        case .write: "Describe what to write"
        }
    }
}

/// One Undo for the latest edit: valid only while the edited text is still exactly at the cursor.
struct EditUndo: Codable, Equatable {
    var documentID: UUID
    var before: String
    var newText: String
    var oldText: String
    var after: String
}

/// The space to put between the text before the cursor and a new result, so consecutive
/// dictations don't run together ("Friday.Please"). Chinese and Japanese take no spaces.
enum CompositionSpacing {
    static func separator(before: String, text: String) -> String {
        guard let last = before.last, let first = text.first else { return "" }
        if last.isWhitespace || first.isWhitespace { return "" }
        if "([{\u{201C}\u{2018}\u{00AB}".contains(last) { return "" }
        if ",.;:!?)]}\u{201D}\u{2019}\u{00BB}\u{2026}".contains(first) { return "" }
        if isCJK(last) || isCJK(first) { return "" }
        return " "
    }

    private static func isCJK(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x3000...0x30FF, 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF, 0xFF00...0xFFEF, 0x20000...0x323AF: true
            default: false
            }
        }
    }
}

struct CompositionCommand: Equatable {
    enum Kind: Equatable { case start, stop, cancel, retry }
    var kind: Kind
    var options: CompositionOptions?
    var requestID: String?
}

enum CompositionCapture {
    static let contextLimit = 2_000

    static func capture(_ context: CompositionTextContext) -> CapturedField {
        CapturedField(
            documentID: context.documentIdentifier,
            selectedText: context.selectedText ?? "",
            contextBefore: String((context.documentContextBeforeInput ?? "").suffix(contextLimit)),
            contextAfter: String((context.documentContextAfterInput ?? "").prefix(contextLimit)),
            contextBeforeAvailable: context.documentContextBeforeInput != nil,
            contextAfterAvailable: context.documentContextAfterInput != nil
        )
    }
}

struct CompositionPreferences: Codable, Equatable {
    var mode: CompositionOptions.Mode
    var style: CompositionOptions.Style
    var targetLanguages: [String]
    var activeTargetLanguage: String?
    var literalDictation = false
    var allowCloudModels = false

    private enum CodingKeys: String, CodingKey {
        case mode, style, targetLanguages, activeTargetLanguage, literalDictation, allowCloudModels
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        mode = try c.decode(CompositionOptions.Mode.self, forKey: .mode)
        style = try c.decode(CompositionOptions.Style.self, forKey: .style)
        targetLanguages = try c.decode([String].self, forKey: .targetLanguages)
        activeTargetLanguage = try c.decodeIfPresent(String.self, forKey: .activeTargetLanguage)
        literalDictation = try c.decodeIfPresent(Bool.self, forKey: .literalDictation) ?? false
        allowCloudModels = try c.decodeIfPresent(Bool.self, forKey: .allowCloudModels) ?? false
    }

    static let starterLanguages = ["en", "zh-Hans", "ja", "es", "fr", "de", "ko", "pt"]
    static let catalog = starterLanguages + ["it", "nl", "sv", "pl", "tr", "vi", "id", "th", "uk", "ar", "hi"]

    init() {
        mode = .dictate
        style = .automatic
        targetLanguages = Self.starterLanguages
        activeTargetLanguage = "en"
    }

    func makeOptions(selectedText: String, contextBefore: String, contextAfter: String) -> CompositionOptions {
        CompositionOptions(
            mode: mode,
            targetLanguage: mode == .translate ? activeTargetLanguage : nil,
            selectedText: selectedText,
            contextBefore: String(contextBefore.suffix(CompositionCapture.contextLimit)),
            contextAfter: String(contextAfter.prefix(CompositionCapture.contextLimit)),
            style: style,
            vocabulary: [],
            literalDictation: literalDictation
        )
    }

    mutating func moveLanguage(from index: Int, to destination: Int) {
        guard targetLanguages.indices.contains(index), index != destination else { return }
        let item = targetLanguages.remove(at: index)
        let clamped = min(max(destination, 0), targetLanguages.count)
        targetLanguages.insert(item, at: clamped)
    }

    mutating func addLanguage(_ code: String) throws {
        if targetLanguages.contains(code) { return }
        try CompositionOptions(mode: .translate, targetLanguage: code).validate()
        targetLanguages.append(code)
        if activeTargetLanguage == nil { activeTargetLanguage = code }
    }

    mutating func removeLanguage(_ code: String) {
        guard targetLanguages.count > 1, let index = targetLanguages.firstIndex(of: code) else { return }
        targetLanguages.remove(at: index)
        if activeTargetLanguage == code { activeTargetLanguage = targetLanguages.first }
    }

    static func title(for code: String) -> String {
        VoxtL10n.shared.locale.localizedString(forIdentifier: code) ?? code
    }
}

struct CompositionGate: Codable, Equatable {
    var origin: FieldOrigin?
    var lastInsertedResultID: String?
    var blockedRequestIDs: [String] = []
    var latestInsertion: LatestInsertion?
    var pendingReEdit: PendingReEdit?
    var activeEditScope: EditScope?
    /// The language a pill chose for the request in flight; saved like the edit scope.
    var activeTargetLanguage: String?
    var editUndo: EditUndo?
    struct Handoff: Codable, Equatable {
        var requestID: String
        var startedAt: Date
        var detached = false
    }
    var handoff: Handoff?

    mutating func begin(_ field: CapturedField, requestID: String) {
        handoff = nil
        if let previous = origin?.requestID, previous != requestID {
            block(previous)
        }
        blockedRequestIDs.removeAll { $0 == requestID }
        // A new composition supersedes any pending re-edit of an earlier result.
        if pendingReEdit?.requestID != requestID { pendingReEdit = nil }
        origin = FieldOrigin(field: field, requestID: requestID)
        editUndo = nil
    }

    mutating func cancel() {
        handoff = nil
        activeTargetLanguage = nil
        if let requestID = origin?.requestID { block(requestID) }
    }

    /// A result that is not the operation this keyboard started must not be inserted on appear.
    /// The in-flight operation's own result is left unclaimed so hiding the keyboard can still deliver it.
    mutating func prepareForAppear(stateRequestID: String?, stateResultID: String?) {
        let ownsState = origin != nil && stateRequestID == origin?.requestID
        guard !ownsState, let stateResultID else { return }
        lastInsertedResultID = stateResultID
    }

    mutating func receive(current: CapturedField?, text: String, resultID: String, requestID: String?) -> InsertionDecision {
        guard lastInsertedResultID != resultID else { return .drop }
        guard let origin else {
            lastInsertedResultID = resultID
            return .drop
        }
        guard let requestID, requestID == origin.requestID, !blockedRequestIDs.contains(requestID) else {
            lastInsertedResultID = resultID
            return .drop
        }
        // Notes can report a new documentIdentifier for the same note mid-request. An
        // exact match of a known, non-empty context before the cursor plus
        // selection and context after identifies the field; nil or empty context does not.
        guard let current,
              current.matchesContext(origin.field),
              current.documentID == origin.field.documentID
                || (origin.field.contextBeforeAvailable == true && !origin.field.contextBefore.isEmpty) else {
            return .recoverFromHistory
        }
        lastInsertedResultID = resultID
        if let pending = pendingReEdit, pending.requestID == requestID {
            pendingReEdit = nil
            // The replacement deletes `oldText`; the text before the new run is
            // the pre-insertion context without it.
            let before = String(pending.field.contextBefore.dropLast(pending.oldText.count))
            latestInsertion = LatestInsertion(
                documentID: current.documentID,
                before: before,
                inserted: text,
                after: pending.field.contextAfter,
                resultID: resultID
            )
            editUndo = EditUndo(documentID: current.documentID, before: before, newText: text,
                                oldText: pending.oldText, after: pending.field.contextAfter)
            return .replace(oldText: pending.oldText, newText: text)
        }
        editUndo = activeEditScope == .selection && !origin.field.selectedText.isEmpty
            ? EditUndo(documentID: current.documentID, before: current.contextBefore, newText: text,
                       oldText: origin.field.selectedText, after: current.contextAfter)
            : nil
        // A selection is replaced in place; only a new result next to existing text gets a space.
        let separator = current.selectedText.isEmpty ? CompositionSpacing.separator(before: current.contextBefore, text: text) : ""
        latestInsertion = LatestInsertion(
            documentID: current.documentID,
            before: current.contextBefore + separator,
            inserted: text,
            after: current.contextAfter,
            resultID: resultID,
            separator: separator.isEmpty ? nil : separator
        )
        return .insert(separator + text)
    }

    private mutating func block(_ requestID: String) {
        if !blockedRequestIDs.contains(requestID) { blockedRequestIDs.append(requestID) }
        if blockedRequestIDs.count > 32 { blockedRequestIDs.removeFirst(blockedRequestIDs.count - 32) }
    }
}

struct CompositionStore {
    var defaults: UserDefaults

    private enum Key {
        static let preferences = "composition.preferences"
        static let gate = "composition.gate"
        static let submittedMode = "composition.submittedMode"
    }

    func loadPreferences() -> CompositionPreferences {
        guard let data = defaults.data(forKey: Key.preferences),
              let value = try? JSONDecoder().decode(CompositionPreferences.self, from: data) else {
            return CompositionPreferences()
        }
        return value
    }

    func save(_ preferences: CompositionPreferences) {
        defaults.set(try? JSONEncoder().encode(preferences), forKey: Key.preferences)
    }

    func loadGate() -> CompositionGate {
        guard let data = defaults.data(forKey: Key.gate),
              let value = try? JSONDecoder().decode(CompositionGate.self, from: data) else {
            return CompositionGate()
        }
        return value
    }

    func save(_ gate: CompositionGate) {
        defaults.set(try? JSONEncoder().encode(gate), forKey: Key.gate)
    }

    func noteSubmittedMode(_ mode: CompositionOptions.Mode) {
        defaults.set(mode.rawValue, forKey: Key.submittedMode)
    }

    func submittedMode() -> CompositionOptions.Mode? {
        defaults.string(forKey: Key.submittedMode).flatMap(CompositionOptions.Mode.init(rawValue:))
    }
}

struct KeyboardComposer {
    var preferences: CompositionPreferences {
        didSet { store.save(preferences) }
    }
    private(set) var gate: CompositionGate
    private var store: CompositionStore

    init(defaults: UserDefaults) {
        store = CompositionStore(defaults: defaults)
        preferences = store.loadPreferences()
        gate = store.loadGate()
    }

    mutating func moveLanguage(from index: Int, to destination: Int) {
        preferences.moveLanguage(from: index, to: destination)
    }

    mutating func addLanguage(_ code: String) throws {
        try preferences.addLanguage(code)
    }

    mutating func removeLanguage(_ code: String) {
        preferences.removeLanguage(code)
    }

    /// What the pen will act on in this field: the selection, else our last result
    /// while it is still at the cursor, else a new text written from a description.
    func penIntent(context: CompositionTextContext?) -> PenIntent? {
        guard let context else { return nil }
        if !CompositionCapture.capture(context).selectedText.isEmpty { return .editSelection }
        if canReEditLatestResult(context: context) { return .editLastResult }
        return .write
    }

    /// Starts one recording. No intent is plain dictation, whatever mode was saved
    /// before; an intent applies to this recording only and is never saved.
    mutating func begin(context: CompositionTextContext, intent: PenIntent?, requestID: String = UUID().uuidString) throws -> CompositionCommand {
        switch intent {
        case .editLastResult:
            return try beginReEdit(context: context, requestID: requestID)
        case .editSelection:
            guard !CompositionCapture.capture(context).selectedText.isEmpty else { throw CompositionEditScopeError.nothingToEdit }
            return try begin(context: context, mode: .edit, requestID: requestID)
        case .write:
            return try begin(context: context, mode: .write, requestID: requestID)
        case nil:
            return try begin(context: context, mode: .dictate, requestID: requestID)
        }
    }

    /// One recording translated into `language` (a pill on the keyboard).
    mutating func begin(context: CompositionTextContext, translateTo language: String, requestID: String = UUID().uuidString) throws -> CompositionCommand {
        try begin(context: context, mode: .translate, targetLanguage: language, requestID: requestID)
    }

    mutating func begin(context: CompositionTextContext, requestID: String = UUID().uuidString) throws -> CompositionCommand {
        let field = CompositionCapture.capture(context)
        if preferences.mode == .edit, field.selectedText.isEmpty {
            guard canReEditLatestResult(context: context) else { throw CompositionEditScopeError.nothingToEdit }
            return try beginReEdit(context: context, requestID: requestID)
        }
        return try begin(context: context, mode: preferences.mode, requestID: requestID)
    }

    private mutating func begin(context: CompositionTextContext, mode: CompositionOptions.Mode, targetLanguage: String? = nil,
                                requestID: String) throws -> CompositionCommand {
        let field = CompositionCapture.capture(context)
        var recording = preferences
        recording.mode = mode
        if let targetLanguage { recording.activeTargetLanguage = targetLanguage }
        var options = recording.makeOptions(
            selectedText: field.selectedText,
            contextBefore: field.contextBefore,
            contextAfter: field.contextAfter
        )
        options.vocabulary = PersonalDictionary(defaults: store.defaults).vocabulary
        options.languageVariants = store.defaults.stringArray(forKey: "language.variants") ?? []
        options.styleExamples = PersonalStyleStore(defaults: store.defaults).examples(for: options.mode, style: options.style)
        try options.validate()
        gate.begin(field, requestID: requestID)
        gate.activeEditScope = options.mode == .edit ? .selection : nil
        gate.activeTargetLanguage = options.mode == .translate ? options.targetLanguage : nil
        store.save(gate)
        store.noteSubmittedMode(options.mode)
        return CompositionCommand(kind: .start, options: options, requestID: requestID)
    }

    mutating func cancel() {
        gate.cancel()
        store.save(gate)
    }

    /// The scope of the in-flight Edit request, shown while listening.
    var editScope: EditScope? { gate.origin == nil ? nil : gate.activeEditScope }

    /// The language the request in flight translates into, for "Translating to …".
    var translatingTo: String? { gate.origin == nil ? nil : gate.activeTargetLanguage }

    func canUndoEdit(context: CompositionTextContext?) -> Bool {
        guard let context, let undo = gate.editUndo else { return false }
        let field = CompositionCapture.capture(context)
        return field.documentID == undo.documentID && field.selectedText.isEmpty
            && field.contextBefore == String((undo.before + undo.newText).suffix(CompositionCapture.contextLimit))
            && field.contextAfter == String(undo.after.prefix(CompositionCapture.contextLimit))
    }

    /// An Undo is offered after every insert while it is still exactly at the cursor.
    func canUndo(context: CompositionTextContext?) -> Bool {
        canUndoEdit(context: context) || canReEditLatestResult(context: context)
    }

    /// Undoes the latest insert once: an edit restores the text it replaced, a dictation
    /// removes itself and the space added before it.
    mutating func undo(context: CompositionTextContext?) -> InsertionDecision? {
        if canUndoEdit(context: context) { return undoEdit(context: context) }
        guard canReEditLatestResult(context: context), let last = gate.latestInsertion else { return nil }
        gate.latestInsertion = nil
        gate.editUndo = nil
        store.save(gate)
        return .replace(oldText: (last.separator ?? "") + last.inserted, newText: "")
    }

    /// Restores the text an edit replaced, once, and only while the edit is verified at the cursor.
    mutating func undoEdit(context: CompositionTextContext?) -> InsertionDecision? {
        guard canUndoEdit(context: context), let undo = gate.editUndo else { return nil }
        gate.editUndo = nil
        gate.latestInsertion = nil
        store.save(gate)
        return .replace(oldText: undo.newText, newText: undo.oldText)
    }

    mutating func persist() {
        store.save(preferences)
        store.save(gate)
    }

    mutating func prepareForHandoff(now: Date = Date()) {
        guard let origin = gate.origin else { return }
        gate.handoff = .init(requestID: origin.requestID, startedAt: now)
        store.save(gate)
    }

    /// iOS 26.3: returning from the microphone handoff can
    /// replace documentIdentifier even for the same unchanged field. Authorize one
    /// return only; ordinary field switches still require the original identifier.
    mutating func resumeAfterHandoff(current: CompositionTextContext?, stateRequestID: String?, now: Date = Date()) {
        guard let handoff = gate.handoff, handoff.detached else { return }
        guard now.timeIntervalSince(handoff.startedAt) >= 0,
              now.timeIntervalSince(handoff.startedAt) <= 60 else {
            gate.handoff = nil
            store.save(gate)
            return
        }
        guard let current, let origin = gate.origin,
              handoff.requestID == origin.requestID,
              stateRequestID == origin.requestID else { return }
        gate.handoff = nil
        let field = CompositionCapture.capture(current)
        if !gate.blockedRequestIDs.contains(origin.requestID),
           origin.field.contextBeforeAvailable != nil,
           origin.field.contextAfterAvailable != nil,
           field.matchesContext(origin.field) {
            // Known-empty fields and two unavailable contexts each match within this one return.
            gate.origin?.field.documentID = field.documentID
            gate.pendingReEdit?.field.documentID = field.documentID
        }
        store.save(gate)
    }

    mutating func prepareForTextInputDetach(isComposingText: Bool, hasPendingTextWrite: Bool) {
        if isComposingText || hasPendingTextWrite { cancel() }
        else if gate.handoff != nil {
            gate.handoff?.detached = true
            store.save(gate)
        }
    }

    mutating func prepareForAppear(stateRequestID: String?, stateResultID: String?) {
        gate.prepareForAppear(stateRequestID: stateRequestID, stateResultID: stateResultID)
        store.save(gate)
    }

    mutating func receive(current: CompositionTextContext?, text: String, resultID: String, requestID: String?, isComposingText: Bool = false, hasPendingTextWrite: Bool = false) -> InsertionDecision {
        // A host can omit marked text from its surrounding context. An unchanged
        // proxy snapshot must not authorize overwriting a live IME composition.
        let field = (isComposingText || hasPendingTextWrite) ? nil : current.map(CompositionCapture.capture)
        let decision = gate.receive(current: field, text: text, resultID: resultID, requestID: requestID)
        store.save(gate)
        return decision
    }

    /// Whether the just-inserted result is still exactly where the keyboard left it,
    /// so "Edit latest result" can run without touching the field first.
    func canReEditLatestResult(context: CompositionTextContext?) -> Bool {
        guard let context, let last = gate.latestInsertion, !last.inserted.isEmpty,
              last.inserted.count <= CompositionCapture.contextLimit else { return false }
        let field = CompositionCapture.capture(context)
        return field.documentID == last.documentID && field.selectedText.isEmpty
            && field.contextBefore == last.before + last.inserted
            && field.contextAfter == last.after
    }

    /// Starts an edit-mode composition over the just-inserted result. The original
    /// text stays in the field; only a strict re-check at result time replaces it.
    mutating func beginReEdit(context: CompositionTextContext, requestID: String = UUID().uuidString) throws -> CompositionCommand {
        guard canReEditLatestResult(context: context), let last = gate.latestInsertion else {
            throw CompositionReEditError.unavailable
        }
        let field = CompositionCapture.capture(context)
        var options = preferences.makeOptions(selectedText: last.inserted, contextBefore: last.before, contextAfter: last.after)
        options.mode = .edit
        options.targetLanguage = nil
        options.vocabulary = PersonalDictionary(defaults: store.defaults).vocabulary
        options.languageVariants = store.defaults.stringArray(forKey: "language.variants") ?? []
        options.styleExamples = PersonalStyleStore(defaults: store.defaults).examples(for: .edit, style: options.style)
        try options.validate()
        gate.begin(field, requestID: requestID)
        gate.pendingReEdit = PendingReEdit(field: field, oldText: last.inserted, requestID: requestID)
        gate.activeEditScope = .lastResult
        gate.activeTargetLanguage = nil
        store.save(gate)
        store.noteSubmittedMode(.edit)
        return CompositionCommand(kind: .start, options: options, requestID: requestID)
    }
}
