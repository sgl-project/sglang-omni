import AVFoundation
import CoreMedia
import XCTest
@testable import VoxtMobile
import VoxtMobileKit

private struct FixtureField: CompositionTextContext {
    var documentIdentifier: UUID
    var selectedText: String?
    var documentContextBeforeInput: String?
    var documentContextAfterInput: String?
}

@MainActor
private final class OptionRecordingTranscriber: Transcribing {
    var options: CompositionOptions?

    func transcribe(wav: Data) async throws -> (text: String, rawText: String) {
        ("plain", "plain")
    }

    func transcribe(wav: Data, options: CompositionOptions) async throws -> (text: String, rawText: String) {
        self.options = options
        return ("edited", options.selectedText)
    }
}

final class CompositionSessionTests: XCTestCase {
    func testCloudConsentDefaultsOffAndPersistsOnlyExplicitChoice() throws {
        var prefs = CompositionPreferences()
        XCTAssertFalse(prefs.allowCloudModels)
        var old = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(prefs)) as? [String: Any])
        old.removeValue(forKey: "allowCloudModels")
        XCTAssertFalse(try JSONDecoder().decode(CompositionPreferences.self, from: JSONSerialization.data(withJSONObject: old)).allowCloudModels)
        prefs.allowCloudModels = true
        XCTAssertTrue(try JSONDecoder().decode(CompositionPreferences.self, from: JSONEncoder().encode(prefs)).allowCloudModels)
    }

    func testLiteralPreferenceDefaultsOffMigratesAndTravelsWithRecording() throws {
        var prefs = CompositionPreferences()
        XCTAssertFalse(prefs.literalDictation)
        var old = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(prefs)) as? [String: Any])
        old.removeValue(forKey: "literalDictation")
        old["mode"] = "translate"
        let restored = try JSONDecoder().decode(CompositionPreferences.self, from: JSONSerialization.data(withJSONObject: old))
        XCTAssertEqual(restored.mode, .translate)
        XCTAssertFalse(restored.literalDictation)
        prefs.literalDictation = true
        let options = prefs.makeOptions(selectedText: "", contextBefore: "", contextAfter: "")
        XCTAssertTrue(options.literalDictation)
        XCTAssertTrue(try JSONDecoder().decode(CompositionPreferences.self, from: JSONEncoder().encode(prefs)).literalDictation)
        prefs.literalDictation = false
        XCTAssertTrue(options.literalDictation, "in-flight options keep their captured preference")
    }


    private var suiteName: String!
    private var defaults: UserDefaults!

    override func setUpWithError() throws {
        suiteName = "voxt-composition-tests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suiteName)
    }

    func testCaptureUsesEmptyPromptSelectionAndTruncatesContextAroundTheCursor() {
        let captured = CompositionCapture.capture(FixtureField(
            documentIdentifier: UUID(),
            selectedText: nil,
            documentContextBeforeInput: String(repeating: "a", count: 2_500),
            documentContextAfterInput: String(repeating: "b", count: 2_500)
        ))
        XCTAssertEqual(captured.selectedText, "")
        XCTAssertEqual(captured.contextBefore, String(repeating: "a", count: 2_000))
        XCTAssertEqual(captured.contextAfter, String(repeating: "b", count: 2_000))
        XCTAssertEqual(captured.contextBefore.last, "a")
    }

    func testBeginStoresSelectionAndSurvivesANewKeyboardInstance() throws {
        let document = UUID()
        var composer = KeyboardComposer(defaults: defaults)
        composer.preferences.mode = .edit
        composer.preferences.style = .formal
        let command = try composer.begin(
            context: FixtureField(
                documentIdentifier: document,
                selectedText: "Hello",
                documentContextBeforeInput: "Dear ",
                documentContextAfterInput: " team"
            ),
            requestID: "req-1"
        )
        XCTAssertEqual(command.kind, .start)
        XCTAssertEqual(command.requestID, "req-1")
        XCTAssertEqual(command.options?.selectedText, "Hello")
        XCTAssertEqual(command.options?.contextBefore, "Dear ")
        XCTAssertEqual(command.options?.mode, .edit)
        XCTAssertEqual(command.options?.style, .formal)

        let restored = KeyboardComposer(defaults: defaults)
        XCTAssertEqual(restored.gate.origin?.requestID, "req-1")
        XCTAssertEqual(restored.gate.origin?.field.documentID, document)
        XCTAssertEqual(restored.gate.origin?.field.selectedText, "Hello")
    }

    func testKeyboardIncludesOnlyEnabledAutomaticPersonalStyle() throws {
        let styles = PersonalStyleStore(defaults: defaults)
        styles.isEnabled = true
        let example = PersonalStyleExample(before: "Thanks, I'll send the file.", after: "Cheers — I'll send the file.")
        try styles.keep(example)
        var composer = KeyboardComposer(defaults: defaults)
        let field = FixtureField(documentIdentifier: UUID(), selectedText: "Text", documentContextBeforeInput: "", documentContextAfterInput: "")
        XCTAssertEqual(try composer.begin(context: field).options?.styleExamples, [example])
        composer.preferences.style = .formal
        XCTAssertEqual(try composer.begin(context: field).options?.styleExamples, [])
        composer.preferences.style = .automatic
        composer.preferences.mode = .translate
        XCTAssertEqual(try composer.begin(context: field).options?.styleExamples, [])
        composer.preferences.mode = .dictate
        styles.isEnabled = false
        XCTAssertEqual(try composer.begin(context: field).options?.styleExamples, [])
    }

    func testOwnResultInsertsWhenTheFieldIsUnchangedAfterDetach() throws {
        let document = UUID()
        let field = FixtureField(documentIdentifier: document, selectedText: "Hello", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team")
        var composer = KeyboardComposer(defaults: defaults)
        composer.preferences.mode = .edit
        _ = try composer.begin(context: field, requestID: "req-1")

        var returned = KeyboardComposer(defaults: defaults)
        returned.prepareForAppear(stateRequestID: "req-1", stateResultID: "result-new")
        let decision = returned.receive(current: field, text: "Hi", resultID: "result-new", requestID: "req-1")
        XCTAssertEqual(decision, .insert("Hi"))
        XCTAssertEqual(KeyboardComposer(defaults: defaults).gate.lastInsertedResultID, "result-new")
    }

    func testActiveCompositionPreventsInsertionEvenWhenProxyOmitsMarkedText() throws {
        let field = FixtureField(documentIdentifier: UUID(), selectedText: "", documentContextBeforeInput: "Before ", documentContextAfterInput: " after")
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: field, requestID: "req-marked")
        XCTAssertEqual(composer.receive(current: field, text: "voice", resultID: "r-marked", requestID: "req-marked", isComposingText: true), .recoverFromHistory)
        XCTAssertEqual(composer.receive(current: field, text: "voice", resultID: "r-marked", requestID: "req-marked", isComposingText: false), .insert("voice"))
    }

    func testPendingTextWritePreventsVoiceInsertionAfterReadingWasCleared() throws {
        let field = FixtureField(documentIdentifier: UUID(), selectedText: "", documentContextBeforeInput: "Before ", documentContextAfterInput: " after")
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: field, requestID: "req-pending")
        XCTAssertEqual(composer.receive(current: field, text: "voice", resultID: "r-pending", requestID: "req-pending", isComposingText: false, hasPendingTextWrite: true), .recoverFromHistory)
    }

    func testDetachingDuringTextWriteCancelsLateVoiceButVoiceOnlyDetachKeepsIt() throws {
        let field = FixtureField(documentIdentifier: UUID(), selectedText: "", documentContextBeforeInput: "Before ", documentContextAfterInput: " after")
        for (reading, pending, expected) in [(false, true, InsertionDecision.drop), (true, false, .drop), (false, false, .insert("voice"))] {
            var composer = KeyboardComposer(defaults: defaults)
            let request = UUID().uuidString
            _ = try composer.begin(context: field, requestID: request)
            composer.prepareForTextInputDetach(isComposingText: reading, hasPendingTextWrite: pending)
            var restored = KeyboardComposer(defaults: defaults)
            XCTAssertEqual(restored.receive(current: field, text: "voice", resultID: UUID().uuidString, requestID: request), expected)
        }
    }

    func testExplicitHandoffRebindsOnlyOneMatchingReturnAcrossKeyboardRecreation() throws {
        let time = Date(timeIntervalSince1970: 1000)
        let original = FixtureField(documentIdentifier: UUID(), selectedText: "Hello", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team")
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: original, requestID: "handoff")
        composer.prepareForHandoff(now: time)
        composer.prepareForTextInputDetach(isComposingText: false, hasPendingTextWrite: false)
        var returned = KeyboardComposer(defaults: defaults)
        var field = original
        field.documentIdentifier = UUID()
        returned.resumeAfterHandoff(current: field, stateRequestID: "handoff", now: time.addingTimeInterval(2))
        XCTAssertEqual(returned.receive(current: field, text: "Hi", resultID: "result", requestID: "handoff"), .insert("Hi"))
        field.documentIdentifier = UUID()
        returned.resumeAfterHandoff(current: field, stateRequestID: "handoff", now: time.addingTimeInterval(3))
        XCTAssertNotEqual(returned.gate.origin?.field.documentID, field.documentIdentifier, "A consumed handoff must not authorize a second field")
    }

    func testHandoffCannotAuthorizeChangedContextExpiredOrCancelledRequest() throws {
        for variant in ["selection", "before", "after", "expired", "cancelled", "request"] {
            let time = Date(timeIntervalSince1970: 1000)
            let original = FixtureField(documentIdentifier: UUID(), selectedText: "Hello", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team")
            var composer = KeyboardComposer(defaults: defaults)
            _ = try composer.begin(context: original, requestID: variant)
            composer.prepareForHandoff(now: time)
            composer.prepareForTextInputDetach(isComposingText: false, hasPendingTextWrite: false)
            if variant == "cancelled" { composer.cancel() }
            var field = original
            field.documentIdentifier = UUID()
            if variant == "selection" { field.selectedText = "Changed" }
            if variant == "before" { field.documentContextBeforeInput = "Changed" }
            if variant == "after" { field.documentContextAfterInput = "Changed" }
            composer.resumeAfterHandoff(current: field, stateRequestID: variant == "request" ? "foreign" : variant,
                now: time.addingTimeInterval(variant == "expired" ? 61 : 2))
            XCTAssertNotEqual(composer.gate.origin?.field.documentID, field.documentIdentifier, variant)
            // An expired or foreign handoff does not rebind, but the unchanged non-empty
            // "Dear " context still identifies the field (Notes can change the identifier mid-request).
            let identifiedByContext = variant == "expired" || variant == "request"
            XCTAssertEqual(composer.receive(current: field, text: "Hi", resultID: "result-" + variant, requestID: variant) == .insert("Hi"),
                           identifiedByContext, variant)
        }
    }

    func testHandoffAllowsDifferentEmptyFieldOnlyWithinSixtySecondReturnWindow() throws {
        let time = Date(timeIntervalSince1970: 1000)
        let original = FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: "", documentContextAfterInput: "")
        for elapsed in [59.0, 61.0] {
            var composer = KeyboardComposer(defaults: defaults)
            _ = try composer.begin(context: original, requestID: "empty")
            composer.prepareForHandoff(now: time)
            composer.prepareForTextInputDetach(isComposingText: false, hasPendingTextWrite: false)
            var otherEmptyField = original
            otherEmptyField.documentIdentifier = UUID()
            composer.resumeAfterHandoff(current: otherEmptyField, stateRequestID: "empty", now: time.addingTimeInterval(elapsed))
            XCTAssertEqual(composer.receive(current: otherEmptyField, text: "Hello", resultID: "empty-result-\(elapsed)", requestID: "empty"),
                           elapsed < 60 ? .insert("Hello") : .recoverFromHistory)
        }
    }

    func testHandoffDistinguishesUnavailableContextFromKnownEmptyContext() throws {
        let time = Date(timeIntervalSince1970: 1000)
        let cases: [(String?, String?, InsertionDecision)] = [
            ("Existing text", nil, .recoverFromHistory),
            (nil, nil, .insert("Hello")),
            ("Existing text", "Different text", .recoverFromHistory),
            ("", nil, .recoverFromHistory),
            (nil, "", .recoverFromHistory),
            ("", "", .insert("Hello")),
            ("Existing text", "Existing text", .insert(" Hello"))
        ]
        for (index, item) in cases.enumerated() {
            let (before, returnedBefore, expected) = item
            var composer = KeyboardComposer(defaults: defaults)
            let requestID = "context-\(index)"
            let original = FixtureField(documentIdentifier: UUID(), selectedText: nil,
                                        documentContextBeforeInput: before, documentContextAfterInput: nil)
            _ = try composer.begin(context: original, requestID: requestID)
            composer.prepareForHandoff(now: time)
            composer.prepareForTextInputDetach(isComposingText: false, hasPendingTextWrite: false)
            // Reload the persisted gate, as a new extension instance does.
            composer = KeyboardComposer(defaults: defaults)
            let returned = FixtureField(documentIdentifier: UUID(), selectedText: nil,
                                        documentContextBeforeInput: returnedBefore, documentContextAfterInput: nil)
            composer.resumeAfterHandoff(current: returned, stateRequestID: requestID, now: time.addingTimeInterval(2))
            XCTAssertEqual(composer.receive(current: returned, text: "Hello", resultID: requestID, requestID: requestID), expected, "case \(index)")
        }
    }

    func testChangedDocumentSelectionOrContextDoesNotInsert() throws {
        let document = UUID()
        let field = FixtureField(documentIdentifier: document, selectedText: "Hello", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team")
        var composer = KeyboardComposer(defaults: defaults)
        composer.preferences.mode = .edit
        _ = try composer.begin(context: field, requestID: "req-1")

        let otherDocumentEmptyBefore = FixtureField(documentIdentifier: UUID(), selectedText: "Hello", documentContextBeforeInput: "", documentContextAfterInput: " team")
        var emptyBefore = KeyboardComposer(defaults: defaults)
        _ = try emptyBefore.begin(context: FixtureField(documentIdentifier: document, selectedText: "Hello", documentContextBeforeInput: "", documentContextAfterInput: " team"), requestID: "req-empty")
        XCTAssertEqual(emptyBefore.receive(current: otherDocumentEmptyBefore, text: "Hi", resultID: "r0", requestID: "req-empty"), .recoverFromHistory,
                       "a new identifier with an empty context before the cursor is not evidence of the same field")
        _ = try composer.begin(context: field, requestID: "req-1")

        var selectionChanged = KeyboardComposer(defaults: defaults)
        let otherSelection = FixtureField(documentIdentifier: document, selectedText: "Hello!", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team")
        XCTAssertEqual(selectionChanged.receive(current: otherSelection, text: "Hi", resultID: "r2", requestID: "req-1"), .recoverFromHistory)

        var contextChanged = KeyboardComposer(defaults: defaults)
        let otherContext = FixtureField(documentIdentifier: document, selectedText: "Hello", documentContextBeforeInput: "Hi ", documentContextAfterInput: " team")
        XCTAssertEqual(contextChanged.receive(current: otherContext, text: "Hi", resultID: "r3", requestID: "req-1"), .recoverFromHistory)
    }

    /// Notes can report a new documentIdentifier for the same note during a long request while
    /// the keyboard stays up; without the context match the result would fall back to Copy.
    func testSameFieldWithANewDocumentIdentifierMidRequestInsertsOnlyWhenTheKnownContextMatches() throws {
        let first = "嗯，那个，我先随便写一点。我觉得这个，就先这样吧。"
        let note = FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: first, documentContextAfterInput: nil)
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: note, requestID: "notes")
        var renamed = note
        renamed.documentIdentifier = UUID()
        XCTAssertEqual(composer.receive(current: renamed, text: "好，我再补一句。", resultID: "notes-result", requestID: "notes"), .insert("好，我再补一句。"))
        XCTAssertEqual(composer.gate.latestInsertion?.documentID, renamed.documentIdentifier)

        // The same rule holds when the extension is silently recreated and reloads the gate.
        _ = try composer.begin(context: note, requestID: "notes-recreated")
        var recreated = KeyboardComposer(defaults: defaults)
        XCTAssertEqual(recreated.receive(current: renamed, text: "再补一句。", resultID: "notes-result-2", requestID: "notes-recreated"), .insert("再补一句。"))

        for (index, changed) in [
            FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: first + "x", documentContextAfterInput: nil),
            FixtureField(documentIdentifier: UUID(), selectedText: "我", documentContextBeforeInput: first, documentContextAfterInput: nil),
            FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: first, documentContextAfterInput: "after"),
        ].enumerated() {
            var gate = KeyboardComposer(defaults: defaults)
            _ = try gate.begin(context: note, requestID: "changed-\(index)")
            XCTAssertEqual(gate.receive(current: changed, text: "late", resultID: "changed-result-\(index)", requestID: "changed-\(index)"), .recoverFromHistory, "case \(index)")
        }
        for (index, before) in [nil, ""].enumerated() {
            let unknown = FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: before, documentContextAfterInput: nil)
            var gate = KeyboardComposer(defaults: defaults)
            _ = try gate.begin(context: unknown, requestID: "unknown-\(index)")
            var other = unknown
            other.documentIdentifier = UUID()
            XCTAssertEqual(gate.receive(current: other, text: "late", resultID: "unknown-result-\(index)", requestID: "unknown-\(index)"), .recoverFromHistory,
                           "nil or empty context before the cursor cannot identify the field without a handoff")
        }
    }

    func testCancelDropsALateResultAndAForeignRequestNeverInserts() throws {
        let field = FixtureField(documentIdentifier: UUID(), selectedText: "", documentContextBeforeInput: nil, documentContextAfterInput: nil)
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: field, requestID: "req-1")
        composer.cancel()
        XCTAssertEqual(composer.receive(current: field, text: "late", resultID: "r-late", requestID: "req-1"), .drop)
        XCTAssertEqual(composer.receive(current: field, text: "other", resultID: "r-other", requestID: "req-2"), .drop)
    }

    func testAppOnlyResultCannotInsertWithoutKeyboardOrigin() {
        var composer = KeyboardComposer(defaults: defaults)
        let field = FixtureField(documentIdentifier: UUID(), selectedText: "", documentContextBeforeInput: nil, documentContextAfterInput: nil)
        XCTAssertEqual(composer.receive(current: field, text: "app result", resultID: "app-result", requestID: nil), .drop)
    }

    @MainActor
    func testDictionaryTermsReachAppAndKeyboardRequests() async throws {
        let dictionary = PersonalDictionary(defaults: defaults)
        try dictionary.add("SGLang")
        var composer = KeyboardComposer(defaults: defaults)
        let command = try composer.begin(context: FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: nil, documentContextAfterInput: nil))
        XCTAssertEqual(command.options?.vocabulary, ["SGLang"])
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        let transcriber = OptionRecordingTranscriber()
        model.controller.setTranscriber(transcriber)
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(transcriber.options?.vocabulary, ["SGLang"])
    }

    @MainActor
    func testLearnedWordReachesBothRequestsAndUndoRemovesIt() async throws {
        let learning = CorrectionLearningStore(defaults: defaults)
        learning.isEnabled = true
        let record = try XCTUnwrap(learning.learn(.init(original: "Quen", replacement: "Qwen3")))
        var composer = KeyboardComposer(defaults: defaults)
        let field = FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: nil, documentContextAfterInput: nil)
        XCTAssertEqual(try composer.begin(context: field).options?.vocabulary, ["Qwen3"])
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        let transcriber = OptionRecordingTranscriber()
        model.controller.setTranscriber(transcriber)
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(transcriber.options?.vocabulary, ["Qwen3"])
        learning.undo(record.id)
        XCTAssertEqual(try composer.begin(context: field).options?.vocabulary, [])
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(transcriber.options?.vocabulary, [])
    }

    func testOwnResultWaitsForTheFieldToReturnAfterDetach() throws {
        let field = FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: "Hello", documentContextAfterInput: nil)
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: field, requestID: "pending")
        XCTAssertEqual(composer.receive(current: nil, text: "world", resultID: "pending-result", requestID: "pending"), .recoverFromHistory)
        XCTAssertEqual(composer.receive(current: field, text: "world", resultID: "pending-result", requestID: "pending"), .insert(" world"),
                       "a result next to text gets a space")
    }

    @MainActor
    func testRegionalPreferencesReachBothAppAndKeyboard() async throws {
        defaults.set(["en-GB", "zh-Hant"], forKey: "language.variants")
        var composer = KeyboardComposer(defaults: defaults)
        let command = try composer.begin(context: FixtureField(documentIdentifier: UUID(), selectedText: nil, documentContextBeforeInput: nil, documentContextAfterInput: nil))
        XCTAssertEqual(command.options?.languageVariants, ["en-GB", "zh-Hant"])
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        let transcriber = OptionRecordingTranscriber()
        model.controller.setTranscriber(transcriber)
        await model.toggleRecording()
        await model.toggleRecording()
        XCTAssertEqual(transcriber.options?.languageVariants, ["en-GB", "zh-Hant"])
    }

    func testReorderPersistsTargetLanguages() {
        var composer = KeyboardComposer(defaults: defaults)
        composer.moveLanguage(from: 0, to: 1)
        let restored = KeyboardComposer(defaults: defaults)
        XCTAssertEqual(restored.preferences.targetLanguages[0], "zh-Hans")
        XCTAssertEqual(restored.preferences.targetLanguages[1], "en")
        XCTAssertThrowsError(try composer.addLanguage("banana"))
    }
}

@MainActor
final class ColdLaunchCompositionTests: XCTestCase {
    private var suiteName: String!

    override func setUpWithError() throws {
        suiteName = "voxt-cold-launch.\(UUID().uuidString)"
    }

    override func tearDownWithError() throws {
        UserDefaults().removePersistentDomain(forName: suiteName)
    }

    func testColdLaunchReadsStartOptionsAfterControllerMarksTheCommandHandled() async throws {
        let defaults = UserDefaults(suiteName: suiteName)!
        let channel = DictationChannel(defaults: defaults, notifier: .none)
        let options = CompositionOptions(
            mode: .edit,
            selectedText: "Hello",
            contextBefore: "Dear ",
            contextAfter: " team",
            style: .formal
        )
        channel.send(.start, options: options, requestID: "req-cold")

        let capture = ScriptedCapturePublic()
        let model = AppModel(injectedCapture: capture, injectedDefaults: defaults)
        await model.controller.handlePendingCommand()
        XCTAssertEqual(model.phase, .idle, "init already marked the command handled")

        let transcriber = OptionRecordingTranscriber()
        model.controller.setTranscriber(transcriber)
        await model.startFromKeyboard()
        XCTAssertEqual(model.phase, .recording)
        await model.controller.stop()

        XCTAssertEqual(transcriber.options?.selectedText, "Hello")
        XCTAssertEqual(transcriber.options?.contextBefore, "Dear ")
        XCTAssertEqual(transcriber.options?.contextAfter, " team")
        XCTAssertEqual(transcriber.options?.mode, .edit)
        XCTAssertEqual(transcriber.options?.style, .formal)
        XCTAssertEqual(DictationChannel(defaults: defaults, notifier: .none).state.requestID, "req-cold")
    }

    func testEditWithoutSelectionDoesNotStart() async {
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: UserDefaults(suiteName: suiteName))
        model.setMode(.edit)
        await model.toggleRecording()
        XCTAssertEqual(model.phase, .idle)
        XCTAssertEqual(model.errorMessage, "Select the text you want to edit first.")
    }

    func testSessionLinkWithoutPendingKeyboardStartDoesNotRecord() async {
        let defaults = UserDefaults(suiteName: suiteName)!
        let channel = DictationChannel(defaults: defaults, notifier: .none)
        channel.send(.cancel, requestID: "cancelled")
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults)
        await model.startFromKeyboard()
        XCTAssertEqual(model.phase, .idle)
        XCTAssertFalse(model.sessionOpen)
    }
}

/// Visible to this file; the attribution tests keep their own scripted capture.
@MainActor
final class ScriptedCapturePublic: AudioCapturing {
    // A 0.3 s tone: silence is rejected before upload by the no-speech gate.
    var samples: [Float] = (0..<4801).map { 0.2 * Float(sin(Double($0) * 2 * .pi * 220 / 16_000)) }
    func startSession() throws {}
    func endSession() {}
    func beginCapture() {}
    func endCapture() -> [Float] { samples }
}

/// One-tap re-edit of the keyboard's just-inserted result: the original text
/// stays in the field; a strict re-check replaces it only when the new result
/// arrives and nothing has moved.
final class ReEditLatestResultTests: XCTestCase {
    private var suiteName: String!
    private var defaults: UserDefaults!

    override func setUpWithError() throws {
        suiteName = "voxt-reedit-tests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suiteName)
    }

    private let document = UUID()

    /// Field as the proxy sees it BEFORE the keyboard inserts: "Dear | team".
    private func emptyField() -> FixtureField {
        FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team")
    }

    /// Field as the proxy sees it right AFTER inserting "Hello": cursor behind
    /// the inserted run, no selection.
    private func insertedField() -> FixtureField {
        FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear Hello", documentContextAfterInput: " team")
    }

    private func composerWithInsertedResult() throws -> KeyboardComposer {
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: emptyField(), requestID: "req-1")
        _ = composer.receive(current: emptyField(), text: "Hello", resultID: "result-1", requestID: "req-1")
        return composer
    }

    // MARK: Spacing between consecutive results ("Friday.Please send" without it)

    func testSeparatorBetweenTheFieldAndANewResult() {
        let cases: [(String, String, String)] = [
            ("Please send Alice fifty dollars by Friday.", "Please send", " "),
            ("Hello", "world", " "),
            ("Hello ", "world", ""),
            ("Hello\n", "world", ""),
            ("", "Hello", ""),
            ("Hello", ", world", ""),
            ("Hello", ". Next", ""),
            ("(", "note", ""),
            ("你好。", "我们明天见。", ""),
            ("すみません。", "いくらですか？", ""),
            ("你好", "OK", ""),
            ("OK", "你好", ""),
            ("Done.", "다음 문장", " "),
        ]
        for (before, text, expected) in cases {
            XCTAssertEqual(CompositionSpacing.separator(before: before, text: text), expected, "\(before)|\(text)")
        }
    }

    func testConsecutiveDictationIsSpacedAndStillReEditable() throws {
        let first = "Please send Alice fifty dollars by Friday."
        let field = FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: first, documentContextAfterInput: "")
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: field, requestID: "spaced")
        XCTAssertEqual(composer.receive(current: field, text: "Thanks.", resultID: "spaced-r", requestID: "spaced"), .insert(" Thanks."))
        XCTAssertEqual(composer.gate.latestInsertion?.before, first + " ")
        XCTAssertEqual(composer.gate.latestInsertion?.inserted, "Thanks.")
        let after = FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: first + " Thanks.", documentContextAfterInput: "")
        XCTAssertTrue(composer.canReEditLatestResult(context: after))
        // A selection edit replaces in place and never adds a space.
        var edit = KeyboardComposer(defaults: defaults)
        edit.preferences.mode = .edit
        let selected = FixtureField(documentIdentifier: document, selectedText: "Hello", documentContextBeforeInput: "Dear", documentContextAfterInput: "")
        _ = try edit.begin(context: selected, requestID: "spaced-edit")
        XCTAssertEqual(edit.receive(current: selected, text: "Hi", resultID: "spaced-edit-r", requestID: "spaced-edit"), .insert("Hi"))
    }

    // MARK: Edit with no selection

    func testEditWithoutSelectionEditsTheLastResultOnlyWhileItIsAtTheCursor() throws {
        var composer = try composerWithInsertedResult()
        composer.preferences.mode = .edit
        let command = try composer.begin(context: insertedField(), requestID: "edit-last")
        XCTAssertEqual(command.options?.mode, .edit)
        XCTAssertEqual(command.options?.selectedText, "Hello")
        XCTAssertEqual(command.options?.contextBefore, "Dear ")
        XCTAssertEqual(composer.gate.pendingReEdit?.oldText, "Hello")
        XCTAssertEqual(composer.editScope, .lastResult)
    }

    func testEditWithoutSelectionOrVerifiedResultLeavesTheFieldAlone() throws {
        let emptySuite = suiteName + ".no-scope"
        let emptyDefaults = UserDefaults(suiteName: emptySuite)!
        defer { emptyDefaults.removePersistentDomain(forName: emptySuite) }
        for (label, makeComposer, field) in [
            ("nothing inserted", { KeyboardComposer(defaults: emptyDefaults) },
             FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Some text", documentContextAfterInput: "")),
            ("typed after the result", { try! self.composerWithInsertedResult() },
             FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear Hello!", documentContextAfterInput: " team")),
        ] as [(String, () -> KeyboardComposer, FixtureField)] {
            var composer = makeComposer()
            composer.preferences.mode = .edit
            let originBefore = composer.gate.origin
            XCTAssertThrowsError(try composer.begin(context: field, requestID: "no-scope")) { error in
                XCTAssertEqual(error as? CompositionEditScopeError, .nothingToEdit, label)
                XCTAssertEqual((error as? LocalizedError)?.errorDescription, "Select text to edit, or dictate to add", label)
            }
            XCTAssertEqual(composer.gate.origin, originBefore, "\(label): no request starts")
            XCTAssertNil(composer.gate.pendingReEdit, label)
        }
    }

    func testEditScopeNamesSelectionWhileListening() throws {
        var composer = KeyboardComposer(defaults: defaults)
        composer.preferences.mode = .edit
        _ = try composer.begin(context: FixtureField(documentIdentifier: document, selectedText: "Hello", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team"))
        XCTAssertEqual(composer.editScope, .selection)
        composer.preferences.mode = .dictate
        _ = try composer.begin(context: emptyField())
        XCTAssertNil(composer.editScope)
    }

    func testOneUndoRestoresTheEditedTextOnlyWhileVerifiedAtTheCursor() throws {
        // Re-edit of the last result: "Hello" -> "Hi".
        var composer = try composerWithInsertedResult()
        let command = try composer.beginReEdit(context: insertedField(), requestID: "undo-reedit")
        XCTAssertEqual(composer.receive(current: insertedField(), text: "Hi", resultID: "undo-r1", requestID: command.requestID),
                       .replace(oldText: "Hello", newText: "Hi"))
        let edited = FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear Hi", documentContextAfterInput: " team")
        XCTAssertTrue(composer.canUndoEdit(context: edited))
        XCTAssertFalse(composer.canUndoEdit(context: FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear Hi!", documentContextAfterInput: " team")), "typed after the edit")
        XCTAssertEqual(composer.undoEdit(context: edited), .replace(oldText: "Hi", newText: "Hello"))
        XCTAssertNil(composer.undoEdit(context: edited), "one undo only")

        // Selection edit: "Hello" selected -> "Hi" replaces it.
        var selection = KeyboardComposer(defaults: defaults)
        selection.preferences.mode = .edit
        let selected = FixtureField(documentIdentifier: document, selectedText: "Hello", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team")
        let start = try selection.begin(context: selected, requestID: "undo-selection")
        XCTAssertEqual(selection.receive(current: selected, text: "Hi", resultID: "undo-r2", requestID: start.requestID), .insert("Hi"))
        var reloaded = KeyboardComposer(defaults: defaults)
        XCTAssertEqual(reloaded.undoEdit(context: edited), .replace(oldText: "Hi", newText: "Hello"),
                       "undo survives a new keyboard instance")

        // A dictation is not an edit and offers no undo.
        var dictation = KeyboardComposer(defaults: defaults)
        dictation.preferences.mode = .dictate
        _ = try dictation.begin(context: emptyField(), requestID: "undo-dictate")
        _ = dictation.receive(current: emptyField(), text: "Hello", resultID: "undo-r3", requestID: "undo-dictate")
        XCTAssertFalse(dictation.canUndoEdit(context: insertedField()))
    }

    func testVoiceInsertionRecordsTheLatestResultForLaterReEdit() throws {
        let composer = try composerWithInsertedResult()

        let restored = KeyboardComposer(defaults: defaults)
        XCTAssertEqual(restored.gate.latestInsertion,
                       LatestInsertion(documentID: document, before: "Dear ", inserted: "Hello", after: " team", resultID: "result-1"))
    }

    func testReEditEligibilityRequiresExactUnchangedPostInsertionState() throws {
        let composer = try composerWithInsertedResult()
        XCTAssertTrue(composer.canReEditLatestResult(context: insertedField()))
        XCTAssertFalse(composer.canReEditLatestResult(context: nil), "no text field")
        XCTAssertFalse(composer.canReEditLatestResult(context: FixtureField(documentIdentifier: UUID(), selectedText: "", documentContextBeforeInput: "Dear Hello", documentContextAfterInput: " team")), "different field")
        XCTAssertFalse(composer.canReEditLatestResult(context: FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear Hel", documentContextAfterInput: "lo team")), "cursor moved into the result")
        XCTAssertFalse(composer.canReEditLatestResult(context: FixtureField(documentIdentifier: document, selectedText: "Hello", documentContextBeforeInput: "Dear ", documentContextAfterInput: " team")), "a selection exists")
        XCTAssertFalse(composer.canReEditLatestResult(context: FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear ", documentContextAfterInput: "Hello team")), "cursor moved before the result")

        let emptySuite = suiteName + ".empty"
        let emptyDefaults = UserDefaults(suiteName: emptySuite)!
        defer { emptyDefaults.removePersistentDomain(forName: emptySuite) }
        var none = KeyboardComposer(defaults: emptyDefaults)
        XCTAssertFalse(none.canReEditLatestResult(context: insertedField()), "nothing inserted yet")
        _ = try none.begin(context: emptyField())
    }

    func testBeginReEditStartsEditModeOverTheResultWithoutDeletingAnything() throws {
        var composer = try composerWithInsertedResult()

        let command = try composer.beginReEdit(context: insertedField(), requestID: "req-reedit")
        XCTAssertEqual(command.kind, .start)
        XCTAssertEqual(command.requestID, "req-reedit")
        XCTAssertEqual(command.options?.mode, .edit)
        XCTAssertEqual(command.options?.selectedText, "Hello")
        XCTAssertEqual(command.options?.contextBefore, "Dear ")
        XCTAssertEqual(command.options?.contextAfter, " team")
        XCTAssertEqual(composer.gate.pendingReEdit?.oldText, "Hello")
        XCTAssertEqual(composer.gate.pendingReEdit?.requestID, "req-reedit")
        XCTAssertEqual(composer.preferences.mode, .dictate, "the user's mode chip is not switched")
        XCTAssertTrue(composer.canReEditLatestResult(context: insertedField()), "the original text is still in the field")
    }

    func testBeginReEditThrowsWhenTheFieldNoLongerMatches() throws {
        var composer = try composerWithInsertedResult()
        let movedCursor = FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear Hel", documentContextAfterInput: "lo team")
        XCTAssertThrowsError(try composer.beginReEdit(context: movedCursor))
        XCTAssertNil(KeyboardComposer(defaults: defaults).gate.pendingReEdit)
    }

    func testUnchangedResultReplacesOnlyTheOldText() throws {
        var composer = try composerWithInsertedResult()
        _ = try composer.beginReEdit(context: insertedField(), requestID: "req-reedit")

        let decision = composer.receive(current: insertedField(), text: "Greetings", resultID: "result-2", requestID: "req-reedit")
        XCTAssertEqual(decision, .replace(oldText: "Hello", newText: "Greetings"))
        XCTAssertNil(composer.gate.pendingReEdit)
        XCTAssertEqual(KeyboardComposer(defaults: defaults).gate.latestInsertion?.inserted, "Greetings", "re-edit chains on its own result")
    }

    func testChangedFieldRecoversWithoutTouchingTheOriginal() throws {
        var composer = try composerWithInsertedResult()
        _ = try composer.beginReEdit(context: insertedField(), requestID: "req-reedit")

        let editedElsewhere = FixtureField(documentIdentifier: document, selectedText: "", documentContextBeforeInput: "Dear Hello!", documentContextAfterInput: " team")
        XCTAssertEqual(composer.receive(current: editedElsewhere, text: "Greetings", resultID: "result-2", requestID: "req-reedit"), .recoverFromHistory)
        XCTAssertEqual(KeyboardComposer(defaults: defaults).gate.latestInsertion?.inserted, "Hello", "the failed replace keeps the previous result addressable")
    }

    func testCancelDropsTheLateReEditResult() throws {
        var composer = try composerWithInsertedResult()
        _ = try composer.beginReEdit(context: insertedField(), requestID: "req-reedit")
        composer.cancel()

        XCTAssertEqual(composer.receive(current: insertedField(), text: "late", resultID: "result-late", requestID: "req-reedit"), .drop)
    }

    func testANewCompositionClearsAPendingReEdit() throws {
        var composer = try composerWithInsertedResult()
        _ = try composer.beginReEdit(context: insertedField(), requestID: "req-reedit")

        _ = try composer.begin(context: insertedField(), requestID: "req-next")
        XCTAssertNil(KeyboardComposer(defaults: defaults).gate.pendingReEdit)
        XCTAssertEqual(composer.receive(current: insertedField(), text: "other", resultID: "result-other", requestID: "req-reedit"), .drop, "the superseded re-edit request is blocked")
    }
}

// The tap block runs on RealtimeMessenger.mServiceQueue; an inherited MainActor check there
// crashes on the first mic tap. Invoke the production callback off-main.
private final class CapturingAudioNode: AVAudioMixerNode {
    nonisolated(unsafe) var installedTap: AVAudioNodeTapBlock?
    override func installTap(onBus bus: AVAudioNodeBus, bufferSize: AVAudioFrameCount,
                             format: AVAudioFormat?, block tapBlock: @escaping AVAudioNodeTapBlock) {
        installedTap = tapBlock
    }
}

private struct BackgroundMicrophoneTap: @unchecked Sendable {
    let callback: AVAudioNodeTapBlock

    func invoke() -> Bool {
        let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 128)!
        buffer.frameLength = 128
        callback(buffer, AVAudioTime(sampleTime: 0, atRate: 48_000))
        return !Thread.isMainThread
    }
}

final class MicrophoneCallbackTests: XCTestCase {
    @MainActor
    func testMicrophoneTapRunsOnAudioThread() async {
        let capture = MicrophoneCapture()
        let input = CapturingAudioNode()
        capture.installInputTap(on: input, format: AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!)
        let invocation = BackgroundMicrophoneTap(callback: input.installedTap!)
        let ranOffMain = await Task.detached { invocation.invoke() }.value
        XCTAssertTrue(ranOffMain)
        withExtendedLifetime(capture) {}
    }

    /// The floating window's playback session changes the audio configuration, which stops
    /// AVAudioEngine and leaves the next dictation near-silent. Restart it while the session
    /// is open, and only then.
    @MainActor
    func testEngineRestartsAfterAConfigurationChangeOnlyWhileTheSessionIsOpen() {
        let center = NotificationCenter()
        let capture = MicrophoneCapture(notificationCenter: center)
        var restarts = 0
        capture.restartEngine = { restarts += 1 }
        center.post(name: .AVAudioEngineConfigurationChange, object: capture.engine)
        XCTAssertEqual(restarts, 0, "no session open")
        capture.isSessionOpen = true
        center.post(name: .AVAudioEngineConfigurationChange, object: capture.engine)
        XCTAssertEqual(restarts, 1)
        center.post(name: .AVAudioEngineConfigurationChange, object: AVAudioEngine())
        XCTAssertEqual(restarts, 1, "another engine's change is ignored")
        center.post(name: AVAudioSession.interruptionNotification, object: nil, userInfo: [
            AVAudioSessionInterruptionTypeKey: AVAudioSession.InterruptionType.ended.rawValue,
        ])
        XCTAssertEqual(restarts, 2, "restart when an interruption ends")
        center.post(name: AVAudioSession.interruptionNotification, object: nil, userInfo: [
            AVAudioSessionInterruptionTypeKey: AVAudioSession.InterruptionType.began.rawValue,
        ])
        XCTAssertEqual(restarts, 2)
        capture.isSessionOpen = false
        center.post(name: .AVAudioEngineConfigurationChange, object: capture.engine)
        XCTAssertEqual(restarts, 2)
    }

    /// dictus (MIT, UnifiedAudioEngine.swift:156-166): an active playAndRecord session
    /// without this flag mutes system haptics on the whole device, keyboards included.
    @MainActor
    func testRecordingSessionKeepsSystemHapticsAndSounds() throws {
        let session = RecordingAudioSession()
        try MicrophoneCapture.configure(session)
        XCTAssertEqual(session.calls, ["category", "allowHaptics:true", "active:true"])
    }
}

private final class RecordingAudioSession: AudioSessionConfiguring {
    var calls: [String] = []
    func setCategory(_ category: AVAudioSession.Category, mode: AVAudioSession.Mode, options: AVAudioSession.CategoryOptions) throws {
        calls.append("category")
    }
    func setAllowHapticsAndSystemSoundsDuringRecording(_ allow: Bool) throws { calls.append("allowHaptics:\(allow)") }
    func activate() throws { calls.append("active:true") }
}

/// startPictureInPicture() does nothing, with no error, while isPictureInPicturePossible is
/// false; tapping Open must not fail silently.
@MainActor
final class FloatingWindowStartTests: XCTestCase {
    final class FakePiP: PictureInPictureControlling {
        var isPictureInPicturePossible = false
        var starts = 0
        var onPossible: (@MainActor @Sendable () -> Void)?
        var onStart: (() -> Void)?
        func startPictureInPicture() { starts += 1; onStart?() }
        func stopPictureInPicture() {}
        func observePossible(_ handler: @escaping @MainActor @Sendable () -> Void) { onPossible = handler }
    }

    func testVariantParsingDefaultsToTheMeasuredFix() {
        let all: FloatingDictationWindow.Variant = [.timebase, .audioSession, .surface, .liveRange]
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: nil), all.union(.deferredStart),
                       "the default starts on the next main-queue turn (second measurement)")
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "all"), all)
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "all-record"), all.union(.recordSession))
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "both"), [.timebase, .audioSession])
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "surface"), [.timebase, .audioSession, .surface])
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "liverange"), [.timebase, .audioSession, .liveRange])
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "none"), [])
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "timebase"), [.timebase])
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "session"), [.audioSession])
        for name in ["none", "timebase", "session", "both", "surface", "liverange", "all", "all-record"] {
            XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: name).name, name)
        }
    }

    /// Without a running timebase the layer never displays a frame and the card's preview
    /// stays blank. The timebase runs on host time, matching frame timestamps.
    func testTimebaseVariantGivesTheStatusLayerARunningClock() {
        let with = FloatingDictationWindow(available: true, controller: FakePiP(), variant: [.timebase])
        XCTAssertNotNil(with.displayLayer.controlTimebase)
        XCTAssertEqual(with.displayLayer.controlTimebase.map(CMTimebaseGetRate), 1)
        let layerTime = with.displayLayer.controlTimebase.map { CMTimeGetSeconds(CMTimebaseGetTime($0)) } ?? 0
        XCTAssertEqual(layerTime, CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock())), accuracy: 1)
        let without = FloatingDictationWindow(available: true, controller: FakePiP(), variant: [])
        XCTAssertNil(without.displayLayer.controlTimebase)
    }

    func testSessionVariantPreparesPlaybackAudioBeforeStarting() {
        var order: [String] = []
        let fake = FakePiP()
        fake.isPictureInPicturePossible = true
        fake.onStart = { order.append("start") }
        let window = FloatingDictationWindow(available: true, controller: fake, variant: [.audioSession],
                                             prepareAudioSession: { order.append("session") })
        window.start()
        XCTAssertEqual(order, ["session", "start"])
        let plain = FloatingDictationWindow(available: true, controller: fake, variant: [],
                                            prepareAudioSession: { order.append("unexpected") })
        plain.start()
        XCTAssertFalse(order.contains("unexpected"))
    }

    func testSessionFailureIsShownInsteadOfASilentNoOp() {
        struct Denied: LocalizedError { var errorDescription: String? { "denied" } }
        let window = FloatingDictationWindow(available: true, controller: FakePiP(), variant: [.audioSession],
                                             prepareAudioSession: { throw Denied() })
        window.start()
        XCTAssertEqual(window.lastError, "Couldn't open the floating window. iOS didn't allow Picture in Picture.")
        XCTAssertTrue(window.diagnostics.contains("session=denied"))
    }

    func testSurfaceVariantUsesIOSurfaceFramesAndKeepsSendingThem() async throws {
        let fake = FakePiP()
        fake.isPictureInPicturePossible = true
        let window = FloatingDictationWindow(available: true, controller: fake, startTimeout: 5,
                                             variant: [.surface], frameInterval: 0.1)
        let plain = FloatingDictationWindow(available: true, controller: FakePiP(), variant: [])
        XCTAssertTrue(window.lastFrameUsesIOSurface)
        XCTAssertFalse(plain.lastFrameUsesIOSurface)
        let before = window.framesEnqueued
        window.start()
        try await Task.sleep(for: .milliseconds(450))
        XCTAssertGreaterThanOrEqual(window.framesEnqueued - before, 3, "a frame on start and then steadily")
        XCTAssertTrue(window.diagnostics.contains("frames="))
        XCTAssertTrue(window.diagnostics.contains("layer="))
        window.handleDidStop()
        let stopped = window.framesEnqueued
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(window.framesEnqueued, stopped, "no frames once the window is closed")
    }

    /// A PiP source layer inside the Settings sheet closes the window when Settings is
    /// dismissed. The source layer is hosted at the app root; the card shows a separate
    /// preview layer with the same frames.
    func testStatusFramesReachTheSourceAndASeparatePreview() {
        let window = FloatingDictationWindow(available: true, controller: FakePiP(), variant: [.surface])
        XCTAssertFalse(window.previewLayer === window.displayLayer)
        let before = window.previewFramesEnqueued
        window.show(.listening)
        XCTAssertEqual(window.previewFramesEnqueued, before + 1)
        XCTAssertGreaterThanOrEqual(window.framesEnqueued, 2)
    }

    func testLiveRangeVariantReportsAnUnboundedLiveStream() {
        let live = FloatingDictationWindow(available: true, controller: FakePiP(), variant: [.liveRange])
        XCTAssertTrue(live.playbackDelegate.timeRange.start.isNegativeInfinity)
        XCTAssertTrue(live.playbackDelegate.timeRange.duration.isPositiveInfinity)
        let plain = FloatingDictationWindow(available: true, controller: FakePiP(), variant: [])
        XCTAssertEqual(plain.playbackDelegate.timeRange.start, .zero)
    }

    /// One variant per difference from a known-working sample-buffer PiP
    /// (uakihir0/UIPiPView): exclusive movie-playback audio, a finite time range, start
    /// on the next main-queue turn, and no linear-playback restriction.
    func testVariantsForEachDifferenceFromAWorkingSampleBufferPiP() {
        let all: FloatingDictationWindow.Variant = [.timebase, .audioSession, .surface, .liveRange]
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "exclusive"), all.union(.exclusiveAudio))
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "finite"), [.timebase, .audioSession, .surface, .finiteRange])
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "deferred"), all.union(.deferredStart))
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "nonlinear"), all.union(.nonLinear))
        XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: "uipip"),
                       [.timebase, .audioSession, .surface, .exclusiveAudio, .finiteRange, .deferredStart, .nonLinear])
        for name in ["exclusive", "finite", "deferred", "nonlinear", "uipip"] {
            XCTAssertEqual(FloatingDictationWindow.Variant(launchValue: name).name, name)
        }
        let exclusive = FloatingDictationWindow.sessionSettings(for: [.audioSession, .exclusiveAudio])
        XCTAssertEqual(exclusive.category, .playback)
        XCTAssertEqual(exclusive.mode, .moviePlayback)
        XCTAssertEqual(exclusive.options, [])
        let mixing = FloatingDictationWindow.sessionSettings(for: [.audioSession])
        XCTAssertEqual(mixing.mode, .default)
        XCTAssertEqual(mixing.options, [.mixWithOthers])
        let finite = FloatingDictationWindow(available: true, controller: FakePiP(), variant: [.finiteRange])
        XCTAssertEqual(finite.playbackDelegate.timeRange.start, .zero)
        XCTAssertEqual(CMTimeGetSeconds(finite.playbackDelegate.timeRange.duration), 86_400)
        XCTAssertTrue(FloatingDictationWindow.requiresLinearPlayback(for: all))
        XCTAssertFalse(FloatingDictationWindow.requiresLinearPlayback(for: all.union(.nonLinear)))
    }

    func testDeferredStartCallsTheSystemOnTheNextMainQueueTurn() async throws {
        let fake = FakePiP()
        fake.isPictureInPicturePossible = true
        let window = FloatingDictationWindow(available: true, controller: fake, startTimeout: 5, variant: [.deferredStart])
        window.start()
        XCTAssertEqual(fake.starts, 0)
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(fake.starts, 1)
    }

    func testRecordSessionVariantUsesTheMicrophoneCategory() {
        XCTAssertEqual(FloatingDictationWindow.sessionCategory(for: [.audioSession]), .playback)
        XCTAssertEqual(FloatingDictationWindow.sessionCategory(for: [.audioSession, .recordSession]), .playAndRecord)
    }

    /// With possible=true, iOS can ignore start() and neither start nor fail; the
    /// synchronous path would show nothing at all.
    func testStartThatTheSystemIgnoresShowsAnError() async throws {
        let fake = FakePiP()
        fake.isPictureInPicturePossible = true
        var failure: String?
        let window = FloatingDictationWindow(available: true, controller: fake, startTimeout: 0.2)
        window.onFailure = { failure = $0 }
        window.start()
        XCTAssertEqual(fake.starts, 1)
        try await Task.sleep(for: .milliseconds(600))
        XCTAssertEqual(window.lastError, "Couldn't open the floating window. iOS didn't allow Picture in Picture.")
        XCTAssertEqual(failure, window.lastError)
        XCTAssertTrue(window.diagnostics.contains("start=ignored"), window.diagnostics)
    }

    func testStartThatSucceedsShowsNoError() async throws {
        let fake = FakePiP()
        fake.isPictureInPicturePossible = true
        let window = FloatingDictationWindow(available: true, controller: fake, startTimeout: 0.2)
        fake.onStart = { window.handleDidStart() }
        window.start()
        try await Task.sleep(for: .milliseconds(600))
        XCTAssertTrue(window.isActive)
        XCTAssertNil(window.lastError)
    }

    func testStartsAtOnceWhenPossible() {
        let fake = FakePiP()
        fake.isPictureInPicturePossible = true
        let window = FloatingDictationWindow(available: true, controller: fake, startTimeout: 0.2)
        window.start()
        XCTAssertEqual(fake.starts, 1)
    }

    func testWaitsUntilPossibleThenStarts() {
        let fake = FakePiP()
        let window = FloatingDictationWindow(available: true, controller: fake, startTimeout: 5)
        window.start()
        XCTAssertEqual(fake.starts, 0, "starting while not possible is a silent no-op on device")
        fake.isPictureInPicturePossible = true
        fake.onPossible?()
        XCTAssertEqual(fake.starts, 1)
        XCTAssertNil(window.lastError)
    }

    func testReportsAVisibleErrorWhenItNeverBecomesPossible() async throws {
        let fake = FakePiP()
        var failure: String?
        let window = FloatingDictationWindow(available: true, controller: fake, startTimeout: 0.2)
        window.onFailure = { failure = $0 }
        window.start()
        try await Task.sleep(for: .milliseconds(600))
        XCTAssertEqual(fake.starts, 0)
        XCTAssertEqual(window.lastError, "Couldn't open the floating window. iOS didn't allow Picture in Picture.")
        XCTAssertEqual(failure, window.lastError)
        XCTAssertFalse(window.isActive)
    }
}
