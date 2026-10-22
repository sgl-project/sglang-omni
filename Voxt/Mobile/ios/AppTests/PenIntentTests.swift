import XCTest
@testable import VoxtMobile
import VoxtMobileKit

private struct Field: CompositionTextContext {
    var documentIdentifier: UUID
    var selectedText: String?
    var documentContextBeforeInput: String?
    var documentContextAfterInput: String?

    init(_ document: UUID, selected: String = "", before: String, after: String = "") {
        documentIdentifier = document
        selectedText = selected
        documentContextBeforeInput = before
        documentContextAfterInput = after
    }
}

/// One pen button decides what it acts on from the field; a plain mic tap
/// always dictates, and a chosen intent applies to one recording only.
final class PenIntentTests: XCTestCase {
    private let document = UUID()

    private func makeComposer() -> KeyboardComposer {
        let name = "pen-intent.\(UUID().uuidString)"
        addTeardownBlock { UserDefaults().removePersistentDomain(forName: name) }
        return KeyboardComposer(defaults: UserDefaults(suiteName: name)!)
    }

    func testPenEditsTheSelectionThenOurLastResultOtherwiseWrites() throws {
        var composer = makeComposer()
        XCTAssertNil(composer.penIntent(context: nil), "no field, nothing to act on")
        XCTAssertEqual(composer.penIntent(context: Field(document, selected: "Monday", before: "See you ")), .editSelection)
        let field = Field(document, before: "Hi ")
        XCTAssertEqual(composer.penIntent(context: field), .write)

        let command = try composer.begin(context: field, intent: nil)
        XCTAssertEqual(composer.receive(current: field, text: "Running late.", resultID: "r1", requestID: command.requestID),
                       .insert("Running late."))
        XCTAssertEqual(composer.penIntent(context: Field(document, before: "Hi Running late.")), .editLastResult)
        XCTAssertEqual(composer.penIntent(context: Field(document, before: "Hi Running late. x")), .write,
                       "typing after the result means it is no longer ours to edit")
        XCTAssertEqual(composer.penIntent(context: Field(document, selected: "late", before: "Hi Running ")), .editSelection,
                       "a selection wins over the last result")
    }

    func testAPlainTapDictatesEvenWithAnOldSavedMode() throws {
        var composer = makeComposer()
        composer.preferences.mode = .translate
        let command = try composer.begin(context: Field(document, before: "Hi "), intent: nil)
        XCTAssertEqual(command.options?.mode, .dictate)
        XCTAssertNil(composer.editScope)
    }

    func testEachIntentStartsItsOwnModeForThisRecordingOnly() throws {
        var composer = makeComposer()
        let selection = try composer.begin(context: Field(document, selected: "Monday", before: "See you "), intent: .editSelection)
        XCTAssertEqual(selection.options?.mode, .edit)
        XCTAssertEqual(selection.options?.selectedText, "Monday")
        XCTAssertEqual(composer.editScope, .selection)
        composer.cancel()

        let write = try composer.begin(context: Field(document, before: ""), intent: .write)
        XCTAssertEqual(write.options?.mode, .write)
        XCTAssertNil(composer.editScope)
        composer.cancel()

        let plain = try composer.begin(context: Field(document, before: ""), intent: nil)
        XCTAssertEqual(plain.options?.mode, .dictate, "the next recording is plain dictation again")
        XCTAssertEqual(composer.preferences.mode, .dictate, "an intent never becomes the saved mode")
    }

    func testEditingTheLastResultTargetsOnlyThatResult() throws {
        var composer = makeComposer()
        let field = Field(document, before: "Hi ", after: " Bye")
        let command = try composer.begin(context: field, intent: nil)
        _ = composer.receive(current: field, text: "Running late.", resultID: "r1", requestID: command.requestID)
        let edit = try composer.begin(context: Field(document, before: "Hi Running late.", after: " Bye"), intent: .editLastResult)
        XCTAssertEqual(edit.options?.mode, .edit)
        XCTAssertEqual(edit.options?.selectedText, "Running late.")
        XCTAssertEqual(composer.editScope, .lastResult)
    }

    func testAnIntentThatNoLongerFitsTheFieldIsRefused() {
        var composer = makeComposer()
        XCTAssertThrowsError(try composer.begin(context: Field(document, before: "Hi "), intent: .editSelection)) {
            XCTAssertEqual($0 as? CompositionEditScopeError, .nothingToEdit)
        }
        XCTAssertThrowsError(try composer.begin(context: Field(document, before: "Hi "), intent: .editLastResult)) {
            XCTAssertEqual($0 as? CompositionReEditError, .unavailable)
        }
    }

    /// A pill translates this one recording into its language; the next tap dictates.
    func testAPillTranslatesThisRecordingOnly() throws {
        var composer = makeComposer()
        let translate = try composer.begin(context: Field(document, before: "Hi "), translateTo: "ja")
        XCTAssertEqual(translate.options?.mode, .translate)
        XCTAssertEqual(translate.options?.targetLanguage, "ja")
        XCTAssertNil(composer.editScope)
        composer.cancel()
        let plain = try composer.begin(context: Field(document, before: "Hi "), intent: nil)
        XCTAssertEqual(plain.options?.mode, .dictate)
        XCTAssertNil(plain.options?.targetLanguage)
        XCTAssertEqual(composer.preferences.mode, .dictate, "a pill never becomes the saved mode")
    }

    /// The keyboard can restart while Voxt takes the mic (handoff); the language must
    /// still show in "Translating to …" when it comes back.
    func testTheTranslationLanguageSurvivesAKeyboardRestart() throws {
        let name = "pen-intent.\(UUID().uuidString)"
        addTeardownBlock { UserDefaults().removePersistentDomain(forName: name) }
        let defaults = UserDefaults(suiteName: name)!
        var composer = KeyboardComposer(defaults: defaults)
        _ = try composer.begin(context: Field(document, before: "Hi "), translateTo: "es")
        XCTAssertEqual(composer.translatingTo, "es")
        XCTAssertEqual(KeyboardComposer(defaults: defaults).translatingTo, "es", "a new keyboard reads it back")
        composer.cancel()
        XCTAssertNil(KeyboardComposer(defaults: defaults).translatingTo)
        _ = try composer.begin(context: Field(document, before: "Hi "), intent: nil)
        XCTAssertNil(composer.translatingTo, "plain dictation translates nothing")
    }

    func testAPillMustNameASupportedLanguage() {
        var composer = makeComposer()
        XCTAssertThrowsError(try composer.begin(context: Field(document, before: ""), translateTo: "xx-unknown"))
    }

    func testThePenLineSaysWhatItWillActOn() {
        XCTAssertEqual(PenIntent.editSelection.line, "Editing your selection")
        XCTAssertEqual(PenIntent.editLastResult.line, "Editing what you just said")
        XCTAssertEqual(PenIntent.write.line, "Describe what to write")
    }
}
