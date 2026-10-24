import XCTest
@testable import VoxtMobile
import VoxtMobileKit

@MainActor
final class AppModelDeveloperPromptsTests: XCTestCase {
    private func makeModel() -> (AppModel, UserDefaults) {
        let name = "developer-prompts.\(UUID())", defaults = UserDefaults(suiteName: name)!
        addTeardownBlock { defaults.removePersistentDomain(forName: name) }
        return (AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults), defaults)
    }

    func testCustomPromptsAreSentOnlyWhileDeveloperModeIsOn() {
        let (model, _) = makeModel()
        XCTAssertFalse(model.developerModeEnabled)
        model.setDeveloperMode(true)
        model.setCustomPrompt("Be brief.", for: .dictate)
        XCTAssertEqual(model.customPromptOverride(for: .dictate), "Be brief.")
        XCTAssertNil(model.customPromptOverride(for: .translate))
        model.setDeveloperMode(false)
        XCTAssertNil(model.customPromptOverride(for: .dictate))
        XCTAssertEqual(model.promptText(for: .dictate), "Be brief.", "turning Developer mode off keeps the prompt")
    }

    func testEditorStartsFromTheMacsBuiltInPromptAndResetReturnsToIt() async {
        let (model, _) = makeModel()
        model.builtInPromptsLoader = { [.dictate: "Clean up the speech.", .edit: "Edit it."] }
        await model.loadBuiltInPrompts()
        model.setDeveloperMode(true)
        XCTAssertEqual(model.promptText(for: .dictate), "Clean up the speech.")
        XCTAssertFalse(model.hasCustomPrompt(for: .dictate))
        model.setCustomPrompt("Clean up the speech.", for: .dictate)
        XCTAssertNil(model.customPromptOverride(for: .dictate), "the built-in text is not a custom prompt")
        model.setCustomPrompt("Be brief.", for: .dictate)
        model.setCustomPrompt("Edit gently.", for: .edit)
        model.resetCustomPrompt(for: .dictate)
        XCTAssertEqual(model.promptText(for: .dictate), "Clean up the speech.")
        XCTAssertEqual(model.customPromptOverride(for: .edit), "Edit gently.")
        model.resetAllCustomPrompts()
        XCTAssertNil(model.customPromptOverride(for: .edit))
    }

    func testBuiltInPromptsThatCannotLoadSayWhy() async {
        let (model, _) = makeModel()
        model.builtInPromptsLoader = { throw VoxtClient.Error.server(status: 404, message: "") }
        await model.loadBuiltInPrompts()
        XCTAssertEqual(model.builtInPromptsError, "Update Voxt on your Mac to see its built-in prompts.")
    }
}
