// SPDX-License-Identifier: Apache-2.0
import AppKit
import ApplicationServices
import Testing
@testable import OmniTyper

@MainActor
struct AppModelTests {
    static func target(selectedText: String = "Selected text") -> InsertionTarget {
        InsertionTarget(applicationName: "Test editor", bundleID: "org.omnityper.test-editor",
                        selectedText: selectedText, application: .current, element: nil, range: nil,
                        value: nil, window: nil, windowTitle: nil, document: nil)
    }

    @Test func shortcutReleaseCapturesOnceAndCancellationDoesNotCapture() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = AppStore(directory: directory)
        store.preferences.pythonExecutable = directory.appendingPathComponent("missing-python").path
        var captures = 0
        let model = AppModel(store: store, captureTarget: { captures += 1; return Self.target() })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.shortcutPressed(pointerX: 0, pointerY: 0)
        #expect(model.phase == .idle && captures == 0)
        model.cancel()
        model.shortcutReleased(pointerX: 0, pointerY: 0)
        #expect(model.phase == .idle && captures == 0)
        model.shortcutPressed(pointerX: 0, pointerY: 0)
        model.shortcutReleased(pointerX: 0, pointerY: 0)
        #expect(model.phase == .starting && captures == 1)
        model.shortcutReleased(pointerX: 0, pointerY: 0)
        #expect(captures == 1)
    }

    @Test func textModesRejectMissingModelBeforeStartingTheWorker() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let model = AppModel(store: AppStore(directory: directory), captureTarget: { Self.target() })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        for mode in [VoiceMode.translate, .edit, .ask] {
            model.toggle(mode)
            #expect(model.error == L("error.model"))
            #expect(model.phase == .idle && !model.worker.isRunning)
        }
    }

    @Test func voiceEditRetryCapturesTheNewSelection() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = AppStore(directory: directory)
        store.preferences.pythonExecutable = directory.appendingPathComponent("missing-python").path
        store.preferences.textSettings.model = "test-model"
        var destination = Self.target(selectedText: "")
        let model = AppModel(store: store, captureTarget: { destination })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.toggle(.edit)
        #expect(model.error == L("error.editNeedsSelection"))
        #expect(model.phase == .idle && !model.worker.isRunning)
        destination = Self.target(selectedText: "Now selected")
        model.start()
        #expect(model.phase == .starting && model.error.isEmpty)
        #expect(try model.payload(audio: nil)["selected_text"] as? String == "Now selected")
    }

    @Test func modeChangesRevalidateTheCapturedTargetAndLockDuringProcessing() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        var captures = 0
        let model = AppModel(store: AppStore(directory: directory), captureTarget: {
            captures += 1
            return Self.target(selectedText: "")
        })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.toggle(.translate)
        model.selectMode(.edit)
        #expect(model.error == L("error.editNeedsSelection"))
        model.selectMode(.dictate)
        #expect(model.error.isEmpty && captures == 1)
        model.phase = .recording
        model.selectMode(.ask)
        #expect(model.error == L("error.model") && model.phase == .recording)
        model.useVerbatimDictation()
        #expect(model.mode == .dictate && model.error.isEmpty)
        #expect(model.phase == .recording && captures == 1)
        model.phase = .processing
        model.selectMode(.translate)
        #expect(model.mode == .dictate)
    }

    @Test func verbatimRecoveryOverridesAppRulesOnlyForThisAttempt() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = AppStore(directory: directory)
        store.preferences.pythonExecutable = directory.appendingPathComponent("missing-python").path
        store.preferences.style = "formal"
        store.rules = [AppRule(bundleID: "org.omnityper.test-editor", name: "Test editor", style: "clean", instructions: "")]
        let model = AppModel(store: store, captureTarget: { Self.target() })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.toggle(.translate)
        model.useVerbatimDictation()
        let request = try model.payload(audio: nil)
        #expect(request["style"] as? String == "verbatim")
        #expect(request["text_model"] == nil)
        #expect(store.preferences.style == "formal" && store.rules.first?.style == "clean")
        model.cancel()
        model.toggle(.dictate)
        #expect(model.error == L("error.model"))
        #expect(model.recordingStyle == "clean")
    }

    @Test func verbatimRecoveryStillRejectsSecureFields() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let model = AppModel(store: AppStore(directory: directory), captureTarget: { throw TextInsertionError.secureField })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.toggle(.translate)
        model.useVerbatimDictation()
        #expect(model.error == L("sys.secureField"))
        #expect(model.phase == .idle && !model.worker.isRunning)
    }

    @Test func startupFailureReturnsToIdleAndPreservesTheError() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = AppStore(directory: directory)
        store.preferences.pythonExecutable = directory.appendingPathComponent("missing-python").path
        let model = AppModel(store: store, captureTarget: { Self.target() })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.start()
        await model.task?.value
        #expect(model.phase == .idle && !model.worker.isRunning)
        #expect(model.error.contains("missing-python"))
        model.cancel()
        #expect(model.error.contains("missing-python"))
    }

    @Test func draftOperationsRespectUnicodeCaretsAndRejectChangedDocuments() throws {
        let text = "Hi 🌏 你好"
        let caret = DraftSelection(text: text, range: NSRange(location: 6, length: 0))
        #expect(try caret.replacing(with: "new ", in: text) == "Hi 🌏 new 你好")
        let selected = DraftSelection(text: text, range: NSRange(location: 3, length: 2))
        #expect(try selected.selectedText() == "🌏")
        #expect(try selected.replacing(with: "world", in: text) == "Hi world 你好")
        #expect(throws: (any Error).self) { try selected.replacing(with: "world", in: text + "!") }
        for range in [NSRange(location: 4, length: 1), NSRange(location: Int.max, length: 2)] {
            #expect(throws: (any Error).self) { try DraftSelection(text: text, range: range).selectedText() }
        }
    }

    @Test func editorCancellationPreservesTheDraftUntilExplicitlyCleared() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = AppStore(directory: directory)
        store.preferences.pythonExecutable = directory.appendingPathComponent("missing-python").path
        store.preferences.textSettings.model = "test-model"
        var captures = 0
        let model = AppModel(store: store, captureTarget: { captures += 1; return Self.target() })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.isReviewingResult = true
        model.resultDraft = "Keep this text"
        model.draftSelection = NSRange(location: 5, length: 4)
        model.selectMode(.edit)
        model.start()
        #expect(captures == 0)
        #expect(try model.payload(audio: nil)["selected_text"] as? String == "this")
        model.cancel()
        #expect(model.phase == .idle && model.isReviewingResult)
        #expect(model.resultDraft == "Keep this text" && model.draftSelection == NSRange(location: 5, length: 4))
        model.clearReviewedResult()
        #expect(model.resultDraft.isEmpty && model.draftSelection == NSRange(location: 0, length: 0))
    }

    @Test func voiceEditWithoutSelectionUsesOnlyTheCurrentFullTextField() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = AppStore(directory: directory)
        store.preferences.textSettings.model = "test-model"
        store.preferences.pythonExecutable = directory.appendingPathComponent("missing-python").path
        let model = AppModel(store: store)
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.mode = .edit
        model.isReviewingResult = true
        model.resultDraft = "Whole 🌏 document"
        model.draftSelection = NSRange(location: 2, length: 0)
        model.start()
        #expect(model.isEditingEntireField && model.notice == L("panel.editingAllText"))
        #expect(try model.payload(audio: nil)["selected_text"] as? String == "Whole 🌏 document")
        model.cancel()
        model.finishReview()
        model.target = InsertionTarget(applicationName: "Test", bundleID: "test", selectedText: "",
                                      application: .current, element: AXUIElementCreateApplication(getpid()),
                                      range: CFRange(location: 3, length: 0), value: "Current field only", window: nil,
                                      windowTitle: nil, document: nil)
        model.prepareEditSelection()
        #expect(model.target?.range?.location == 3 && model.target?.range?.length == 0)
        #expect(model.target?.replacementRange?.length == 18)
        #expect(try model.payload(audio: nil)["selected_text"] as? String == "Current field only")
        model.mode = .dictate
        model.prepareEditSelection()
        #expect(model.target?.replacementRange == nil && !model.isEditingEntireField)
    }

    @Test func failedReviewInsertionKeepsTheDraftAndAllowsRetry() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        var rejectInsertion = true
        var inserted: [String] = []
        let model = AppModel(store: AppStore(directory: directory), insertText: { text, _, restoringFocus in
            #expect(restoringFocus)
            if rejectInsertion { throw Failure("sys.contentChanged") }
            inserted.append(text)
        })
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.isReviewingResult = true
        model.resultDraft = "Reviewed text"
        model.reviewTarget = Self.target()
        model.insertReviewedResult()
        await model.task?.value
        #expect(inserted.isEmpty && !model.error.isEmpty)
        #expect(model.resultDraft == "Reviewed text" && model.canInsertReview)
        rejectInsertion = false
        model.insertReviewedResult()
        await model.task?.value
        #expect(inserted == ["Reviewed text"] && model.error.isEmpty)
        #expect(model.resultDraft == "Reviewed text" && !model.canInsertReview)
        model.insertReviewedResult()
        #expect(inserted.count == 1)
    }
}
