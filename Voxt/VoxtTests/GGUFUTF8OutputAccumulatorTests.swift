// GGUFUTF8OutputAccumulatorTests.swift
// Provides GGUF UTF-8 output accumulator tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class GGUFUTF8OutputAccumulatorTests: XCTestCase {
    func testWaitsForCompleteMultibyteSequenceBeforeDecoding() {
        var accumulator = GGUFUTF8OutputAccumulator()
        let text = "繁體工程師 français"
        let bytes = Array(text.utf8)
        let splitIndex = bytes.firstIndex(of: 0x94) ?? 1

        XCTAssertNil(accumulator.append(Array(bytes[..<splitIndex])))
        XCTAssertEqual(accumulator.append(Array(bytes[splitIndex...])), text)
        XCTAssertEqual(accumulator.finalizedText(), text)
        XCTAssertFalse(accumulator.finalizedWithReplacementCharacters)
    }

    func testFinalizesInvalidUTF8WithReplacementFlag() {
        var accumulator = GGUFUTF8OutputAccumulator()

        XCTAssertNil(accumulator.append([0xE7]))
        XCTAssertEqual(accumulator.finalizedText(), "\u{FFFD}")
        XCTAssertTrue(accumulator.finalizedWithReplacementCharacters)
    }

    @MainActor
    func testApplicationTerminationShutdownRejectsNewGGUFInference() async {
        let manager = GGUFTranslationModelManager(modelID: .hyMT2Q4KM)
        await manager.shutdownForApplicationTermination()
        let request = LLMCompiledRequest(
            taskLabel: "translation",
            instructions: "Translate the text.",
            prompt: "hello",
            debugInput: "hello",
            fallbackText: "hello",
            inputCharacterCount: 5,
            outputTokenBudgetHint: nil,
            conversationHistory: [],
            previousResponseID: nil,
            responseFormat: nil
        )

        do {
            _ = try await manager.executeCompiledRequest(request, modelID: .hyMT2Q4KM)
            XCTFail("GGUF inference should not start after application termination shutdown.")
        } catch is CancellationError {
            // Expected: shutdown closes the native backend and rejects new work.
        } catch {
            XCTFail("Expected CancellationError, got \(error).")
        }
    }

    @MainActor
    func testInstalledGGUFModelIsExplicitlyReleasedDuringApplicationTermination() async throws {
        try ModelTestGate.requireEnabled("GGUF native termination integration test")
        let manager = GGUFTranslationModelManager(modelID: .hyMT2Q4KM)
        _ = try await manager.refreshInstallation(id: .hyMT2Q4KM)
        guard manager.isModelDownloaded(id: .hyMT2Q4KM) else {
            throw XCTSkip("Installed GGUF model is required for the native termination integration test.")
        }
        let request = LLMCompiledRequest(
            taskLabel: "termination-integration",
            instructions: "Translate the user text to Chinese. Return only the translation.",
            prompt: "hello",
            debugInput: "hello",
            fallbackText: "你好",
            inputCharacterCount: 5,
            outputTokenBudgetHint: 48,
            conversationHistory: [],
            previousResponseID: nil,
            responseFormat: nil
        )

        _ = try await manager.executeCompiledRequest(request, modelID: .hyMT2Q4KM)
        await manager.shutdownForApplicationTermination()

        // The llama backend is process-scoped. Reacquiring it after a complete release verifies
        // that one runtime cannot permanently poison later runtimes in the same process.
        let restartedManager = GGUFTranslationModelManager(modelID: .hyMT2Q4KM)
        _ = try await restartedManager.executeCompiledRequest(request, modelID: .hyMT2Q4KM)
        await restartedManager.shutdownForApplicationTermination()
    }
}
