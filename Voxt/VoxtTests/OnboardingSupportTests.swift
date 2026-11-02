// OnboardingSupportTests.swift
// Provides Onboarding Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class OnboardingSupportTests: XCTestCase {
    func testTranscriptionPermissionsIncludeSpeechRecognitionForDictation() {
        let permissions = OnboardingPermissionRequirementResolver.requiredPermissions(
            for: .transcription,
            context: OnboardingPermissionRequirementContext(
                selectedEngine: .dictation
            )
        )

        XCTAssertEqual(
            permissions,
            [.microphone, .accessibility, .speechRecognition]
        )
    }

    func testTranscriptionRequiresOnlyMicrophoneAndAccessibility() {
        let permissions = OnboardingPermissionRequirementResolver.requiredPermissions(
            for: .transcription,
            context: OnboardingPermissionRequirementContext(
                selectedEngine: .mlxAudio
            )
        )

        XCTAssertEqual(
            permissions,
            [.microphone, .accessibility]
        )
    }

    func testNonRecordingStepsDoNotRequirePermissions() {
        let context = OnboardingPermissionRequirementContext(
            selectedEngine: .mlxAudio
        )

        XCTAssertTrue(OnboardingPermissionRequirementResolver.requiredPermissions(for: .language, context: context).isEmpty)
        XCTAssertTrue(OnboardingPermissionRequirementResolver.requiredPermissions(for: .model, context: context).isEmpty)
        XCTAssertTrue(OnboardingPermissionRequirementResolver.requiredPermissions(for: .translation, context: context).isEmpty)
        XCTAssertTrue(OnboardingPermissionRequirementResolver.requiredPermissions(for: .rewrite, context: context).isEmpty)
        XCTAssertTrue(OnboardingPermissionRequirementResolver.requiredPermissions(for: .appEnhancement, context: context).isEmpty)
        XCTAssertTrue(OnboardingPermissionRequirementResolver.requiredPermissions(for: .finish, context: context).isEmpty)
    }

    func testFeatureSelectionResolverMapsASRSelections() {
        XCTAssertEqual(
            OnboardingFeatureSelectionResolver.asrSelectionID(
                selectedEngine: .mlxAudio,
                mlxModelRepo: "mlx-community/whisper-large-v3-turbo-4bit",
                remoteASRProvider: .doubaoASR
            ),
            .mlx("mlx-community/whisper-large-v3-turbo-4bit")
        )

        XCTAssertEqual(
            OnboardingFeatureSelectionResolver.asrSelectionID(
                selectedEngine: .remote,
                mlxModelRepo: "",
                remoteASRProvider: .doubaoASR
            ),
            .remoteASR(.doubaoASR)
        )
    }

    func testFeatureSelectionResolverMapsLLMSelections() {
        XCTAssertEqual(
            OnboardingFeatureSelectionResolver.llmSelectionID(
                choice: .local,
                localLLMRepo: "mlx-community/Qwen3-4B-4bit",
                remoteLLMProvider: .openAI
            ),
            .localLLM("mlx-community/Qwen3-4B-4bit")
        )

        XCTAssertEqual(
            OnboardingFeatureSelectionResolver.llmSelectionID(
                choice: .system,
                localLLMRepo: "",
                remoteLLMProvider: .openAI
            ),
            .appleIntelligence
        )
    }

    func testFeatureSelectionResolverMigratesLegacyWhisperDirectTranslateToLocalLLM() {
        let selection = OnboardingFeatureSelectionResolver.translationSelectionID(
            llmSelection: .appleIntelligence,
            asrSelection: .mlx(MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: "large-v3")),
            existingSelection: FeatureModelSelectionID(rawValue: "whisper-direct-translate"),
            fallbackLocalLLMRepo: "mlx-community/Qwen3-4B-4bit"
        )

        XCTAssertEqual(selection, .localLLM("mlx-community/Qwen3-4B-4bit"))
    }

    func testFeatureSelectionResolverPreservesExistingTranslationSelectionWhenCompatible() {
        let selection = OnboardingFeatureSelectionResolver.translationSelectionID(
            llmSelection: .appleIntelligence,
            asrSelection: .mlx("mlx-community/whisper-large-v3-turbo-4bit"),
            existingSelection: .remoteLLM(.openAI),
            fallbackLocalLLMRepo: "mlx-community/Qwen3-4B-4bit"
        )

        XCTAssertEqual(selection, .remoteLLM(.openAI))
    }
}
