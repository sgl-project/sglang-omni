// OnboardingSupport.swift
// Provides Onboarding Support for onboarding settings.

import SwiftUI
import Foundation
import AVFoundation
import Speech
import ApplicationServices

enum OnboardingModelPathChoice: String, CaseIterable, Identifiable {
    case local
    case remote
    case dictation

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .local:
            return "Local"
        case .remote:
            return "Remote"
        case .dictation:
            return "System"
        }
    }
}

enum OnboardingTextModelPathChoice: String, CaseIterable, Identifiable {
    case local
    case remote
    case system

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .local:
            return "Local"
        case .remote:
            return "Remote"
        case .system:
            return "System"
        }
    }
}

enum OnboardingContextualPermission: Hashable {
    case microphone
    case speechRecognition
    case accessibility

    var titleKey: LocalizedStringKey {
        switch self {
        case .microphone:
            return "Microphone Permission"
        case .speechRecognition:
            return "Speech Recognition Permission"
        case .accessibility:
            return "Accessibility Permission"
        }
    }

    var descriptionKey: LocalizedStringKey {
        switch self {
        case .microphone:
            return "Required to capture audio for transcription."
        case .speechRecognition:
            return "Required for Apple Direct Dictation engine."
        case .accessibility:
            return "Required for global shortcuts and inserting text into other apps."
        }
    }
}

struct OnboardingPermissionRequirementContext {
    let selectedEngine: TranscriptionEngine
}

enum OnboardingPermissionRequirementResolver {
    static func requiredPermissions(
        for step: OnboardingStep,
        context: OnboardingPermissionRequirementContext
    ) -> [OnboardingContextualPermission] {
        switch step {
        case .transcription:
            var permissions: [OnboardingContextualPermission] = [
                .microphone,
                .accessibility
            ]
            if context.selectedEngine == .dictation {
                permissions.append(.speechRecognition)
            }
            return permissions
        case .language, .model, .translation, .rewrite, .meeting, .appEnhancement, .finish:
            return []
        }
    }
}

enum OnboardingPermissionGrantResolver {
    static func isGranted(_ permission: OnboardingContextualPermission) -> Bool {
        switch permission {
        case .microphone:
            return AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        case .speechRecognition:
            return SFSpeechRecognizer.authorizationStatus() == .authorized
        case .accessibility:
            return AccessibilityPermissionManager.isTrusted()
        }
    }
}

enum OnboardingFeatureSelectionResolver {
    static func asrSelectionID(
        selectedEngine: TranscriptionEngine,
        mlxModelRepo: String,
        remoteASRProvider: RemoteASRProvider
    ) -> FeatureModelSelectionID {
        switch selectedEngine {
        case .dictation:
            return .dictation
        case .mlxAudio:
            return .mlx(mlxModelRepo)
        case .remote:
            return .remoteASR(remoteASRProvider)
        }
    }

    static func llmSelectionID(
        choice: OnboardingTextModelPathChoice,
        localLLMRepo: String,
        remoteLLMProvider: RemoteLLMProvider
    ) -> FeatureModelSelectionID {
        switch choice {
        case .local:
            return .localLLM(localLLMRepo)
        case .remote:
            return .remoteLLM(remoteLLMProvider)
        case .system:
            return .appleIntelligence
        }
    }

    static func translationSelectionID(
        llmSelection: FeatureModelSelectionID,
        asrSelection: FeatureModelSelectionID,
        existingSelection: FeatureModelSelectionID,
        fallbackLocalLLMRepo: String
    ) -> FeatureModelSelectionID {
        if existingSelection.rawValue == "whisper-direct-translate" {
            return .localLLM(fallbackLocalLLMRepo)
        }

        switch llmSelection.textSelection {
        case .localLLM(let repo):
            return .localLLM(repo)
        case .remoteLLM(let provider):
            return .remoteLLM(provider)
        case .appleIntelligence:
            switch existingSelection.translationSelection {
            case .localLLM, .localGGUF, .remoteLLM:
                return existingSelection
            case .none:
                return .localLLM(fallbackLocalLLMRepo)
            }
        case .none:
            return .localLLM(fallbackLocalLLMRepo)
        }
    }
}

enum OnboardingRewriteTest {
    static var defaultPrompt: String {
        defaultPrompt(localeIdentifier: AppLocalization.language.localeIdentifier)
    }

    static var defaultSourceText: String {
        defaultSourceText(localeIdentifier: AppLocalization.language.localeIdentifier)
    }

    static func defaultPrompt(localeIdentifier: String) -> String {
        AppLocalization.localizedString("Make this shorter and more polite.", localeIdentifier: localeIdentifier)
    }

    static func defaultSourceText(localeIdentifier: String) -> String {
        AppLocalization.localizedString(
            "Hi team, I wanted to follow up about tomorrow's launch. We are still waiting on the final banner image, so please send it over before 3 PM if possible. Thanks.",
            localeIdentifier: localeIdentifier
        )
    }
}

enum OnboardingTranslationTest {
    static var defaultInput: String {
        defaultInput(localeIdentifier: AppLocalization.language.localeIdentifier)
    }

    static func defaultInput(localeIdentifier: String) -> String {
        AppLocalization.localizedString(
            "Thanks for joining the call. I'll send the updated timeline and action items after lunch.",
            localeIdentifier: localeIdentifier
        )
    }
}

enum OnboardingVideoDemo {
    static let appEnhancementURL = URL(string: "https://storage.actnow.dev/common/voxt/voxt-app-branch-demo.mp4")!
}
