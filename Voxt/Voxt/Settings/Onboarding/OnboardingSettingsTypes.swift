// OnboardingSettingsTypes.swift
// Provides Onboarding Settings Types for onboarding settings.

import SwiftUI

enum SettingsDisplayMode: Equatable {
    case normal
    case onboarding(step: OnboardingStep)
}

enum OnboardingStep: String, CaseIterable, Identifiable {
    case language
    case model
    case transcription
    case translation
    case rewrite
    case meeting
    case appEnhancement
    case finish

    var id: String { rawValue }

    var title: String {
        switch self {
        case .language:
            return AppLocalization.localizedString("Language")
        case .model:
            return AppLocalization.localizedString("Model")
        case .transcription:
            return AppLocalization.localizedString("Transcription")
        case .translation:
            return AppLocalization.localizedString("Translation")
        case .rewrite:
            return AppLocalization.localizedString("Rewrite")
        case .meeting:
            return AppLocalization.localizedString("Meeting")
        case .appEnhancement:
            return AppLocalization.localizedString("App Enhancement")
        case .finish:
            return AppLocalization.localizedString("Finish")
        }
    }

    var subtitle: String {
        switch self {
        case .language:
            return AppLocalization.localizedString("Choose interface language and main language.")
        case .model:
            return AppLocalization.localizedString("Choose one ASR model path and one LLM path for the rest of onboarding.")
        case .transcription:
            return AppLocalization.localizedString("Confirm microphone behavior, shortcut preset, and transcription basics.")
        case .translation:
            return AppLocalization.localizedString("Adjust output behavior and verify the current translation model path.")
        case .rewrite:
            return AppLocalization.localizedString("Understand voice rewrite mode for selected text and prompt-style generation.")
        case .meeting:
            return AppLocalization.localizedString("Review meeting capture, summary, segmentation, and speaker separation settings.")
        case .appEnhancement:
            return AppLocalization.localizedString("Review how app-aware prompt switching works across apps and pages.")
        case .finish:
            return AppLocalization.localizedString("Review your setup, then start using Voxt.")
        }
    }

    var stepNumber: Int {
        (Self.allCases.firstIndex(of: self) ?? 0) + 1
    }

    var previous: OnboardingStep? {
        guard let index = Self.allCases.firstIndex(of: self),
              index > 0 else {
            return nil
        }
        return Self.allCases[index - 1]
    }

    var next: OnboardingStep? {
        guard let index = Self.allCases.firstIndex(of: self),
              index + 1 < Self.allCases.count else {
            return nil
        }
        return Self.allCases[index + 1]
    }
}

enum OnboardingStepStatus: String {
    case ready
    case needsSetup
    case optional
    case done

    var titleKey: LocalizedStringKey {
        switch self {
        case .ready:
            return "Ready"
        case .needsSetup:
            return "Needs Setup"
        case .optional:
            return "Optional"
        case .done:
            return "Done"
        }
    }

    var tint: Color {
        switch self {
        case .ready:
            return .green
        case .needsSetup:
            return .orange
        case .optional:
            return .secondary
        case .done:
            return .accentColor
        }
    }
}

struct OnboardingStepStatusSnapshot {
    var hasModelIssues: Bool
    var hasRecordingMicrophone: Bool
    var hasRecordingPermissions: Bool
    var hasRewriteIssues: Bool
    var appEnhancementEnabled: Bool
}

enum OnboardingStepStatusResolver {
    static func resolve(
        step: OnboardingStep,
        snapshot: OnboardingStepStatusSnapshot
    ) -> OnboardingStepStatus {
        switch step {
        case .language:
            return .ready
        case .model:
            return snapshot.hasModelIssues ? .needsSetup : .ready
        case .transcription:
            return (snapshot.hasRecordingMicrophone && snapshot.hasRecordingPermissions) ? .ready : .needsSetup
        case .translation:
            return .ready
        case .rewrite:
            return snapshot.hasRewriteIssues ? .needsSetup : .ready
        case .meeting:
            return .ready
        case .appEnhancement:
            return .ready
        case .finish:
            return .done
        }
    }
}
