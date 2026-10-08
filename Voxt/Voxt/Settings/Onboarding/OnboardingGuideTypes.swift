// OnboardingGuideTypes.swift
// The six-step, task-based setup guide. Raw values remain stable for saved progress.

import Foundation

enum OnboardingGuidePhase: String, CaseIterable, Identifiable {
    case basics, workflows, finish

    var id: String { rawValue }

    var title: String {
        switch self {
        case .basics: return AppLocalization.localizedString("Basics")
        case .workflows: return AppLocalization.localizedString("Workflows")
        case .finish: return AppLocalization.localizedString("Finish")
        }
    }
}

enum OnboardingGuideStep: String, CaseIterable, Identifiable {
    case permissions
    case models
    case transcriptionShortcut
    case translationShortcut
    case translationSelection
    case finish

    var id: String { rawValue }

    var phase: OnboardingGuidePhase {
        switch self {
        case .permissions, .models: return .basics
        case .transcriptionShortcut, .translationShortcut, .translationSelection: return .workflows
        case .finish: return .finish
        }
    }

    var title: String {
        switch self {
        case .permissions: return AppLocalization.localizedString("Get Permissions")
        case .models: return AppLocalization.localizedString("Choose Models")
        case .transcriptionShortcut: return AppLocalization.localizedString("Try Voice Input")
        case .translationShortcut: return AppLocalization.localizedString("Try Voice Translation")
        case .translationSelection: return AppLocalization.localizedString("Translate Selected Text")
        case .finish: return AppLocalization.localizedString("Explore More")
        }
    }

    var subtitle: String {
        switch self {
        case .permissions:
            return AppLocalization.localizedString("Allow Voxt to hear you, read shortcuts, and insert text into active apps.")
        case .models:
            return AppLocalization.localizedString("One speech model is enough to start. Translation is optional.")
        case .transcriptionShortcut:
            return AppLocalization.localizedString("Read the sample aloud and watch your words appear. Your own words work too.")
        case .translationShortcut:
            return AppLocalization.localizedString("Say it in your language. Let Voxt type it in another.")
        case .translationSelection:
            return AppLocalization.localizedString("Select the sample, then use the same translation shortcut.")
        case .finish:
            return AppLocalization.localizedString("You know the basics. Discover these features whenever you need them.")
        }
    }

    var stepNumber: Int { (Self.allCases.firstIndex(of: self) ?? 0) + 1 }

    var previous: Self? {
        guard let index = Self.allCases.firstIndex(of: self), index > 0 else { return nil }
        return Self.allCases[index - 1]
    }

    var next: Self? {
        guard let index = Self.allCases.firstIndex(of: self), index + 1 < Self.allCases.count else { return nil }
        return Self.allCases[index + 1]
    }

    var isPractice: Bool { phase == .workflows }

    var sidebarIconKind: SettingsSidebarIconKind {
        switch self {
        case .permissions: return .permissions
        case .models: return .model
        case .transcriptionShortcut: return .transcription
        case .translationShortcut, .translationSelection: return .translation
        case .finish: return .home
        }
    }

    static func restored(from rawValue: String) -> Self? {
        switch rawValue {
        case "microphone", "language": return .permissions
        case "model": return .models
        case "transcriptionEnhancement", "transcription": return .transcriptionShortcut
        case "translation": return .translationShortcut
        case "rewriteShortcut", "rewriteSelection", "rewrite", "appEnhancement", "meeting": return .finish
        default: return Self(rawValue: rawValue)
        }
    }
}

enum OnboardingGuideModelFocus {
    case local
    case remote
}
