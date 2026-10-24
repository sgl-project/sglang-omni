// Permissions required by the configured recording/text features.
// System audio has no public preflight API: meetings request it by starting
// capture after user intent, rather than treating an unknown state as denied.

import SwiftUI
import AVFoundation
import Speech

enum SettingsPermissionKind: String, CaseIterable, Identifiable {
    case microphone
    case speechRecognition
    case accessibility
    case reminders

    var id: String { rawValue }

    var logKey: String {
        switch self {
        case .microphone: return "mic"
        case .speechRecognition: return "speech"
        case .accessibility: return "accessibility"
        case .reminders: return "reminders"
        }
    }

    var titleKey: LocalizedStringKey {
        switch self {
        case .microphone: return "Microphone Permission"
        case .speechRecognition: return "Speech Recognition Permission"
        case .accessibility: return "Accessibility Permission"
        case .reminders: return "Reminders Permission"
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
        case .reminders:
            return "Required to sync Voxt notes into Apple Reminders."
        }
    }
}

struct SettingsPermissionRequirementContext {
    let selectedEngine: TranscriptionEngine
    let featureSettings: FeatureSettings?
}

enum SettingsPermissionRequirementResolver {
    static func requirementContext(
        selectedEngine: TranscriptionEngine,
        featureSettings: FeatureSettings
    ) -> SettingsPermissionRequirementContext {
        SettingsPermissionRequirementContext(
            selectedEngine: selectedEngine,
            featureSettings: featureSettings
        )
    }

    static func sidebarRequirementContext(
        selectedEngine: TranscriptionEngine,
        featureSettings: FeatureSettings
    ) -> SettingsPermissionRequirementContext {
        requirementContext(selectedEngine: selectedEngine, featureSettings: featureSettings)
    }

    static func requiredPermissions(
        context: SettingsPermissionRequirementContext
    ) -> [SettingsPermissionKind] {
        var permissions: [SettingsPermissionKind] = [.microphone, .accessibility]
        let settings = context.featureSettings
        var featureSelections = [settings?.transcription.asrSelectionID.asrSelection]
        if settings?.availability.translationEnabled == true {
            featureSelections.append(settings?.translation.asrSelectionID.asrSelection)
        }
        if settings?.availability.rewriteEnabled == true {
            featureSelections.append(settings?.rewrite.asrSelectionID.asrSelection)
        }
        if settings?.availability.meetingEnabled == true || settings?.availability.filesEnabled == true {
            featureSelections.append(settings?.meeting.asrSelectionID.asrSelection)
        }
        let needsSpeechRecognition = context.selectedEngine == .dictation || featureSelections.contains { selection in
            if case .dictation = selection { return true }
            return false
        }
        if needsSpeechRecognition {
            permissions.append(.speechRecognition)
        }
        if settings?.transcription.notes.enabled == true,
           settings?.transcription.notes.remindersSync.enabled == true {
            permissions.append(.reminders)
        }
        return permissions
    }

    static func hasMissingPermissions(context: SettingsPermissionRequirementContext) -> Bool {
        requiredPermissions(context: context)
            .contains { !SettingsPermissionGrantResolver.isGranted($0) }
    }
}

enum SettingsPermissionGrantResolver {
    static func isGranted(_ permission: SettingsPermissionKind) -> Bool {
        switch permission {
        case .microphone:
            return AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        case .speechRecognition:
            return SFSpeechRecognizer.authorizationStatus() == .authorized
        case .accessibility:
            return AccessibilityPermissionManager.isTrusted()
        case .reminders:
            return RemindersPermissionManager.isAuthorized()
        }
    }
}
