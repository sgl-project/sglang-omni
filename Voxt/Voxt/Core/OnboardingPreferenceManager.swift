// OnboardingPreferenceManager.swift
// Provides Onboarding Preference Manager for core app behavior.

import Foundation

enum OnboardingPreferenceManager {
    static func shouldPresentOnLaunch(
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default,
        bundleIdentifier: String? = Bundle.main.bundleIdentifier
    ) -> Bool {
        !resolvedCompletionState(
            defaults: defaults,
            fileManager: fileManager,
            bundleIdentifier: bundleIdentifier
        )
    }

    @discardableResult
    static func resolvedCompletionState(
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default,
        bundleIdentifier: String? = Bundle.main.bundleIdentifier
    ) -> Bool {
        if defaults.object(forKey: AppPreferenceKey.onboardingCompleted) != nil {
            return defaults.bool(forKey: AppPreferenceKey.onboardingCompleted)
        }

        let completed = hasExistingUserData(
            defaults: defaults,
            fileManager: fileManager,
            bundleIdentifier: bundleIdentifier
        )
        defaults.set(completed, forKey: AppPreferenceKey.onboardingCompleted)
        return completed
    }

    static func markCompleted(defaults: UserDefaults = .standard) {
        defaults.set(true, forKey: AppPreferenceKey.onboardingCompleted)
        defaults.removeObject(forKey: AppPreferenceKey.onboardingLastStepID)
    }

    static func saveLastStep(
        _ step: OnboardingStep,
        defaults: UserDefaults = .standard
    ) {
        defaults.set(step.rawValue, forKey: AppPreferenceKey.onboardingLastStepID)
    }

    static func saveLastGuideStep(
        _ step: OnboardingGuideStep,
        defaults: UserDefaults = .standard
    ) {
        defaults.set(step.rawValue, forKey: AppPreferenceKey.onboardingLastStepID)
    }

    static func savedLastStep(defaults: UserDefaults = .standard) -> OnboardingStep? {
        guard let rawValue = defaults.string(forKey: AppPreferenceKey.onboardingLastStepID) else {
            return nil
        }
        return OnboardingStep(rawValue: rawValue)
    }

    static func savedLastGuideStep(defaults: UserDefaults = .standard) -> OnboardingGuideStep? {
        guard let rawValue = defaults.string(forKey: AppPreferenceKey.onboardingLastStepID) else {
            return nil
        }
        return OnboardingGuideStep.restored(from: rawValue)
    }

    static func hasExistingUserData(
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default,
        bundleIdentifier: String? = Bundle.main.bundleIdentifier
    ) -> Bool {
        if let bundleIdentifier,
           let domain = defaults.persistentDomain(forName: bundleIdentifier) {
            let meaningfulKeys = domain.keys.filter {
                $0 != AppPreferenceKey.onboardingCompleted &&
                $0 != AppPreferenceKey.onboardingLastStepID
            }
            if !meaningfulKeys.isEmpty {
                return true
            }
        }

        guard let appSupportDirectory = try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        ) else {
            return false
        }

        let voxtDirectory = appSupportDirectory.appendingPathComponent("Voxt", isDirectory: true)
        return fileManager.fileExists(atPath: voxtDirectory.path)
    }
}
