// ASRLanguageSelection.swift
// Provides the language a single transcription runs in: the user's settings, or one a Local API
// client names for its request (nil there means detect it).

import Foundation

nonisolated enum ASRLanguageSelection: Equatable, Sendable {
    case userSettings
    case request(String?)
}
