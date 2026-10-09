// VoxtRuntimeEnvironment.swift
// Provides runtime environment checks for logging behavior.

import Foundation

enum VoxtRuntimeEnvironment {
    nonisolated static var isRunningUnitTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }
}
