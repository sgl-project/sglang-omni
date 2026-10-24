// MicrophonePreferenceModels.swift
// Provides Microphone Preference Models for core app behavior.

import Foundation
import CoreAudio

enum MicrophoneDisplayStatus: String, Sendable {
    case inUse
    case available
    case offline

    var titleKey: String {
        switch self {
        case .inUse:
            return "Microphone Status In Use"
        case .available:
            return "Microphone Status Available"
        case .offline:
            return "Microphone Status Offline"
        }
    }
}

struct TrackedMicrophoneRecord: Codable, Hashable, Identifiable, Sendable {
    let uid: String
    var lastKnownName: String

    var id: String { uid }
}

struct MicrophoneDisplayEntry: Identifiable, Hashable, Sendable {
    let uid: String
    let name: String
    let device: AudioInputDevice?
    let isTracked: Bool
    let status: MicrophoneDisplayStatus

    var id: String { uid }
    var deviceID: AudioDeviceID? { device?.id }
    var isAvailable: Bool { device != nil }
    var isActive: Bool { status == .inUse }
}

struct MicrophoneResolvedState: Sendable {
    let activeDevice: AudioInputDevice?
    let entries: [MicrophoneDisplayEntry]
    let priorityUIDs: [String]
    let activeUID: String?
    let autoSwitchEnabled: Bool

    static let empty = MicrophoneResolvedState(
        activeDevice: nil,
        entries: [],
        priorityUIDs: [],
        activeUID: nil,
        autoSwitchEnabled: true
    )

    var hasAvailableDevices: Bool {
        entries.contains(where: \.isAvailable)
    }

    var hasTrackedDevices: Bool {
        entries.contains(where: \.isTracked)
    }
}
