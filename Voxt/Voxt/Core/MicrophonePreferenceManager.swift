// MicrophonePreferenceManager.swift
// Provides Microphone Preference Manager for core app behavior.

import Foundation
import CoreAudio

enum MicrophonePreferenceManager {
    static func syncState(
        defaults: UserDefaults = .standard,
        availableDevices: [AudioInputDevice],
        previousAvailableUIDs: Set<String>? = nil,
        lockedActiveUID: String? = nil
    ) -> MicrophoneResolvedState {
        migrateLegacySelectionIfNeeded(defaults: defaults, availableDevices: availableDevices)

        let priorityUIDs = normalizedPriorityUIDs(from: defaults)
        let currentActiveUID = defaults.string(forKey: AppPreferenceKey.activeInputDeviceUID)
        let autoSwitchEnabled = autoSwitchEnabled(defaults: defaults)
        let defaultUID = AudioInputDeviceManager.defaultInputDeviceUID(from: availableDevices)
        let availableByUID = Dictionary(uniqueKeysWithValues: availableDevices.map { ($0.uid, $0) })
        let activeUID = resolvedActiveUID(
            currentActiveUID: currentActiveUID,
            priorityUIDs: priorityUIDs,
            autoSwitchEnabled: autoSwitchEnabled,
            availableDevices: availableDevices,
            previousAvailableUIDs: previousAvailableUIDs,
            defaultUID: defaultUID,
            lockedActiveUID: lockedActiveUID
        )
        let activeDevice = activeUID.flatMap { availableByUID[$0] }

        if let activeDevice {
            defaults.set(activeDevice.uid, forKey: AppPreferenceKey.activeInputDeviceUID)
        } else {
            defaults.removeObject(forKey: AppPreferenceKey.activeInputDeviceUID)
        }

        let records = syncedTrackedRecords(
            priorityUIDs: priorityUIDs,
            availableDevices: availableDevices,
            defaults: defaults
        )
        saveTrackedRecords(records, defaults: defaults)
        defaults.set(priorityUIDs, forKey: AppPreferenceKey.microphonePriorityUIDs)

        let addedUIDs = newlyAvailableUIDs(
            availableDevices: availableDevices,
            previousAvailableUIDs: previousAvailableUIDs
        )
        let removedUIDs = removedUIDs(
            availableDevices: availableDevices,
            previousAvailableUIDs: previousAvailableUIDs
        )
        if shouldLogSyncResolution(
            currentActiveUID: currentActiveUID,
            resolvedUID: activeUID,
            addedUIDs: addedUIDs,
            removedUIDs: removedUIDs
        ) {
            VoxtLog.audio(
                """
                Microphone sync resolved. current=\(currentActiveUID ?? "none"), resolved=\(activeUID ?? "none"), default=\(defaultUID ?? "none"), autoSwitch=\(autoSwitchEnabled), added=\(formatUIDList(addedUIDs)), removed=\(formatUIDList(removedUIDs)), priority=\(formatUIDList(priorityUIDs)), available=\(formatDevices(availableDevices))
                """
            )
        }

        return resolvedState(
            activeDevice: activeDevice,
            availableDevices: availableDevices,
            priorityUIDs: priorityUIDs,
            records: records,
            autoSwitchEnabled: autoSwitchEnabled
        )
    }

    static func setFocusedDevice(
        uid: String,
        defaults: UserDefaults = .standard,
        availableDevices: [AudioInputDevice]
    ) -> MicrophoneResolvedState {
        defaults.set(uid, forKey: AppPreferenceKey.activeInputDeviceUID)
        return syncState(defaults: defaults, availableDevices: availableDevices)
    }

    static func setAutoSwitchEnabled(
        _ enabled: Bool,
        defaults: UserDefaults = .standard,
        availableDevices: [AudioInputDevice]
    ) -> MicrophoneResolvedState {
        defaults.set(enabled, forKey: AppPreferenceKey.microphoneAutoSwitchEnabled)
        return syncState(defaults: defaults, availableDevices: availableDevices)
    }

    static func reorderPriority(
        orderedUIDs: [String],
        defaults: UserDefaults = .standard,
        availableDevices: [AudioInputDevice]
    ) -> MicrophoneResolvedState {
        persistPriorityUIDs(orderedUIDs, defaults: defaults, availableDevices: availableDevices)
        return syncState(defaults: defaults, availableDevices: availableDevices)
    }

    static func activeInputDeviceID(
        defaults: UserDefaults = .standard,
        availableDevices: [AudioInputDevice]
    ) -> AudioDeviceID? {
        let activeUID = defaults.string(forKey: AppPreferenceKey.activeInputDeviceUID)
        return availableDevices.first(where: { $0.uid == activeUID })?.id
    }

    static func activeInputDeviceUID(defaults: UserDefaults = .standard) -> String? {
        defaults.string(forKey: AppPreferenceKey.activeInputDeviceUID)
    }

    static func priorityUIDs(defaults: UserDefaults = .standard) -> [String] {
        normalizedPriorityUIDs(from: defaults)
    }

    static func autoSwitchEnabled(defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: AppPreferenceKey.microphoneAutoSwitchEnabled) as? Bool ?? true
    }

    static func trackedRecords(defaults: UserDefaults = .standard) -> [TrackedMicrophoneRecord] {
        guard let raw = defaults.string(forKey: AppPreferenceKey.trackedMicrophoneRecords),
              let data = raw.data(using: .utf8),
              let decoded = try? JSONDecoder().decode([TrackedMicrophoneRecord].self, from: data)
        else {
            return []
        }
        return decoded
    }

    private static func resolvedState(
        activeDevice: AudioInputDevice?,
        availableDevices: [AudioInputDevice],
        priorityUIDs: [String],
        records: [TrackedMicrophoneRecord],
        autoSwitchEnabled: Bool
    ) -> MicrophoneResolvedState {
        let availableByUID = Dictionary(uniqueKeysWithValues: availableDevices.map { ($0.uid, $0) })
        let trackedEntries: [MicrophoneDisplayEntry] = records.map { record in
            let device = availableByUID[record.uid]
            let status = resolvedStatus(
                uid: record.uid,
                device: device,
                activeUID: activeDevice?.uid
            )
            return MicrophoneDisplayEntry(
                uid: record.uid,
                name: device?.name ?? record.lastKnownName,
                device: device,
                isTracked: true,
                status: status
            )
        }

        let trackedUIDSet = Set(records.map(\.uid))
        let untrackedEntries = availableDevices
            .filter { !trackedUIDSet.contains($0.uid) }
            .map { device in
                MicrophoneDisplayEntry(
                    uid: device.uid,
                    name: device.name,
                    device: device,
                    isTracked: false,
                    status: resolvedStatus(
                        uid: device.uid,
                        device: device,
                        activeUID: activeDevice?.uid
                    )
                )
            }

        return MicrophoneResolvedState(
            activeDevice: activeDevice,
            entries: trackedEntries + untrackedEntries,
            priorityUIDs: priorityUIDs,
            activeUID: activeDevice?.uid,
            autoSwitchEnabled: autoSwitchEnabled
        )
    }

    private static func resolvedStatus(
        uid: String,
        device: AudioInputDevice?,
        activeUID: String?
    ) -> MicrophoneDisplayStatus {
        guard device != nil else { return .offline }
        return uid == activeUID ? .inUse : .available
    }

    private static func persistPriorityUIDs(
        _ orderedUIDs: [String],
        defaults: UserDefaults,
        availableDevices: [AudioInputDevice]
    ) {
        let normalizedUIDs = uniqueUIDs(from: orderedUIDs)
        defaults.set(normalizedUIDs, forKey: AppPreferenceKey.microphonePriorityUIDs)
        let records = syncedTrackedRecords(
            priorityUIDs: normalizedUIDs,
            availableDevices: availableDevices,
            defaults: defaults
        )
        saveTrackedRecords(records, defaults: defaults)
    }

    private static func syncedTrackedRecords(
        priorityUIDs: [String],
        availableDevices: [AudioInputDevice],
        defaults: UserDefaults
    ) -> [TrackedMicrophoneRecord] {
        let availableByUID = Dictionary(uniqueKeysWithValues: availableDevices.map { ($0.uid, $0) })
        let existingByUID = Dictionary(uniqueKeysWithValues: trackedRecords(defaults: defaults).map { ($0.uid, $0) })

        return priorityUIDs.map { uid in
            if let device = availableByUID[uid] {
                return TrackedMicrophoneRecord(uid: uid, lastKnownName: device.name)
            }
            if let existing = existingByUID[uid] {
                return existing
            }
            return TrackedMicrophoneRecord(uid: uid, lastKnownName: uid)
        }
    }

    private static func normalizedPriorityUIDs(from defaults: UserDefaults) -> [String] {
        uniqueUIDs(from: defaults.stringArray(forKey: AppPreferenceKey.microphonePriorityUIDs) ?? [])
    }

    private static func uniqueUIDs(from values: [String]) -> [String] {
        var seen = Set<String>()
        return values.compactMap { raw in
            let uid = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !uid.isEmpty, seen.insert(uid).inserted else { return nil }
            return uid
        }
    }

    private static func saveTrackedRecords(_ records: [TrackedMicrophoneRecord], defaults: UserDefaults) {
        guard let data = try? JSONEncoder().encode(records),
              let raw = String(data: data, encoding: .utf8)
        else {
            defaults.removeObject(forKey: AppPreferenceKey.trackedMicrophoneRecords)
            return
        }
        defaults.set(raw, forKey: AppPreferenceKey.trackedMicrophoneRecords)
    }

    private static func migrateLegacySelectionIfNeeded(
        defaults: UserDefaults,
        availableDevices: [AudioInputDevice]
    ) {
        if defaults.object(forKey: AppPreferenceKey.microphoneAutoSwitchEnabled) == nil {
            defaults.set(true, forKey: AppPreferenceKey.microphoneAutoSwitchEnabled)
        }

        if defaults.object(forKey: AppPreferenceKey.activeInputDeviceUID) == nil,
           let legacyUID = defaults.string(forKey: "manualSelectedInputDeviceUID")?.trimmingCharacters(in: .whitespacesAndNewlines),
           !legacyUID.isEmpty {
            defaults.set(legacyUID, forKey: AppPreferenceKey.activeInputDeviceUID)
            defaults.removeObject(forKey: "manualSelectedInputDeviceUID")
            return
        }

        guard defaults.object(forKey: AppPreferenceKey.activeInputDeviceUID) == nil,
              let legacyValue = defaults.object(forKey: AppPreferenceKey.selectedInputDeviceID) as? Int,
              legacyValue > 0,
              let device = availableDevices.first(where: { Int($0.id) == legacyValue })
        else {
            return
        }

        defaults.set(device.uid, forKey: AppPreferenceKey.activeInputDeviceUID)
    }

    private static func resolvedActiveUID(
        currentActiveUID: String?,
        priorityUIDs: [String],
        autoSwitchEnabled: Bool,
        availableDevices: [AudioInputDevice],
        previousAvailableUIDs: Set<String>?,
        defaultUID: String?,
        lockedActiveUID: String?
    ) -> String? {
        let availableByUID = Dictionary(uniqueKeysWithValues: availableDevices.map { ($0.uid, $0) })

        if let lockedActiveUID,
           availableByUID[lockedActiveUID] != nil {
            if currentActiveUID != lockedActiveUID {
                VoxtLog.audio(
                    "Microphone selection preserved during active session. previous=\(currentActiveUID ?? "none"), locked=\(lockedActiveUID)"
                )
            }
            return lockedActiveUID
        }

        if let currentActiveUID,
           availableByUID[currentActiveUID] != nil {
            guard autoSwitchEnabled,
                  let previousAvailableUIDs
            else {
                return currentActiveUID
            }

            let currentRank = priorityRank(for: currentActiveUID, priorityUIDs: priorityUIDs)
            if let promotedUID = priorityUIDs.first(where: { uid in
                guard availableByUID[uid] != nil else { return false }
                guard !previousAvailableUIDs.contains(uid) else { return false }
                return priorityRank(for: uid, priorityUIDs: priorityUIDs) < currentRank
            }) {
                VoxtLog.audio(
                    "Microphone auto selection promoted higher-priority device. previous=\(currentActiveUID), promoted=\(promotedUID), previousRank=\(currentRank), promotedRank=\(priorityRank(for: promotedUID, priorityUIDs: priorityUIDs))"
                )
                return promotedUID
            }

            return currentActiveUID
        }

        if let prioritizedUID = priorityUIDs.first(where: { availableByUID[$0] != nil }) {
            VoxtLog.audio("Microphone auto selection chose prioritized device. uid=\(prioritizedUID)")
            return prioritizedUID
        }

        if let defaultUID, availableByUID[defaultUID] != nil {
            VoxtLog.audio("Microphone auto selection fell back to system default. uid=\(defaultUID)")
            return defaultUID
        }

        let fallbackUID = availableDevices.first?.uid
        VoxtLog.audio("Microphone auto selection fell back to first available device. uid=\(fallbackUID ?? "none")")
        return fallbackUID
    }

    private static func priorityRank(for uid: String, priorityUIDs: [String]) -> Int {
        priorityUIDs.firstIndex(of: uid) ?? Int.max
    }

    private static func newlyAvailableUIDs(
        availableDevices: [AudioInputDevice],
        previousAvailableUIDs: Set<String>?
    ) -> [String] {
        guard let previousAvailableUIDs else { return [] }
        return availableDevices.map(\.uid).filter { !previousAvailableUIDs.contains($0) }
    }

    private static func removedUIDs(
        availableDevices: [AudioInputDevice],
        previousAvailableUIDs: Set<String>?
    ) -> [String] {
        guard let previousAvailableUIDs else { return [] }
        let currentUIDs = Set(availableDevices.map(\.uid))
        return previousAvailableUIDs.filter { !currentUIDs.contains($0) }.sorted()
    }

    private static func formatUIDList(_ uids: [String]) -> String {
        uids.isEmpty ? "[]" : "[\(uids.joined(separator: ", "))]"
    }

    private static func formatDevices(_ devices: [AudioInputDevice]) -> String {
        guard !devices.isEmpty else { return "[]" }
        return devices
            .map { "\($0.name){uid=\($0.uid),id=\($0.id)}" }
            .joined(separator: ", ")
    }

    private static func shouldLogSyncResolution(
        currentActiveUID: String?,
        resolvedUID: String?,
        addedUIDs: [String],
        removedUIDs: [String]
    ) -> Bool {
        currentActiveUID != resolvedUID || !addedUIDs.isEmpty || !removedUIDs.isEmpty
    }
}
