// RemindersPermissionManager.swift
// Provides Reminders Permission Manager for permissions and secure storage.

import Foundation
@preconcurrency import EventKit

enum RemindersAuthorizationState: String, Codable, Hashable, Sendable {
    case notDetermined
    case restricted
    case denied
    case authorized
}

struct RemindersListDescriptor: Identifiable, Hashable, Sendable {
    let identifier: String
    let title: String
    let sourceTitle: String

    var id: String { identifier }

    var displayTitle: String {
        let trimmedSourceTitle = sourceTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedSourceTitle.isEmpty, trimmedSourceTitle != title else {
            return title
        }
        return "\(title) · \(trimmedSourceTitle)"
    }
}

enum RemindersPermissionManager {
    // EventKit keeps an XPC connection to calaccessd for each event store.
    // Reuse one store for the lifetime of the app instead of creating one
    // every time permissions or reminder lists are queried.
    static let sharedEventStore = EKEventStore()

    static func authorizationState() -> RemindersAuthorizationState {
        switch EKEventStore.authorizationStatus(for: .reminder) {
        case .fullAccess, .authorized:
            return .authorized
        case .notDetermined:
            return .notDetermined
        case .restricted:
            return .restricted
        case .denied, .writeOnly:
            return .denied
        @unknown default:
            return .denied
        }
    }

    static func isAuthorized() -> Bool {
        authorizationState() == .authorized
    }

    static func requestAccess(completion: @escaping (Bool) -> Void) {
        let eventStore = sharedEventStore
        if #available(macOS 14.0, *) {
            eventStore.requestFullAccessToReminders { granted, _ in
                completion(granted)
            }
        } else {
            eventStore.requestAccess(to: .reminder) { granted, _ in
                completion(granted)
            }
        }
    }

    static func writableLists(eventStore: EKEventStore = sharedEventStore) -> [RemindersListDescriptor] {
        eventStore.calendars(for: .reminder)
            .filter(\.allowsContentModifications)
            .map {
                RemindersListDescriptor(
                    identifier: $0.calendarIdentifier,
                    title: $0.title.trimmingCharacters(in: .whitespacesAndNewlines),
                    sourceTitle: $0.source.title.trimmingCharacters(in: .whitespacesAndNewlines)
                )
            }
            .sorted { lhs, rhs in
                lhs.displayTitle.localizedCaseInsensitiveCompare(rhs.displayTitle) == .orderedAscending
            }
    }
}
