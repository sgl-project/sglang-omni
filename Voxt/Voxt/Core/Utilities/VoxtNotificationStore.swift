// VoxtNotificationStore.swift
// Provides resilient app notification loading and caching for the home screen.

import Foundation
import Combine

struct VoxtAppNotification: Identifiable, Equatable, Codable {
    let id: String
    let title: String
    let content: String
    let coverImageURL: URL?
    let version: String
    let publishedAt: Date
}

@MainActor
final class VoxtNotificationStore: ObservableObject {
    @Published private(set) var notifications: [VoxtAppNotification]
    @Published private(set) var isLoading = false
    @Published private(set) var lastErrorMessage: String?
    @Published private(set) var viewedNotificationIDs: Set<String>

    private let endpointURL: URL
    private let sessionProvider: @MainActor () -> URLSession
    private let userDefaults: UserDefaults
    private let cacheURL: URL?
    private let now: () -> Date
    private let minimumRefreshInterval: TimeInterval
    private var revision: String?
    private var lastRefreshAttemptAt: Date?

    private static let viewedNotificationIDsKey = "VoxtNotificationStore.viewedNotificationIDs"

    init(
        endpointURL: URL = URL(string: "https://voxt.actnow.dev/api/app/notifications")!,
        sessionProvider: @escaping @MainActor () -> URLSession = { VoxtNetworkSession.active },
        userDefaults: UserDefaults = .standard,
        cacheURL: URL? = nil,
        minimumRefreshInterval: TimeInterval = 10 * 60,
        now: @escaping () -> Date = Date.init
    ) {
        self.endpointURL = endpointURL
        self.sessionProvider = sessionProvider
        self.userDefaults = userDefaults
        self.cacheURL = cacheURL ?? Self.defaultCacheURL()
        self.minimumRefreshInterval = minimumRefreshInterval
        self.now = now
        self.viewedNotificationIDs = Set(userDefaults.stringArray(forKey: Self.viewedNotificationIDsKey) ?? [])

        let cachedPayload = self.cacheURL.flatMap(Self.loadCache(from:))
        self.notifications = cachedPayload?.notifications ?? []
        self.revision = cachedPayload?.revision
    }

    var latestNotification: VoxtAppNotification? {
        notifications.first
    }

    var hasUnreadLatestNotification: Bool {
        guard let latestNotification else { return false }
        return !viewedNotificationIDs.contains(latestNotification.id)
    }

    func markAsViewed(_ notification: VoxtAppNotification?) {
        guard let notification else { return }
        guard !viewedNotificationIDs.contains(notification.id) else { return }
        var updatedViewedNotificationIDs = viewedNotificationIDs
        updatedViewedNotificationIDs.insert(notification.id)
        viewedNotificationIDs = updatedViewedNotificationIDs
        userDefaults.set(Array(viewedNotificationIDs), forKey: Self.viewedNotificationIDsKey)
    }

    func refreshIfNeeded() async {
        await refresh()
    }

    func refresh(force: Bool = false) async {
        guard !isLoading else { return }

        let attemptDate = now()
        if !force,
           let lastRefreshAttemptAt,
           attemptDate.timeIntervalSince(lastRefreshAttemptAt) < minimumRefreshInterval {
            return
        }

        lastRefreshAttemptAt = attemptDate
        isLoading = true
        defer { isLoading = false }

        do {
            let session = sessionProvider()
            let manifest = try await Self.fetchManifest(endpointURL: endpointURL, session: session)
            guard manifest.schemaVersion == 1 else {
                throw NotificationError.unsupportedManifest
            }

            if revision != manifest.revision || (notifications.isEmpty && manifest.count > 0) {
                let payload = try await Self.fetchNotifications(
                    endpointURL: endpointURL,
                    revision: manifest.revision,
                    session: session
                )
                guard payload.revision == manifest.revision else {
                    throw NotificationError.revisionMismatch
                }
                notifications = payload.notifications.sorted { $0.publishedAt > $1.publishedAt }
                revision = payload.revision
                persistCache()
            }

            lastErrorMessage = nil
        } catch {
            lastErrorMessage = AppLocalization.localizedString(
                notifications.isEmpty
                    ? "Notification service is temporarily unavailable. Please try again later."
                    : "Notification service is temporarily unavailable. Showing previously loaded notifications."
            )
            if notifications.isEmpty {
                VoxtLog.warning("Failed to refresh app notifications. error=\(error.localizedDescription)")
            } else {
                VoxtLog.info(
                    "App notification refresh failed; using cached notifications. error=\(error.localizedDescription)",
                    verbose: true
                )
            }
        }
    }

    private func persistCache() {
        guard let cacheURL, let revision else { return }
        do {
            let payload = CachedPayload(revision: revision, notifications: notifications)
            let data = try Self.makeEncoder().encode(payload)
            try FileManager.default.createDirectory(
                at: cacheURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: cacheURL, options: [.atomic])
        } catch {
            VoxtLog.warning("Failed to persist app notification cache. error=\(error.localizedDescription)")
        }
    }

    private static func fetchManifest(endpointURL: URL, session: URLSession) async throws -> ManifestPayload {
        let manifestURL = endpointURL.appendingPathComponent("manifest")
        let data = try await fetchData(from: manifestURL, session: session)
        return try JSONDecoder().decode(ManifestPayload.self, from: data)
    }

    private static func fetchNotifications(
        endpointURL: URL,
        revision: String,
        session: URLSession
    ) async throws -> ResponsePayload {
        guard var components = URLComponents(url: endpointURL, resolvingAgainstBaseURL: false) else {
            throw URLError(.badURL)
        }
        components.queryItems = (components.queryItems ?? []) + [URLQueryItem(name: "revision", value: revision)]
        guard let requestURL = components.url else { throw URLError(.badURL) }

        let data = try await fetchData(from: requestURL, session: session)
        return try makeDecoder().decode(ResponsePayload.self, from: data)
    }

    private static func fetchData(from url: URL, session: URLSession) async throws -> Data {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 12
        request.cachePolicy = .useProtocolCachePolicy

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode)
        else {
            throw URLError(.badServerResponse)
        }
        return data
    }

    private static func loadCache(from url: URL) -> CachedPayload? {
        do {
            let data = try Data(contentsOf: url)
            return try makeDecoder().decode(CachedPayload.self, from: data)
        } catch CocoaError.fileReadNoSuchFile {
            return nil
        } catch {
            VoxtLog.warning("Failed to load app notification cache. error=\(error.localizedDescription)")
            return nil
        }
    }

    private static func defaultCacheURL() -> URL? {
        guard let appSupport = try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ) else {
            return nil
        }
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("notifications-cache.json")
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(String.self)
            let iso8601Formatter = ISO8601DateFormatter()
            iso8601Formatter.formatOptions = [.withInternetDateTime]
            let fractionalISO8601Formatter = ISO8601DateFormatter()
            fractionalISO8601Formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let date = fractionalISO8601Formatter.date(from: value) ?? iso8601Formatter.date(from: value) {
                return date
            }
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Invalid ISO8601 date: \(value)"
            )
        }
        return decoder
    }
}

private enum NotificationError: Error {
    case unsupportedManifest
    case revisionMismatch
}

private struct ManifestPayload: Decodable {
    let schemaVersion: Int
    let revision: String
    let count: Int
}

private struct ResponsePayload: Decodable {
    let revision: String
    let remoteNotifications: [RemoteNotification]

    var notifications: [VoxtAppNotification] {
        remoteNotifications.map(\.notification)
    }

    private enum CodingKeys: String, CodingKey {
        case revision
        case remoteNotifications = "notifications"
    }
}

private struct CachedPayload: Codable {
    let revision: String
    let notifications: [VoxtAppNotification]
}

private struct RemoteNotification: Decodable {
    let id: String
    let title: String
    let content: String
    let coverImage: URL?
    let version: String
    let publishedAt: Date

    var notification: VoxtAppNotification {
        VoxtAppNotification(
            id: id,
            title: title,
            content: content,
            coverImageURL: coverImage,
            version: version,
            publishedAt: publishedAt
        )
    }
}
