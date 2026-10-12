// VoxtNotificationStoreTests.swift
// Covers notification manifest refresh, persistence, and service failure behavior.

import XCTest
@testable import Voxt

final class VoxtNotificationStoreTests: XCTestCase {
    private final class MockNotificationURLProtocol: URLProtocol, @unchecked Sendable {
        struct Stub {
            let statusCode: Int
            let body: Data
        }

        private static let lock = NSLock()
        private static var stubs: [Stub] = []
        private static var recordedRequests: [URLRequest] = []

        static func install(_ stubs: [Stub]) {
            lock.lock()
            self.stubs = stubs
            recordedRequests = []
            lock.unlock()
        }

        static func requests() -> [URLRequest] {
            lock.lock()
            defer { lock.unlock() }
            return recordedRequests
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            Self.lock.lock()
            guard !Self.stubs.isEmpty else {
                Self.lock.unlock()
                client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
                return
            }
            let stub = Self.stubs.removeFirst()
            Self.recordedRequests.append(request)
            Self.lock.unlock()

            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: stub.statusCode,
                httpVersion: "HTTP/1.1",
                headerFields: nil
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: stub.body)
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}
    }

    @MainActor
    func testRefreshDownloadsOnlyWhenManifestRevisionChanges() async throws {
        let cacheURL = temporaryCacheURL()
        defer { try? FileManager.default.removeItem(at: cacheURL.deletingLastPathComponent()) }
        let session = makeSession()
        let endpoint = URL(string: "https://example.com/api/app/notifications")!

        MockNotificationURLProtocol.install([
            .init(statusCode: 200, body: manifestData(revision: "revision-1", count: 1)),
            .init(statusCode: 200, body: notificationData(revision: "revision-1")),
            .init(statusCode: 200, body: manifestData(revision: "revision-1", count: 1)),
        ])

        let store = VoxtNotificationStore(
            endpointURL: endpoint,
            sessionProvider: { session },
            userDefaults: TestDoubles.makeUserDefaults(),
            cacheURL: cacheURL,
            minimumRefreshInterval: 0
        )

        await store.refresh()
        await store.refresh()

        XCTAssertEqual(store.notifications.map(\.id), ["notification-1"])
        XCTAssertNil(store.lastErrorMessage)
        XCTAssertEqual(MockNotificationURLProtocol.requests().count, 3)
        XCTAssertEqual(MockNotificationURLProtocol.requests()[0].url?.path, "/api/app/notifications/manifest")
        XCTAssertEqual(
            URLComponents(url: MockNotificationURLProtocol.requests()[1].url!, resolvingAgainstBaseURL: false)?
                .queryItems?.first(where: { $0.name == "revision" })?.value,
            "revision-1"
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: cacheURL.path))
    }

    @MainActor
    func testServiceFailureKeepsPreviouslyCachedNotifications() async throws {
        let cacheURL = temporaryCacheURL()
        defer { try? FileManager.default.removeItem(at: cacheURL.deletingLastPathComponent()) }
        let session = makeSession()
        let endpoint = URL(string: "https://example.com/api/app/notifications")!

        MockNotificationURLProtocol.install([
            .init(statusCode: 200, body: manifestData(revision: "revision-1", count: 1)),
            .init(statusCode: 200, body: notificationData(revision: "revision-1")),
        ])
        let initialStore = VoxtNotificationStore(
            endpointURL: endpoint,
            sessionProvider: { session },
            userDefaults: TestDoubles.makeUserDefaults(),
            cacheURL: cacheURL,
            minimumRefreshInterval: 0
        )
        await initialStore.refresh()

        MockNotificationURLProtocol.install([
            .init(statusCode: 503, body: Data()),
        ])
        let restoredStore = VoxtNotificationStore(
            endpointURL: endpoint,
            sessionProvider: { session },
            userDefaults: TestDoubles.makeUserDefaults(),
            cacheURL: cacheURL,
            minimumRefreshInterval: 0
        )

        XCTAssertEqual(restoredStore.notifications.map(\.id), ["notification-1"])
        await restoredStore.refresh(force: true)

        XCTAssertEqual(restoredStore.notifications.map(\.id), ["notification-1"])
        XCTAssertNotNil(restoredStore.lastErrorMessage)
    }

    private func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockNotificationURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    private func temporaryCacheURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathComponent("notifications-cache.json")
    }

    private func manifestData(revision: String, count: Int) -> Data {
        Data("""
        {"schemaVersion":1,"revision":"\(revision)","count":\(count),"latestNotificationID":"notification-1"}
        """.utf8)
    }

    private func notificationData(revision: String) -> Data {
        Data("""
        {
          "revision":"\(revision)",
          "notifications":[{
            "id":"notification-1",
            "title":"Hello",
            "content":"Cached content",
            "coverImage":null,
            "version":"1.0.0",
            "publishedAt":"2026-07-12T08:00:00Z"
          }]
        }
        """.utf8)
    }
}
