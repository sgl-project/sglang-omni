// CustomLLMModelDownloadSupportTests.swift
// Provides Custom LLMModel Download Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class CustomLLMModelDownloadSupportTests: XCTestCase {
    private let resumableTestPolicy = ResumableDownloadPolicy(
        resumeThresholdBytes: 1,
        stallTimeout: .seconds(45),
        stallPollInterval: .seconds(5),
        maxRecoveryAttempts: 5,
        initialBackoffSeconds: 1,
        maxBackoffSeconds: 30
    )

    private final class MockResumableDownloadURLProtocol: URLProtocol, @unchecked Sendable {
        struct Stub {
            let statusCode: Int
            let headers: [String: String]
            let body: Data
        }

        private static let lock = NSLock()
        private static var stubs: [Stub] = []
        private static var recordedRequests: [URLRequest] = []

        static func install(stubs: [Stub]) {
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

        override class func canInit(with request: URLRequest) -> Bool {
            true
        }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest {
            request
        }

        override func startLoading() {
            Self.lock.lock()
            guard !Self.stubs.isEmpty else {
                Self.lock.unlock()
                client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
                return
            }
            let stub = Self.stubs.removeFirst()
            Self.recordedRequests.append(request)
            Self.lock.unlock()

            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: stub.statusCode,
                httpVersion: "HTTP/1.1",
                headerFields: stub.headers
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            if !stub.body.isEmpty {
                client?.urlProtocol(self, didLoad: stub.body)
            }
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}
    }

    func testFallbackHubBaseURLUsesMirrorOnlyForPrimaryHost() {
        let primaryURL = URL(string: "https://huggingface.co")!
        let mirrorURL = URL(string: "https://hf-mirror.com")!

        XCTAssertEqual(
            CustomLLMModelDownloadSupport.fallbackHubBaseURL(
                from: primaryURL,
                mirrorBaseURL: mirrorURL
            ),
            mirrorURL
        )
        XCTAssertNil(
            CustomLLMModelDownloadSupport.fallbackHubBaseURL(
                from: mirrorURL,
                mirrorBaseURL: mirrorURL
            )
        )
    }

    func testInFlightBytesUsesReportedProgressWhenAvailable() {
        let progress = Progress(totalUnitCount: 100)
        progress.completedUnitCount = 42

        let current = ModelDownloadProgress.inFlightBytes(
            progress: progress,
            expectedFileBytes: 1_024,
            startTime: Date().addingTimeInterval(-5)
        )

        XCTAssertEqual(current, 42)
    }

    func testInFlightBytesEstimatesProgressWhenProgressStillZero() {
        let progress = Progress(totalUnitCount: 100)
        progress.completedUnitCount = 0

        let current = ModelDownloadProgress.inFlightBytes(
            progress: progress,
            expectedFileBytes: 20 * 1024 * 1024,
            startTime: Date().addingTimeInterval(-10)
        )

        XCTAssertGreaterThan(current, 0)
        XCTAssertLessThan(current, Int64(Double(20 * 1024 * 1024) * 0.95) + 1)
    }

    func testCanReuseExistingDownloadRequiresExactExpectedSize() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let fileURL = directory.appendingPathComponent("weights.safetensors")
        try Data(repeating: 0x1, count: 32).write(to: fileURL)

        XCTAssertTrue(
            MLXModelDownloadSupport.canReuseExistingDownload(
                at: fileURL,
                expectedSize: 32,
                fileManager: .default
            )
        )
        XCTAssertFalse(
            MLXModelDownloadSupport.canReuseExistingDownload(
                at: fileURL,
                expectedSize: 31,
                fileManager: .default
            )
        )
    }

    func testCanReuseExistingDownloadAllowsUnknownSizeWhenFileIsNonEmpty() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let fileURL = directory.appendingPathComponent("config.json")
        try Data("{}".utf8).write(to: fileURL)

        XCTAssertTrue(
            MLXModelDownloadSupport.canReuseExistingDownload(
                at: fileURL,
                expectedSize: nil,
                fileManager: .default
            )
        )
    }

    func testResumableDownloadSendsRangeAndIfRangeForExistingPartialFile() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = directory.appendingPathComponent("model.safetensors")
        let partURL = destination.appendingPathExtension("part")
        let sidecarURL = partURL.appendingPathExtension("json")
        try Data("hello".utf8).write(to: partURL)
        try JSONEncoder().encode(
            ResumableDownloadState(
                relativePath: "model.safetensors",
                sourceURL: "https://example.com/model.safetensors",
                expectedSize: 10,
                etag: "\"etag-1\"",
                downloadedBytes: 5,
                updatedAt: Date(),
                rangeSupported: true
            )
        ).write(to: sidecarURL)

        MockResumableDownloadURLProtocol.install(
            stubs: [
                .init(
                    statusCode: 206,
                    headers: [
                        "Content-Range": "bytes 5-9/10",
                        "Content-Length": "5",
                        "ETag": "\"etag-1\"",
                        "Accept-Ranges": "bytes"
                    ],
                    body: Data("world".utf8)
                )
            ]
        )

        let progress = Progress(totalUnitCount: 10)
        let result = try await ResumableModelDownloadSupport.download(
            ResumableDownloadDescriptor(
                sourceURL: URL(string: "https://example.com/model.safetensors")!,
                destinationURL: destination,
                relativePath: "model.safetensors",
                expectedSize: 10,
                userAgent: "VoxtTests",
                disableProxy: false,
                policy: resumableTestPolicy,
                protocolClasses: [MockResumableDownloadURLProtocol.self]
            ),
            progress: progress
        )

        XCTAssertEqual(result.bytesDownloaded, 10)
        XCTAssertEqual(result.resumedFromBytes, 5)
        XCTAssertEqual(try Data(contentsOf: destination), Data("helloworld".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: sidecarURL.path))
        let request = try XCTUnwrap(MockResumableDownloadURLProtocol.requests().first)
        XCTAssertEqual(request.value(forHTTPHeaderField: "Range"), "bytes=5-")
        XCTAssertEqual(request.value(forHTTPHeaderField: "If-Range"), "\"etag-1\"")
    }

    func testResumableDownloadFailsWhenStrictExpectedSizeDiffersFromResponse() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = directory.appendingPathComponent("archive.tar.bz2")
        MockResumableDownloadURLProtocol.install(
            stubs: [
                .init(
                    statusCode: 200,
                    headers: [
                        "Content-Length": "5",
                        "ETag": "\"etag-1\"",
                        "Accept-Ranges": "bytes"
                    ],
                    body: Data("hello".utf8)
                )
            ]
        )

        do {
            _ = try await ResumableModelDownloadSupport.download(
                ResumableDownloadDescriptor(
                    sourceURL: URL(string: "https://example.com/archive.tar.bz2")!,
                    destinationURL: destination,
                    relativePath: "archive.tar.bz2",
                    expectedSize: 10,
                    userAgent: "VoxtTests",
                    disableProxy: false,
                    policy: resumableTestPolicy,
                    protocolClasses: [MockResumableDownloadURLProtocol.self]
                ),
                progress: Progress(totalUnitCount: 10)
            )
            XCTFail("Expected strict size validation to fail.")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Download size mismatch"))
        }
    }

    func testResumableDownloadAllowsResponseSizeMismatchWhenStrictMatchIsDisabled() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = directory.appendingPathComponent("archive.tar.bz2")
        MockResumableDownloadURLProtocol.install(
            stubs: [
                .init(
                    statusCode: 200,
                    headers: [
                        "Content-Length": "5",
                        "ETag": "\"etag-1\"",
                        "Accept-Ranges": "bytes"
                    ],
                    body: Data("hello".utf8)
                )
            ]
        )

        let result = try await ResumableModelDownloadSupport.download(
            ResumableDownloadDescriptor(
                sourceURL: URL(string: "https://example.com/archive.tar.bz2")!,
                destinationURL: destination,
                relativePath: "archive.tar.bz2",
                expectedSize: 10,
                requiresExpectedSizeMatch: false,
                userAgent: "VoxtTests",
                disableProxy: false,
                policy: resumableTestPolicy,
                protocolClasses: [MockResumableDownloadURLProtocol.self]
            ),
            progress: Progress(totalUnitCount: 10)
        )

        XCTAssertEqual(result.bytesDownloaded, 5)
        XCTAssertEqual(try Data(contentsOf: destination), Data("hello".utf8))
    }

    func testResumableDownloadRestartsFromZeroWhenServerIgnoresRange() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = directory.appendingPathComponent("model.safetensors")
        let partURL = destination.appendingPathExtension("part")
        let sidecarURL = partURL.appendingPathExtension("json")
        try Data("hello".utf8).write(to: partURL)
        try JSONEncoder().encode(
            ResumableDownloadState(
                relativePath: "model.safetensors",
                sourceURL: "https://example.com/model.safetensors",
                expectedSize: 10,
                etag: "\"etag-1\"",
                downloadedBytes: 5,
                updatedAt: Date(),
                rangeSupported: true
            )
        ).write(to: sidecarURL)

        MockResumableDownloadURLProtocol.install(
            stubs: [
                .init(
                    statusCode: 200,
                    headers: [
                        "Content-Length": "10",
                        "ETag": "\"etag-1\"",
                        "Accept-Ranges": "bytes"
                    ],
                    body: Data()
                ),
                .init(
                    statusCode: 200,
                    headers: [
                        "Content-Length": "10",
                        "ETag": "\"etag-1\"",
                        "Accept-Ranges": "bytes"
                    ],
                    body: Data("helloworld".utf8)
                )
            ]
        )

        let progress = Progress(totalUnitCount: 10)
        let result = try await ResumableModelDownloadSupport.download(
            ResumableDownloadDescriptor(
                sourceURL: URL(string: "https://example.com/model.safetensors")!,
                destinationURL: destination,
                relativePath: "model.safetensors",
                expectedSize: 10,
                userAgent: "VoxtTests",
                disableProxy: false,
                policy: resumableTestPolicy,
                protocolClasses: [MockResumableDownloadURLProtocol.self]
            ),
            progress: progress
        )

        XCTAssertEqual(result.bytesDownloaded, 10)
        XCTAssertEqual(try Data(contentsOf: destination), Data("helloworld".utf8))
        let requests = MockResumableDownloadURLProtocol.requests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests.first?.value(forHTTPHeaderField: "Range"), "bytes=5-")
        XCTAssertNil(requests.last?.value(forHTTPHeaderField: "Range"))
    }

    func testResumableDownloadTreats416AsCompletedWhenPartialMatchesExpectedSize() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = directory.appendingPathComponent("model.safetensors")
        let partURL = destination.appendingPathExtension("part")
        let sidecarURL = partURL.appendingPathExtension("json")
        try Data("helloworld".utf8).write(to: partURL)
        try JSONEncoder().encode(
            ResumableDownloadState(
                relativePath: "model.safetensors",
                sourceURL: "https://example.com/model.safetensors",
                expectedSize: 10,
                etag: "\"etag-1\"",
                downloadedBytes: 10,
                updatedAt: Date(),
                rangeSupported: true
            )
        ).write(to: sidecarURL)

        MockResumableDownloadURLProtocol.install(
            stubs: [
                .init(
                    statusCode: 416,
                    headers: [
                        "Content-Range": "bytes */10",
                        "ETag": "\"etag-1\""
                    ],
                    body: Data()
                )
            ]
        )

        let progress = Progress(totalUnitCount: 10)
        let result = try await ResumableModelDownloadSupport.download(
            ResumableDownloadDescriptor(
                sourceURL: URL(string: "https://example.com/model.safetensors")!,
                destinationURL: destination,
                relativePath: "model.safetensors",
                expectedSize: 10,
                userAgent: "VoxtTests",
                disableProxy: false,
                policy: resumableTestPolicy,
                protocolClasses: [MockResumableDownloadURLProtocol.self]
            ),
            progress: progress
        )

        XCTAssertEqual(result.bytesDownloaded, 10)
        XCTAssertEqual(result.resumedFromBytes, 10)
        XCTAssertEqual(try Data(contentsOf: destination), Data("helloworld".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: sidecarURL.path))
    }

    func testResumableDownloadRestartsFromZeroWhenETagChanges() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = directory.appendingPathComponent("model.safetensors")
        let partURL = destination.appendingPathExtension("part")
        let sidecarURL = partURL.appendingPathExtension("json")
        try Data("hello".utf8).write(to: partURL)
        try JSONEncoder().encode(
            ResumableDownloadState(
                relativePath: "model.safetensors",
                sourceURL: "https://example.com/model.safetensors",
                expectedSize: 10,
                etag: "\"etag-1\"",
                downloadedBytes: 5,
                updatedAt: Date(),
                rangeSupported: true
            )
        ).write(to: sidecarURL)

        MockResumableDownloadURLProtocol.install(
            stubs: [
                .init(
                    statusCode: 206,
                    headers: [
                        "Content-Range": "bytes 5-9/10",
                        "Content-Length": "5",
                        "ETag": "\"etag-2\"",
                        "Accept-Ranges": "bytes"
                    ],
                    body: Data("world".utf8)
                ),
                .init(
                    statusCode: 200,
                    headers: [
                        "Content-Length": "10",
                        "ETag": "\"etag-2\"",
                        "Accept-Ranges": "bytes"
                    ],
                    body: Data("helloworld".utf8)
                )
            ]
        )

        let progress = Progress(totalUnitCount: 10)
        let result = try await ResumableModelDownloadSupport.download(
            ResumableDownloadDescriptor(
                sourceURL: URL(string: "https://example.com/model.safetensors")!,
                destinationURL: destination,
                relativePath: "model.safetensors",
                expectedSize: 10,
                userAgent: "VoxtTests",
                disableProxy: false,
                policy: resumableTestPolicy,
                protocolClasses: [MockResumableDownloadURLProtocol.self]
            ),
            progress: progress
        )

        XCTAssertEqual(result.bytesDownloaded, 10)
        XCTAssertEqual(try Data(contentsOf: destination), Data("helloworld".utf8))
        let requests = MockResumableDownloadURLProtocol.requests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests.first?.value(forHTTPHeaderField: "Range"), "bytes=5-")
        XCTAssertNil(requests.last?.value(forHTTPHeaderField: "Range"))
    }

    func testResumableDownloadDropsCorruptSidecarAndRestartsFresh() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = directory.appendingPathComponent("model.safetensors")
        let partURL = destination.appendingPathExtension("part")
        let sidecarURL = partURL.appendingPathExtension("json")
        try Data("hello".utf8).write(to: partURL)
        try Data("not-json".utf8).write(to: sidecarURL)

        MockResumableDownloadURLProtocol.install(
            stubs: [
                .init(
                    statusCode: 200,
                    headers: [
                        "Content-Length": "10",
                        "ETag": "\"etag-1\"",
                        "Accept-Ranges": "bytes"
                    ],
                    body: Data("helloworld".utf8)
                )
            ]
        )

        let progress = Progress(totalUnitCount: 10)
        let result = try await ResumableModelDownloadSupport.download(
            ResumableDownloadDescriptor(
                sourceURL: URL(string: "https://example.com/model.safetensors")!,
                destinationURL: destination,
                relativePath: "model.safetensors",
                expectedSize: 10,
                userAgent: "VoxtTests",
                disableProxy: false,
                policy: resumableTestPolicy,
                protocolClasses: [MockResumableDownloadURLProtocol.self]
            ),
            progress: progress
        )

        XCTAssertEqual(result.bytesDownloaded, 10)
        XCTAssertEqual(result.resumedFromBytes, 0)
        XCTAssertEqual(try Data(contentsOf: destination), Data("helloworld".utf8))
        let request = try XCTUnwrap(MockResumableDownloadURLProtocol.requests().first)
        XCTAssertNil(request.value(forHTTPHeaderField: "Range"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: sidecarURL.path))
    }

    func testPurgePartialArtifactsRemovesPartAndSidecar() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = directory.appendingPathComponent("model.safetensors")
        let partURL = destination.appendingPathExtension("part")
        let sidecarURL = partURL.appendingPathExtension("json")
        try Data("hello".utf8).write(to: partURL)
        try Data("{}".utf8).write(to: sidecarURL)

        try ResumableModelDownloadSupport.purgePartialArtifacts(for: destination)

        XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: sidecarURL.path))
    }
}
