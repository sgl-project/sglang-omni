// CustomLLMModelDownloadSupportTests.swift
// Provides Custom LLMModel Download Support Tests for Voxt test coverage.

import HuggingFace
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

    @MainActor
    func testDownloadAttemptReusesCompleteFilesAndPreservesTransferDescriptorAndProgressOrder() async throws {
        let directory = try TemporaryDirectory()
        try Data("ok".utf8).write(to: directory.url.appendingPathComponent("config.json"))
        try FileManager.default.createDirectory(at: directory.url.appendingPathComponent("weights"), withIntermediateDirectories: true)
        try Data("no".utf8).write(to: directory.url.appendingPathComponent("weights/model file.safetensors"))
        let context = makeAttemptContext(entries: [
            .init(path: "config.json", size: 2),
            .init(path: "weights/model file.safetensors", size: 3)
        ], totalBytes: 5)
        var updates: [String] = []
        var transferCount = 0
        let result = try await CustomLLMModelDownloadAttempt.run(
            context: context,
            baseURL: URL(string: "https://hf-mirror.com")!,
            directory: directory.url,
            userAgent: "VoxtTests",
            bearerToken: "test-token",
            transfer: { descriptor, progress in
                transferCount += 1
                XCTAssertEqual(descriptor.sourceURL.absoluteString, "https://hf-mirror.com/test/repo/resolve/main/weights/model%20file.safetensors?download=true")
                XCTAssertEqual(descriptor.destinationURL, directory.url.appendingPathComponent("weights/model file.safetensors"))
                XCTAssertEqual(descriptor.relativePath, "weights/model file.safetensors")
                XCTAssertEqual(descriptor.expectedSize, 3)
                XCTAssertEqual(descriptor.userAgent, "VoxtTests")
                XCTAssertEqual(descriptor.bearerToken, "test-token")
                XCTAssertTrue(descriptor.disableProxy)
                XCTAssertEqual(updates.last, "started:2:1")
                try Data("abc".utf8).write(to: descriptor.destinationURL)
                progress.completedUnitCount = 3
            },
            onProgress: { event in
                switch event {
                case .updated(let fraction, let completedBytes, let totalBytes, let currentFile, let completedFiles, let totalFiles):
                    XCTAssertEqual(totalBytes, 5)
                    XCTAssertEqual(totalFiles, 2)
                    XCTAssertEqual(fraction, Double(completedBytes) / 5)
                    updates.append("\(currentFile ?? "done"):\(completedBytes):\(completedFiles)")
                case .transferStarted(_, let expectedFileBytes, let completedBytes, _, _, let completedFiles, _):
                    XCTAssertEqual(expectedFileBytes, 3)
                    updates.append("started:\(completedBytes):\(completedFiles)")
                case .transferFinished:
                    updates.append("finished")
                }
            }
        )
        XCTAssertEqual(result, directory.url)
        XCTAssertEqual(transferCount, 1)
        XCTAssertEqual(updates, ["config.json:0:0", "done:2:1", "weights/model file.safetensors:2:1", "started:2:1", "finished", "done:5:2"])
        XCTAssertEqual(try Data(contentsOf: result.appendingPathComponent("weights/model file.safetensors")), Data("abc".utf8))
    }

    @MainActor
    func testDownloadAttemptRetainsUnknownAndOneByteSizePolicyAndClampsDisplayBytes() async throws {
        let directory = try TemporaryDirectory()
        let context = makeAttemptContext(entries: [
            .init(path: "unknown.json", size: nil), .init(path: "single.json", size: 1)
        ], totalBytes: 1)
        var transferredPaths: [String] = []
        var completedFileCounts: [Int] = []
        _ = try await CustomLLMModelDownloadAttempt.run(
            context: context, baseURL: URL(string: "https://example.com")!,
            directory: directory.url, userAgent: "VoxtTests", bearerToken: nil,
            transfer: { descriptor, progress in
                transferredPaths.append(descriptor.relativePath)
                XCTAssertNil(descriptor.expectedSize)
                XCTAssertNil(descriptor.bearerToken)
                XCTAssertFalse(descriptor.disableProxy)
                XCTAssertEqual(progress.totalUnitCount, 1)
                progress.completedUnitCount = 10
            },
            onProgress: { event in
                if case .updated(let fraction, let completedBytes, _, let currentFile, let completedFiles, _) = event, currentFile == nil {
                    XCTAssertEqual(fraction, 1)
                    XCTAssertEqual(completedBytes, 1)
                    completedFileCounts.append(completedFiles)
                }
            }
        )
        XCTAssertEqual(transferredPaths, ["unknown.json", "single.json"])
        XCTAssertEqual(completedFileCounts, [1, 2])
    }

    @MainActor
    func testDownloadAttemptPropagatesFailureWithoutFinishingProgressOrStartingNextFile() async throws {
        let directory = try TemporaryDirectory()
        let context = makeAttemptContext(entries: [
            .init(path: "first.json", size: 2), .init(path: "next.json", size: 2)
        ], totalBytes: 4)
        var startedCount = 0
        var finishedCount = 0
        var transferCount = 0
        do {
            _ = try await CustomLLMModelDownloadAttempt.run(
                context: context, baseURL: URL(string: "https://example.com")!,
                directory: directory.url, userAgent: "VoxtTests", bearerToken: nil,
                transfer: { _, _ in
                    transferCount += 1
                    throw URLError(.notConnectedToInternet)
                },
                onProgress: { event in
                    switch event {
                    case .transferStarted: startedCount += 1
                    case .transferFinished: finishedCount += 1
                    case .updated: break
                    }
                }
            )
            XCTFail("Expected transport failure")
        } catch let error as URLError {
            XCTAssertEqual(error.code, .notConnectedToInternet)
        }
        XCTAssertEqual(transferCount, 1)
        XCTAssertEqual(startedCount, 1)
        XCTAssertEqual(finishedCount, 0)
    }

    @MainActor
    func testDownloadAttemptPropagatesTransferCancellationWithoutCompletionEvents() async throws {
        let directory = try TemporaryDirectory()
        var eventCount = 0
        do {
            _ = try await CustomLLMModelDownloadAttempt.run(
                context: makeAttemptContext(entries: [.init(path: "config.json", size: 2)], totalBytes: 2),
                baseURL: URL(string: "https://example.com")!, directory: directory.url,
                userAgent: "VoxtTests", bearerToken: nil,
                transfer: { _, _ in throw CancellationError() },
                onProgress: { _ in eventCount += 1 }
            )
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            XCTAssertEqual(eventCount, 2)
        }
    }

    @MainActor
    func testDownloadAttemptCancellationReachesSuspendedTransferWithoutCompletingItEarly() async throws {
        let directory = try TemporaryDirectory()
        let barrier = ManualTaskBarrier()
        let context = makeAttemptContext(entries: [.init(path: "config.json", size: 2)], totalBytes: 2)
        var eventCount = 0
        let task = Task { @MainActor in
            try await CustomLLMModelDownloadAttempt.run(
                context: context, baseURL: URL(string: "https://example.com")!,
                directory: directory.url, userAgent: "VoxtTests", bearerToken: nil,
                transfer: { _, _ in
                    await barrier.wait()
                    try Task.checkCancellation()
                },
                onProgress: { _ in eventCount += 1 }
            )
        }
        await barrier.waitUntilEntered()
        task.cancel()
        XCTAssertEqual(eventCount, 2)
        barrier.release()
        do {
            _ = try await task.value
            XCTFail("Expected suspended transfer cancellation")
        } catch is CancellationError {
            XCTAssertEqual(eventCount, 2)
        }
    }

    @MainActor
    func testDownloadAttemptChecksCancellationBeforeCreatingDestination() async throws {
        let directory = try TemporaryDirectory()
        let destination = directory.url.appendingPathComponent("not-created")
        let context = makeAttemptContext(entries: [.init(path: "config.json", size: 2)], totalBytes: 2)
        let task = Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            return try await CustomLLMModelDownloadAttempt.run(
                context: context, baseURL: URL(string: "https://example.com")!,
                directory: destination, userAgent: "VoxtTests", bearerToken: nil,
                transfer: { _, _ in XCTFail("Cancelled attempt must not transfer") },
                onProgress: { _ in XCTFail("Cancelled attempt must not publish progress") }
            )
        }
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        }
    }

    @MainActor
    func testDownloadAttemptRejectsEscapingPathBeforeTransferRegistration() async throws {
        let directory = try TemporaryDirectory()
        var eventCount = 0
        do {
            _ = try await CustomLLMModelDownloadAttempt.run(
                context: makeAttemptContext(entries: [.init(path: "../outside.json", size: 2)], totalBytes: 2),
                baseURL: URL(string: "https://example.com")!, directory: directory.url,
                userAgent: "VoxtTests", bearerToken: nil,
                transfer: { _, _ in XCTFail("Invalid path must not transfer") },
                onProgress: { _ in eventCount += 1 }
            )
            XCTFail("Expected invalid path")
        } catch let error as NSError {
            XCTAssertEqual(error.domain, "Voxt.CustomLLM")
            XCTAssertEqual(error.code, 1002)
        }
        XCTAssertEqual(eventCount, 1)
    }

    @MainActor
    private func makeAttemptContext(
        entries: [MLXModelDownloadSupport.ModelFileEntry], totalBytes: Int64
    ) -> CustomLLMModelDownloadSupport.DownloadContext {
        .init(repoID: Repo.ID(rawValue: "test/repo")!, entries: entries, totalBytes: totalBytes)
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
