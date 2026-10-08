import Foundation
import XCTest
@testable import Voxt

@MainActor
final class RemoteLLMRuntimeClientFailureTests: XCTestCase {
    func testChatStreamFailureBeforeTextFallsBackToJSONOnce() async throws {
        let fixture = Fixture(replies: [.body("data: {\"error\":{\"message\":\"stream unsupported\"}}\n\n"), .body(#"{"choices":[{"message":{"content":"final"}}]}"#)])
        let result = try await chat(fixture)
        XCTAssertEqual(result, "final")
        XCTAssertEqual(fixture.script.requests.count, 2)
        XCTAssertTrue(fixture.script.requests[0].value(forHTTPHeaderField: "Accept")?.contains("text/event-stream") == true)
        XCTAssertFalse(fixture.script.requests[1].value(forHTTPHeaderField: "Accept")?.contains("text/event-stream") == true)
    }

    func testChatFailureAfterPartialDoesNotIssueAnotherRequest() async {
        let fixture = Fixture(replies: [.body("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: {\"error\":{\"message\":\"failed\"}}\n\n")])
        do { _ = try await chat(fixture); XCTFail("Expected partial stream failure") }
        catch let failure as RemoteLLMRuntimeClient.StreamingFailure {
            XCTAssertEqual(failure.partialText, "partial")
            XCTAssertEqual(failure.emittedChunkCount, 1)
        } catch { XCTFail("Unexpected error: \(error)") }
        XCTAssertEqual(fixture.script.requests.count, 1)
    }

    func testTransportFailureBeforeHeadersFallsBackOnce() async throws {
        let fixture = Fixture(replies: [.failure(.networkConnectionLost), .body(#"{"choices":[{"message":{"content":"recovered"}}]}"#)])
        let result = try await chat(fixture)
        XCTAssertEqual(result, "recovered")
        XCTAssertEqual(fixture.script.requests.count, 2)
    }

    func testCancelledChatStreamDoesNotFallback() async {
        let started = expectation(description: "stream opened")
        let fixture = Fixture(replies: [.hold], onStart: { started.fulfill() })
        let task = Task { try await chat(fixture) }
        await fulfillment(of: [started], timeout: 3)
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError || (error as NSError).code == NSURLErrorCancelled) }
        XCTAssertEqual(fixture.script.requests.count, 1)
    }

    func testResponsesStreamFailureBeforeTextFallsBackOnce() async throws {
        let fixture = Fixture(replies: [.body("data: {\"error\":{\"message\":\"stream unavailable\"}}\n\n"), .body(#"{"id":"response-test","status":"completed","output_text":"final"}"#)])
        let result = try await responses(fixture)
        XCTAssertEqual(result.text, "final")
        XCTAssertEqual(fixture.script.requests.count, 2)
    }

    func testResponsesFailureAfterPartialDoesNotRetry() async {
        let fixture = Fixture(replies: [.body("data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\ndata: {\"error\":{\"message\":\"failed\"}}\n\n")])
        do { _ = try await responses(fixture); XCTFail("Expected partial stream failure") }
        catch let failure as RemoteLLMRuntimeClient.StreamingFailure {
            XCTAssertEqual(failure.partialText, "partial")
            XCTAssertEqual(failure.emittedChunkCount, 1)
        } catch { XCTFail("Unexpected error: \(error)") }
        XCTAssertEqual(fixture.script.requests.count, 1)
    }

    func testCancelledResponsesStreamDoesNotFallback() async {
        let started = expectation(description: "responses stream opened")
        let fixture = Fixture(replies: [.hold], onStart: { started.fulfill() })
        let task = Task { try await responses(fixture) }
        await fulfillment(of: [started], timeout: 3)
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError || (error as NSError).code == NSURLErrorCancelled) }
        XCTAssertEqual(fixture.script.requests.count, 1)
    }

    private func chat(_ fixture: Fixture) async throws -> String {
        try await fixture.client.complete(
            systemPrompt: "", debugInput: "fixture", userPrompt: "fixture", inputTextLength: 7,
            intent: .rewrite, provider: .deepseek,
            configuration: fixture.configuration(provider: .deepseek, path: "/v1/chat/completions"),
            onPartialText: { _ in }
        )
    }

    private func responses(_ fixture: Fixture) async throws -> RemoteLLMRuntimeClient.ResponsesStreamingResult {
        try await fixture.client.completeResponses(
            systemPrompt: "", debugInput: "fixture", requestContentForLog: "fixture",
            inputPayload: "fixture", inputTextLength: 7, intent: .rewrite, provider: .openAI,
            configuration: fixture.configuration(provider: .openAI, path: "/v1/responses"),
            onPartialText: { _ in }
        )
    }
}

@MainActor
private final class Fixture {
    let script: Script
    let host = "\(UUID().uuidString.lowercased()).invalid"
    let session: URLSession
    var client: RemoteLLMRuntimeClient { RemoteLLMRuntimeClient(session: session) }

    init(replies: [Script.Reply], onStart: (@Sendable () -> Void)? = nil) {
        script = Script(replies: replies, onStart: onStart)
        FaultProtocol.register(script, host: host)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FaultProtocol.self]
        configuration.timeoutIntervalForRequest = 5
        session = URLSession(configuration: configuration)
    }

    func configuration(provider: RemoteLLMProvider, path: String) -> RemoteProviderConfiguration {
        RemoteProviderConfiguration(providerID: provider.rawValue, model: provider.suggestedModel,
                                    endpoint: "https://\(host)\(path)", apiKey: "fixture-only")
    }

    isolated deinit {
        session.invalidateAndCancel()
        FaultProtocol.unregister(host: host)
    }
}

nonisolated private final class Script: @unchecked Sendable {
    enum Reply: Sendable {
        case body(String)
        case failure(URLError.Code)
        case hold
    }
    private let lock = NSLock()
    private var replies: [Reply]
    private var recorded: [URLRequest] = []
    let onStart: (@Sendable () -> Void)?
    var requests: [URLRequest] { lock.withLock { recorded } }
    init(replies: [Reply], onStart: (@Sendable () -> Void)?) { self.replies = replies; self.onStart = onStart }
    func next(_ request: URLRequest) -> Reply {
        lock.withLock {
            recorded.append(request)
            return replies.isEmpty ? .failure(.badServerResponse) : replies.removeFirst()
        }
    }
}

nonisolated private final class FaultProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var scripts: [String: Script] = [:]
    static func register(_ script: Script, host: String) { lock.withLock { scripts[host] = script } }
    static func unregister(host: String) { _ = lock.withLock { scripts.removeValue(forKey: host) } }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url, let script = Self.lock.withLock({ Self.scripts[url.host ?? ""] }) else {
            // Never allow a fixture fallback to contact a real provider host.
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        let reply = script.next(request)
        script.onStart?()
        switch reply {
        case .failure(let code):
            client?.urlProtocol(self, didFailWithError: URLError(code))
        case .hold:
            break
        case .body(let text):
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil,
                                           headerFields: ["Content-Type": text.hasPrefix("data:") ? "text/event-stream" : "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(text.utf8))
            client?.urlProtocolDidFinishLoading(self)
        }
    }
    override func stopLoading() {}
}
