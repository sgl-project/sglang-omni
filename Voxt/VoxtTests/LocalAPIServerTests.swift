// LocalAPIServerTests.swift
// Provides Local API loopback server tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LocalAPIServerTests: XCTestCase {
    func testServesRequestsOnLoopbackAndPassesBodyThrough() async throws {
        let server = LocalAPIServer(port: 0, maxBodyBytes: 5_000_000) { request in
            LocalAPIHTTPResponse.json(status: 200, [
                "path": request.path,
                "bytes": String(request.body.count),
                "auth": request.header("authorization") ?? ""
            ])
        }
        let port = try await server.start()
        defer { server.stop() }
        XCTAssertGreaterThan(port, 0)

        // Bigger than one TCP read, so the connection has to accumulate chunks.
        let body = Data(repeating: 7, count: 1_200_000)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/dictations")!)
        request.httpMethod = "POST"
        request.setValue("Bearer t", forHTTPHeaderField: "Authorization")
        request.setValue("audio/wav", forHTTPHeaderField: "Content-Type")
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)

        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: String])
        XCTAssertEqual(json["path"], "/v1/dictations")
        XCTAssertEqual(json["bytes"], "1200000")
        XCTAssertEqual(json["auth"], "Bearer t")
    }

    func testOversizedBodyGets413WithoutCallingHandler() async throws {
        let called = expectation(description: "handler")
        called.isInverted = true
        let server = LocalAPIServer(port: 0, maxBodyBytes: 10) { _ in
            called.fulfill()
            return LocalAPIHTTPResponse.json(status: 200, ["ok": "1"])
        }
        let port = try await server.start()
        defer { server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/dictations")!)
        request.httpMethod = "POST"
        request.httpBody = Data(repeating: 1, count: 100)
        let (_, response) = try await URLSession.shared.data(for: request)

        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 413)
        await fulfillment(of: [called], timeout: 0.3)
    }

    func testListenerIsNotReachableOnNonLoopbackAddress() async throws {
        let server = LocalAPIServer(port: 0, maxBodyBytes: 10) { _ in
            LocalAPIHTTPResponse.json(status: 200, ["ok": "1"])
        }
        let port = try await server.start()
        defer { server.stop() }
        guard let lanAddress = LocalAPIServerTests.firstNonLoopbackIPv4() else {
            throw XCTSkip("No non-loopback IPv4 address on this machine")
        }

        var request = URLRequest(url: URL(string: "http://\(lanAddress):\(port)/v1/health")!)
        request.timeoutInterval = 2
        do {
            _ = try await URLSession.shared.data(for: request)
            XCTFail("Server answered on \(lanAddress); it must bind to loopback only")
        } catch {
            // Expected: connection refused.
        }
    }

    private static func firstNonLoopbackIPv4() -> String? {
        var pointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&pointer) == 0, let first = pointer else { return nil }
        defer { freeifaddrs(pointer) }
        for entry in sequence(first: first, next: { $0.pointee.ifa_next }) {
            guard let address = entry.pointee.ifa_addr, address.pointee.sa_family == UInt8(AF_INET),
                  (entry.pointee.ifa_flags & UInt32(IFF_LOOPBACK)) == 0 else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(address, socklen_t(address.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                return String(cString: host)
            }
        }
        return nil
    }
}
