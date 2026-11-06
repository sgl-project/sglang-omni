import Foundation
import XCTest
@testable import VoxtNewLLMKit

/// Decodes every message of llm/Fixtures with the Swift types and
/// re-encodes it: the result must equal the fixture, so the Swift types
/// carry every field the Go side (and the spec) has, with the same
/// optionality. The Go side runs the same fixtures
/// (go test ./internal/llmrt/wire).
final class FixtureTests: XCTestCase {
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    func fixtureFiles() throws -> [URL] {
        let files = try FileManager.default.contentsOfDirectory(at: Self.fixtures, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "jsonl" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertGreaterThanOrEqual(files.count, 4, "fixtures in \(Self.fixtures.path)")
        return files
    }

    func testFixturesRoundTrip() throws {
        var checked = 0
        var seen = Set<String>()
        for file in try fixtureFiles() {
            let text = try String(contentsOf: file, encoding: .utf8)
            var methodByID: [String: String] = [:]
            for (n, line) in text.split(separator: "\n").enumerated() where !line.trimmingCharacters(in: .whitespaces).isEmpty {
                let where_ = "\(file.lastPathComponent):\(n + 1)"
                let wrapper = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any], where_)
                let to = try XCTUnwrap(wrapper["to"] as? String, where_)
                let raw = try JSONSerialization.data(withJSONObject: XCTUnwrap(wrapper["msg"], where_))
                let msg = try Incoming.decode(raw)
                let reencoded: Data
                switch msg.kind {
                case .request:
                    XCTAssertEqual(to, "llm", where_)
                    methodByID[msg.id!] = msg.method!
                    seen.insert(msg.method!)
                    guard let type = paramsType(msg.method!) else {
                        // An unknown method (method_not_found): kept raw.
                        let p = try msg.params.map { try JSONDecoder().decode(JSONValue.self, from: $0) }
                        reencoded = try makeEncoder().encode(Outgoing<JSONValue>(id: msg.id, method: msg.method, params: p))
                        break
                    }
                    reencoded = try encodeParams(msg, type, where_)
                case .notification:
                    XCTAssertEqual(to, "app", where_)
                    seen.insert(msg.method!)
                    let type = try XCTUnwrap(paramsType(msg.method!), "\(where_): unknown notification \(msg.method!)")
                    XCTAssertTrue(Method.notifications.contains(msg.method!), where_)
                    reencoded = try encodeParams(msg, type, where_)
                case .response:
                    XCTAssertEqual(to, "app", where_)
                    let method = try XCTUnwrap(methodByID[msg.id!], "\(where_): response to an unknown id")
                    if let err = msg.error {
                        XCTAssertTrue(Code.all.contains(err.code), "\(where_): unknown code \(err.code)")
                        reencoded = try makeEncoder().encode(Outgoing<Empty>.failure(msg.id!, err))
                    } else {
                        let type = try XCTUnwrap(resultType(method), "\(where_): no result type for \(method)")
                        reencoded = try encodeResult(msg, type, where_)
                    }
                case .invalid:
                    XCTFail("\(where_): not a request, notification or response")
                    continue
                }
                let want = try JSONSerialization.jsonObject(with: raw)
                let got = try JSONSerialization.jsonObject(with: reencoded)
                XCTAssertTrue(jsonEqual(got, want), "\(where_): re-encoded differs\n got \(String(decoding: reencoded, as: UTF8.self))\nwant \(String(decoding: raw, as: UTF8.self))")
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 40)
        for m in Method.requests + Method.notifications {
            XCTAssertTrue(seen.contains(m), "no fixture shows \(m)")
        }
    }

    /// Equal JSON values; numbers compare as doubles, within rounding.
    func jsonEqual(_ a: Any, _ b: Any) -> Bool {
        switch (a, b) {
        case let (x as [String: Any], y as [String: Any]):
            return x.count == y.count && x.allSatisfy { k, v in y[k].map { jsonEqual(v, $0) } ?? false }
        case let (x as [Any], y as [Any]):
            return x.count == y.count && zip(x, y).allSatisfy { jsonEqual($0, $1) }
        case let (x as NSNumber, y as NSNumber):
            let xb = CFGetTypeID(x) == CFBooleanGetTypeID(), yb = CFGetTypeID(y) == CFBooleanGetTypeID()
            let d = abs(x.doubleValue - y.doubleValue)
            return xb == yb && (xb ? x.boolValue == y.boolValue : d <= 1e-12 * max(1, abs(x.doubleValue)))
        case let (x as String, y as String):
            return x == y
        case (is NSNull, is NSNull):
            return true
        default:
            return false
        }
    }

    func encodeParams<T: Codable & Sendable>(_ msg: Incoming, _ type: T.Type, _ where_: String) throws -> Data {
        do {
            let params = try msg.params.map { try JSONDecoder().decode(T.self, from: $0) }
            return try makeEncoder().encode(Outgoing<T>(id: msg.id, method: msg.method, params: params))
        } catch {
            XCTFail("\(where_): \(error)")
            throw error
        }
    }

    func encodeResult<T: Codable & Sendable>(_ msg: Incoming, _ type: T.Type, _ where_: String) throws -> Data {
        do {
            let result = try JSONDecoder().decode(T.self, from: msg.result!)
            return try makeEncoder().encode(Outgoing<T>.response(msg.id!, result))
        } catch {
            XCTFail("\(where_): \(error)")
            throw error
        }
    }

    /// The process implements exactly the requests the handshake fixture lists.
    func testHelloListsEveryRequest() throws {
        let text = try String(contentsOf: Self.fixtures.appendingPathComponent("handshake.jsonl"), encoding: .utf8)
        for line in text.split(separator: "\n") {
            let wrapper = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
            let msg = try Incoming.decode(JSONSerialization.data(withJSONObject: wrapper["msg"]!))
            guard msg.kind == .response, msg.result != nil, let hello = try? msg.decodeResult(HelloResult.self) else { continue }
            XCTAssertEqual(hello.methods, Method.requests)
            return
        }
        XCTFail("no hello result")
    }

    func testTemplateValues() throws {
        let p = try Incoming.decode(Data(#"{"id":"c","method":"chat","params":{"model":{"id":"a","format":"mlx","path":"/m"},"messages":[{"role":"user","content":"x"}],"maxTokens":8,"sampling":{"temperature":0},"template":{"enable_thinking":false,"n":2}}}"#.utf8))
            .decodeParams(ChatParams.self)
        XCTAssertEqual(p.template?["enable_thinking"], .bool(false))
        XCTAssertEqual(p.template?["enable_thinking"]?.sendable as? Bool, false)
        XCTAssertEqual(p.template?["n"]?.sendable as? Int, 2)
        XCTAssertNil(p.sampling.topP)
    }
}
