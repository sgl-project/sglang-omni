import Foundation
import XCTest
@testable import VoxtNewHelperKit

/// Decodes every message of helper/Fixtures with the Swift types and
/// re-encodes it: the result must equal the fixture, so the Swift types
/// carry every field the Go side (and the spec) has, with the same
/// optionality. The Go side runs the same fixtures
/// (go test ./internal/helper/proto).
final class FixtureTests: XCTestCase {
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    func fixtureFiles() throws -> [URL] {
        let files = try FileManager.default.contentsOfDirectory(at: Self.fixtures, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "jsonl" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertGreaterThanOrEqual(files.count, 9, "fixtures in \(Self.fixtures.path)")
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
                    XCTAssertEqual(to, "helper", where_)
                    methodByID[msg.id!] = msg.method!
                    seen.insert(msg.method!)
                    guard let type = WireTypes.params(msg.method!) else {
                        // An unknown method (method_not_found): kept raw.
                        let p = try msg.params.map { try JSONDecoder().decode(JSONValue.self, from: $0) }
                        reencoded = try makeEncoder().encode(Outgoing<JSONValue>(id: msg.id, method: msg.method, params: p))
                        break
                    }
                    reencoded = try encodeParams(msg, type, where_)
                case .notification:
                    XCTAssertEqual(to, "app", where_)
                    seen.insert(msg.method!)
                    let type = try XCTUnwrap(WireTypes.params(msg.method!), "\(where_): unknown notification \(msg.method!)")
                    XCTAssertTrue(Method.notifications.contains(msg.method!), where_)
                    reencoded = try encodeParams(msg, type, where_)
                case .response:
                    XCTAssertEqual(to, "app", where_)
                    let method = try XCTUnwrap(methodByID[msg.id!], "\(where_): response to an unknown id")
                    if let err = msg.error {
                        XCTAssertTrue(Code.all.contains(err.code), "\(where_): unknown code \(err.code)")
                        reencoded = try makeEncoder().encode(Outgoing<Empty>.failure(msg.id!, err))
                    } else {
                        let type = try XCTUnwrap(WireTypes.result(method), "\(where_): no result type for \(method)")
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
        XCTAssertGreaterThan(checked, 80)
        for m in Method.requests + Method.notifications {
            XCTAssertTrue(seen.contains(m), "no fixture shows \(m)")
        }
    }

    /// Equal JSON values; numbers compare as doubles, within rounding
    /// (JSONSerialization keeps decimals as written: 0.1 and
    /// 0.10000000000000001 come out apart).
    func jsonEqual(_ a: Any, _ b: Any) -> Bool {
        switch (a, b) {
        case let (x as [String: Any], y as [String: Any]):
            return x.count == y.count && x.allSatisfy { k, v in y[k].map { jsonEqual(v, $0) } ?? false }
        case let (x as [Any], y as [Any]):
            return x.count == y.count && zip(x, y).allSatisfy { jsonEqual($0, $1) }
        case let (x as NSNumber, y as NSNumber):
            let xb = CFGetTypeID(x) == CFBooleanGetTypeID(), yb = CFGetTypeID(y) == CFBooleanGetTypeID()
            // NSDecimalNumber's doubleValue is not always the nearest double.
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

    /// The helper implements exactly the requests the handshake fixture lists.
    func testHelloListsEveryRequest() throws {
        let text = try String(contentsOf: Self.fixtures.appendingPathComponent("handshake.jsonl"), encoding: .utf8)
        for line in text.split(separator: "\n") {
            let wrapper = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
            let msg = try Incoming.decode(JSONSerialization.data(withJSONObject: wrapper["msg"]!))
            guard msg.kind == .response, msg.result != nil, let hello = try? msg.decodeResult(HelloResult.self) else { continue }
            XCTAssertEqual(Set(hello.methods), Set(Method.requests))
            return
        }
        XCTFail("no hello result")
    }

    func testPCMDataIsBase64() throws {
        let p = PCMParams(job: "j", seq: 0, offset: 0, data: Data([0, 1, 255]))
        let json = String(decoding: try makeEncoder().encode(p), as: UTF8.self)
        XCTAssertTrue(json.contains(#""data":"AAH/""#), json)
    }

    func testIncoming() throws {
        XCTAssertEqual(try Incoming.decode(Data(#"{"id":"c1","method":"hotkey.status"}"#.utf8)).kind, .request)
        XCTAssertEqual(try Incoming.decode(Data(#"{"method":"log","params":{}}"#.utf8)).kind, .notification)
        XCTAssertEqual(try Incoming.decode(Data(#"{"id":"c1","result":{}}"#.utf8)).kind, .response)
        XCTAssertEqual(try Incoming.decode(Data(#"{"id":"c1"}"#.utf8)).kind, .invalid)
        XCTAssertThrowsError(try Incoming.decode(Data("[1]".utf8)))
        // Missing optional members take their defaults; unknown ones are ignored.
        let p = try Incoming.decode(Data(#"{"id":"j","method":"textio.paste","params":{"text":"hi","future":1}}"#.utf8))
            .decodeParams(PasteParams.self)
        XCTAssertEqual(p.text, "hi")
        XCTAssertNil(p.keepClipboard)
        XCTAssertNil(p.press)
        // Bad params are a bad_request.
        let bad = try Incoming.decode(Data(#"{"id":"j","method":"textio.paste","params":{"text":1}}"#.utf8))
        XCTAssertThrowsError(try bad.decodeParams(PasteParams.self)) { XCTAssertEqual(($0 as? RPCError)?.code, Code.badRequest) }
    }

    func testMergeApps() {
        let running = [
            AppInfo(bundleId: "com.tinyspeck.slackmacgap", name: "Slack", running: true, pid: 812, path: "/Applications/Slack.app"),
            AppInfo(bundleId: "com.google.Chrome", name: "Google Chrome", running: true, pid: 901),
        ]
        let installed = [
            AppInfo(bundleId: "com.google.chrome", name: "Google Chrome", running: false, path: "/Applications/Google Chrome.app"),
            AppInfo(bundleId: "com.apple.Notes", name: "notes", running: false, path: "/System/Applications/Notes.app"),
            AppInfo(bundleId: "", name: "No id", running: false),
        ]
        let apps = mergeApps(running: running, installed: installed)
        XCTAssertEqual(apps.map(\.bundleId), ["com.google.Chrome", "com.apple.Notes", "com.tinyspeck.slackmacgap"])
        XCTAssertEqual(apps[0].path, "/Applications/Google Chrome.app") // the running one, with the installed one's path
        XCTAssertTrue(apps[0].running)
        XCTAssertEqual(apps[0].pid, 901)
    }

    func testIconSide() {
        XCTAssertEqual(iconSide(nil), 64)
        XCTAssertEqual(iconSide(0), 64)
        XCTAssertEqual(iconSide(32), 32)
        XCTAssertEqual(iconSide(4096), maxIconSize)
        // app.icons' params: path is optional, size too.
        let p = try? Incoming.decode(Data(#"{"id":"c9","method":"app.icons","params":{"apps":[{"bundleId":"com.apple.finder"}]}}"#.utf8))
            .decodeParams(AppIconsParams.self)
        XCTAssertEqual(p?.apps, [AppRef(bundleId: "com.apple.finder")])
        XCTAssertNil(p?.size)
    }

    func testOutermostWebArea() {
        // The focused element's ancestors, nearest first: an iframe's web
        // area inside the page's.
        XCTAssertEqual(outermost(["AXTextArea", "AXGroup", "AXWebArea", "AXGroup", "AXWebArea", "AXScrollArea", "AXWindow"], role: "AXWebArea"), 4)
        XCTAssertNil(outermost(["AXTextField", "AXToolbar", "AXWindow"], role: "AXWebArea"))
    }
}
