import ApplicationServices
import Foundation
@testable import VoxtNewHelper
import XCTest

final class ChromeNativeProbeTests: XCTestCase {
    func testNativeChromeLifecycleProbe() throws {
        guard let path = ProcessInfo.processInfo.environment["CHROME_NATIVE_PROBE_DIR"] else { throw XCTSkip("explicit native Chrome probe runner only") }
        let dir = URL(fileURLWithPath: path)
        let logURL = dir.appendingPathComponent("native-probe-events.jsonl")
        _ = FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: logURL)
        defer { try? log.close() }
        let front = FrontApps(out: Writer(handle: log))
        try Data("ready".utf8).write(to: dir.appendingPathComponent("native-probe.ready"), options: .atomic)
        let deadline = Date().addingTimeInterval(900)
        var last = ""
        var initial: [Int: Bool] = [:]
        defer { front.manual.configure([]); front.releaseUnusedAccessibility() }
        while Date() < deadline {
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("native-command.json")),
                  let command = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = command["id"] as? String, id != last,
                  let phase = command["phase"] as? String,
                  let pid = command["pid"] as? Int,
                  let bundle = command["bundle"] as? String else {
                Thread.sleep(forTimeInterval: 0.025)
                continue
            }
            last = id
            let app = AXUIElementCreateApplication(pid_t(pid))
            AXUIElementSetMessagingTimeout(app, 0.3)
            if initial[pid] == nil {
                var before: CFTypeRef?
                XCTAssertEqual(AXUIElementCopyAttributeValue(app, "AXEnhancedUserInterface" as CFString, &before), .success)
                initial[pid] = try XCTUnwrap(before as? Bool)
            }
            if phase == "active" {
                front.manual.configure([bundle])
                if front.manual.claim(pid: pid, .activation) {
                    front.setManualAccessibility(pid: pid_t(pid), bundleId: bundle, why: "native validation")
                }
            } else {
                front.manual.configure([])
                front.releaseUnusedAccessibility()
            }
            var value: CFTypeRef?
            let error = AXUIElementCopyAttributeValue(app, "AXEnhancedUserInterface" as CFString, &value)
            XCTAssertEqual(error, .success)
            XCTAssertEqual(value as? Bool, phase == "active" ? true : initial[pid])
            let response: [String: Any] = ["id": id, "phase": phase, "pid": pid, "error": error.rawValue, "enhanced": (value as? Bool).map { $0 as Any } ?? NSNull()]
            try JSONSerialization.data(withJSONObject: response).write(to: dir.appendingPathComponent("native-response.json"), options: .atomic)
            if phase == "quit" { return }
        }
        XCTFail("native probe was not closed within its time budget")
    }
}
