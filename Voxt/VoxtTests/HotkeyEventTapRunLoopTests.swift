import XCTest
import CoreFoundation
@testable import Voxt

final class HotkeyEventTapRunLoopTests: XCTestCase {
    func testStoppedOwnerCannotRestartThread() throws {
        let loop = HotkeyEventTapRunLoop()
        loop.stop()
        XCTAssertFalse(loop.addSource(try makeSource()))
    }

    func testStopIsIdempotentAfterSourceRegistration() throws {
        let loop = HotkeyEventTapRunLoop()
        let source = try makeSource()
        defer { loop.stop() }
        XCTAssertTrue(loop.addSource(source))
        loop.removeSource(source, retaining: NSObject())
        loop.stop()
        loop.stop()
        XCTAssertFalse(loop.addSource(source))
    }

    func testRetiringOldRunLoopDoesNotStopReplacement() throws {
        let old = HotkeyEventTapRunLoop()
        let replacement = HotkeyEventTapRunLoop()
        defer { old.stop(); replacement.stop() }
        XCTAssertTrue(old.addSource(try makeSource()))
        XCTAssertTrue(replacement.addSource(try makeSource()))
        old.stop()
        XCTAssertTrue(replacement.addSource(try makeSource()))
    }

    func testConcurrentStopsAndStartDoNotLeaveReusableRetiredOwner() async throws {
        let loop = HotkeyEventTapRunLoop()
        let source = try makeSource()
        XCTAssertTrue(loop.addSource(source))
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<4 { group.addTask { loop.stop() } }
        }
        XCTAssertFalse(loop.addSource(try makeSource()))
    }

    private func makeSource() throws -> CFRunLoopSource {
        var context = CFRunLoopSourceContext()
        return try XCTUnwrap(CFRunLoopSourceCreate(kCFAllocatorDefault, 0, &context))
    }
}
