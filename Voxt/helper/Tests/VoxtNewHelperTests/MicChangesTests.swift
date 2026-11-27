import Foundation
@testable import VoxtNewHelper
import VoxtNewHelperKit
import XCTest

final class MicChangesTests: XCTestCase {
    func testDisabledWatchDoesNotScheduleOrReadChanges() {
        let watch = MicWatch()
        var reads = 0
        let mic = Mic(out: Writer(handle: FileHandle.nullDevice), watch: watch, read: {
            reads += 1
            return MicUsers()
        })
        var work: (@Sendable () -> Void)?
        mic.changed(after: { work = $0 })
        XCTAssertNil(work, "off: no debounce or follow-up reads are scheduled")
        XCTAssertEqual(reads, 0)

        watch.set(true, at: MicUsers())
        mic.changed(after: { work = $0 })
        XCTAssertNotNil(work)
        watch.set(false, at: MicUsers())
        work?()
        XCTAssertEqual(reads, 0, "turned off while debouncing: no Core Audio walk")

        watch.set(true, at: MicUsers())
        work = nil
        mic.changed(after: { work = $0 })
        XCTAssertNotNil(work, "the cancelled debounce must not leave pending set")
        work?()
        XCTAssertEqual(reads, 1, "an enabled watch still reads changes")
        watch.set(false, at: MicUsers())
    }
}
