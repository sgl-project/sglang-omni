@testable import VoxtNewHelperKit
import XCTest

/// The voice processing child's samples through the pipe (VPChild): a
/// read can end inside a sample, and the samples after it must not shift.
final class FloatStreamTests: XCTestCase {
    func testSamplesSplitBetweenReadsComeOutWholeAndInOrder() {
        let samples: [Float] = [0.5, -0.25, 1, -1, 0.125]
        let bytes = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        var stream = FloatStream()
        var out: [Float] = []
        var at = 0
        for size in [3, 1, 6, 2, 7, 1] {
            out += stream.push(bytes.subdata(in: at ..< at + size))
            at += size
        }
        XCTAssertEqual(at, bytes.count)
        XCTAssertEqual(out, samples)
    }
}
