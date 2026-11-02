import Foundation
import XCTest
@testable import Voxt

final class ModelDownloadProgressTests: XCTestCase {
    func testBackwardClockDoesNotProduceNegativeBytes() {
        XCTAssertEqual(ModelDownloadProgress.inFlightBytes(
            progress: Progress(totalUnitCount: 1000), expectedFileBytes: 1000,
            startTime: Date(timeIntervalSinceReferenceDate: 10), now: Date(timeIntervalSinceReferenceDate: 0)
        ), 0)
    }

    func testHugeSizeAndElapsedTimeClampBeforeIntegerConversion() {
        let value = ModelDownloadProgress.inFlightBytes(
            progress: Progress(totalUnitCount: Int64.max), expectedFileBytes: Int64.max,
            startTime: Date(timeIntervalSinceReferenceDate: 0), now: Date(timeIntervalSinceReferenceDate: 1e20)
        )
        XCTAssertGreaterThan(value, 0)
        XCTAssertLessThan(value, Int64.max)
    }

    func testUnknownAndNegativeSizesHaveNoSyntheticProgress() {
        for size in [Int64(0), -1, Int64.min] {
            XCTAssertEqual(ModelDownloadProgress.inFlightBytes(
                progress: Progress(totalUnitCount: 1), expectedFileBytes: size,
                startTime: Date(timeIntervalSinceReferenceDate: 0), now: Date(timeIntervalSinceReferenceDate: 10)
            ), 0)
        }
    }
}
