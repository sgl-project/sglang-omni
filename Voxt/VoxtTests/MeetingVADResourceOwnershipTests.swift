import XCTest
@testable import Voxt

@MainActor
final class MeetingVADResourceOwnershipTests: XCTestCase {
    func testIdleReclamationDetachesResourcesBeforeAsyncRelease() {
        var resources = MeetingVADResources()
        let originalStreaming = resources.streaming
        let originalOffline = resources.offline

        let idleResources = resources.replaceForIdleReclamation()

        XCTAssertIdentical(idleResources.streaming, originalStreaming)
        XCTAssertIdentical(idleResources.offline, originalOffline)
        XCTAssertFalse(resources.streaming === originalStreaming)
        XCTAssertFalse(resources.offline === originalOffline)
    }
}
