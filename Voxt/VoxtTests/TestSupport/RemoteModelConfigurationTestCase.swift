// Keep credential storage reset around every configuration test, including subclasses.
import XCTest
@testable import Voxt

class RemoteModelConfigurationTestCase: XCTestCase {
    override func setUp() {
        super.setUp()
        VoxtSecureStorage.clearAllForTesting()
    }

    override func tearDown() {
        VoxtSecureStorage.clearAllForTesting()
        super.tearDown()
    }
}
