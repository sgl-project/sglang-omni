import Foundation
import Testing
@testable import VoxtTailcat

@Suite struct CNativeDriverTests {
    @Test func nativeIdentityCreatesDistinctWellFormedKeys() throws {
        let first = try TailcatIdentity.generate(), second = try TailcatIdentity.generate()
        let distinct = first != second
        #expect(distinct)
        let separateEncodings = first.privateKey != first.publicKey
        #expect(separateEncodings)
        let firstValid = try NativeResponse.identity(from: JSONEncoder().encode(first)) == first
        #expect(firstValid)
        let secondValid = try NativeResponse.identity(from: JSONEncoder().encode(second)) == second
        #expect(secondValid)
    }
    @Test func nativeBoundaryRejectsMalformedAddressAndStopIsIdempotent() throws {
        let identity = try TailcatIdentity.generate()
        let driver = CNativeDriver()
        #expect(throws: TailcatError.startFailed) {
            try driver.start(.init(address: "banana", clientKey: identity.privateKey, remotePort: 1234))
        }
        driver.stop(0)
        driver.stop(-1)
        driver.stop(-1)
    }
}
