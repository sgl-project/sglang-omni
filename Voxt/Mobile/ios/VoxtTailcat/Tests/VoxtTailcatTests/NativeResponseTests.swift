import Foundation
import Testing
@testable import VoxtTailcat

@Suite struct NativeResponseTests {
    @Test func endpointAcceptsCompleteSuccessAndRejectsIncompleteOrInvalidReplies() throws {
        let valid = try NativeResponse.endpoint(from: Data(#"{"handle":7,"port":49123}"#.utf8))
        #expect(valid.handle == 7 && valid.port == 49123)
        for json in ["{}", "not-json", #"{"handle":0,"port":49123}"#, #"{"handle":1,"port":0}"#, #"{"handle":1,"port":65536}"#, #"{"handle":"1","port":49123}"#] {
            #expect(throws: TailcatError.invalidResponse) { try NativeResponse.endpoint(from: Data(json.utf8)) }
        }
        #expect(throws: TailcatError.startFailed) {
            try NativeResponse.endpoint(from: Data(#"{"error":"invalid address","handle":1,"port":49123}"#.utf8))
        }
    }
    @Test func identityRequiresBothKeyEncodingsAndErrorsExposeNoNativePayload() throws {
        let privateKey = "privkey:" + String(repeating: "a", count: 64)
        let publicKey = "nodekey:" + String(repeating: "b", count: 64)
        let json = "{\"privateKey\":\"\(privateKey)\",\"publicKey\":\"\(publicKey)\"}"
        let identity = try NativeResponse.identity(from: Data(json.utf8))
        #expect(identity.privateKey == privateKey && identity.publicKey == publicKey)
        for bad in ["{}", #"{"privateKey":"privkey:short","publicKey":"nodekey:short"}"#, "{\"privateKey\":\"\(publicKey)\",\"publicKey\":\"\(privateKey)\"}"] {
            #expect(throws: TailcatError.invalidResponse) { try NativeResponse.identity(from: Data(bad.utf8)) }
        }
        #expect(throws: TailcatError.startFailed) {
            try NativeResponse.identity(from: Data(#"{"error":"fixture confidential text"}"#.utf8))
        }
    }
}
