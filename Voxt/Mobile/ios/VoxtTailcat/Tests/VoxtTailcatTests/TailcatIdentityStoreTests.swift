import Foundation
import Testing
@testable import VoxtTailcat

@MainActor private final class IdentityStorage {
    var value: String?
    var failWrite = false
    var generations = 0
    func store() -> TailcatIdentityStore {
        TailcatIdentityStore(read: { self.value }, write: {
            if self.failWrite { throw TailcatError.startFailed }
            self.value = $0
        }, generate: {
            self.generations += 1
            return TailcatIdentity(privateKey: "privkey:" + String(repeating: "a", count: 64), publicKey: "nodekey:" + String(repeating: "b", count: 64))
        })
    }
}

@Suite @MainActor struct TailcatIdentityStoreTests {
    @Test func generatedIdentityIsPersistedAndReusedAfterRelaunch() throws {
        let storage = IdentityStorage()
        let first = try storage.store().identity()
        #expect(storage.value != nil)
        #expect(try storage.store().identity() == first)
        #expect(storage.generations == 1)
    }
    @Test func failedPersistenceDoesNotReturnOrCacheAnUnstoredIdentity() throws {
        let storage = IdentityStorage(), subject = storage.store()
        storage.failWrite = true
        #expect(throws: TailcatError.startFailed) { try subject.identity() }
        #expect(storage.value == nil)
        storage.failWrite = false
        _ = try subject.identity()
        #expect(storage.generations == 2)
        #expect(storage.value != nil)
    }
    @Test func corruptStoredIdentityFailsWithoutReplacingTheAuthorizedKey() {
        let storage = IdentityStorage()
        for corrupt in ["garbage", #"{"privateKey":"privkey:short","publicKey":"nodekey:short"}"#] {
            storage.value = corrupt
            #expect(throws: TailcatError.invalidResponse) { try storage.store().identity() }
            #expect(storage.value == corrupt)
            #expect(storage.generations == 0)
        }
    }
}
