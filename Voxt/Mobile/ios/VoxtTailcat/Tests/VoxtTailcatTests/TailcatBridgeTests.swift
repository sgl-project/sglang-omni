import Foundation
import Testing
@testable import VoxtTailcat

private final class Driver: NativeTailcatDriver, @unchecked Sendable {
    private let lock = NSLock()
    private var starts = 0
    private var stops: [Int64] = []
    var fail = false
    func start(_ options: TailcatOptions) throws -> NativeEndpoint {
        lock.lock(); defer { lock.unlock() }
        if fail { throw TailcatError.startFailed }
        starts += 1
        return NativeEndpoint(handle: Int64(starts), port: 48000 + starts)
    }
    func stop(_ handle: Int64) { lock.lock(); stops.append(handle); lock.unlock() }
    var counts: (Int, [Int64]) { lock.lock(); defer { lock.unlock() }; return (starts, stops) }
}
private let first = TailcatOptions(address: "tc-first-fixture", clientKey: "privkey:fixture", remotePort: 48931)
private let second = TailcatOptions(address: "tc-second-fixture", clientKey: "privkey:fixture", remotePort: 48931)

@Suite struct TailcatBridgeTests {
    @Test func invalidatedPairingCannotBeReopenedByAnOldRequest() async throws {
        let driver = Driver()
        let subject = TailcatBridge(driver: driver)
        _ = try await subject.endpoint(for: first)
        await subject.invalidate()
        await #expect(throws: CancellationError.self) { try await subject.endpoint(for: first) }
        #expect(driver.counts.0 == 1)
        #expect(driver.counts.1 == [1])
    }

    @Test func concurrentRequestsReuseOneBridgeAndCloseAllowsNewPort() async throws {
        let driver = Driver()
        let subject = TailcatBridge(driver: driver)
        let urls = try await withThrowingTaskGroup(of: URL.self) { group in
            for _ in 0..<20 { group.addTask { try await subject.endpoint(for: first) } }
            var result: [URL] = []; for try await url in group { result.append(url) }; return result
        }
        #expect(Set(urls.map(\.absoluteString)) == ["http://127.0.0.1:48001"])
        #expect(driver.counts.0 == 1)
        await subject.close(); await subject.close()
        #expect(driver.counts.1 == [1])
        #expect(try await subject.endpoint(for: first).port == 48002)
        await subject.close()
    }
    @Test func replacingConfigClosesPreviousBridgeEvenWhenReplacementFails() async throws {
        let driver = Driver()
        let bridge = TailcatBridge(driver: driver)
        _ = try await bridge.endpoint(for: first)
        driver.fail = true
        await #expect(throws: TailcatError.startFailed) { try await bridge.endpoint(for: second) }
        #expect(driver.counts.1 == [1])
        driver.fail = false
        #expect(try await bridge.endpoint(for: first).port == 48002)
        await bridge.close()
    }
    @Test func invalidPortNeverStartsNativeBridge() async {
        let driver = Driver()
        let subject = TailcatBridge(driver: driver)
        await #expect(throws: TailcatError.invalidConfiguration) {
            try await subject.endpoint(for: .init(address: first.address, clientKey: first.clientKey, remotePort: 0))
        }
        #expect(driver.counts.0 == 0)
    }
}
