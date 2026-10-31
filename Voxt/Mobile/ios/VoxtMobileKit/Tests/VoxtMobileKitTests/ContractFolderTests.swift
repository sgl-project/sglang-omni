import Foundation
import Testing

/// Upstream carries the contract once, in Voxt/Mobile/contract; this repository keeps a copy
/// beside the tests. VOXT_CONTRACT_DIR overrides both.
@Suite struct ContractFolderTests {
    /// A package laid out as Voxt/Mobile/ios/VoxtMobileKit, with or without Voxt/Mobile/contract.
    private func layout(withSharedContract: Bool) throws -> (tests: URL, shared: URL, cleanUp: () -> Void) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("contract-folder-\(UUID().uuidString)")
        let tests = root.appendingPathComponent("Voxt/Mobile/ios/VoxtMobileKit/Tests/VoxtMobileKitTests")
        let shared = root.appendingPathComponent("Voxt/Mobile/contract")
        try FileManager.default.createDirectory(at: tests.appendingPathComponent("Fixtures/contract"), withIntermediateDirectories: true)
        if withSharedContract { try FileManager.default.createDirectory(at: shared, withIntermediateDirectories: true) }
        return (tests, shared, { try? FileManager.default.removeItem(at: root) })
    }

    @Test func upstreamReadsTheSharedContract() throws {
        let (tests, shared, cleanUp) = try layout(withSharedContract: true)
        defer { cleanUp() }
        #expect(Contract.folder(environment: [:], testsDirectory: tests).standardizedFileURL == shared.standardizedFileURL)
    }

    @Test func withoutTheSharedContractTheCopyBesideTheTestsIsRead() throws {
        let (tests, _, cleanUp) = try layout(withSharedContract: false)
        defer { cleanUp() }
        #expect(Contract.folder(environment: [:], testsDirectory: tests) == tests.appendingPathComponent("Fixtures/contract"))
    }

    @Test func theEnvironmentNamesTheFolderFirst() throws {
        let (tests, _, cleanUp) = try layout(withSharedContract: true)
        defer { cleanUp() }
        #expect(Contract.folder(environment: ["VOXT_CONTRACT_DIR": "/elsewhere/contract"], testsDirectory: tests)
                == URL(fileURLWithPath: "/elsewhere/contract"))
    }
}
