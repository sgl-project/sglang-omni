import Foundation
import OmniModelArtifacts

/// Owns one Native process; retirement never releases a replacement process.
actor MeetingSortformerModelLease {
    nonisolated let configuration: MeetingSortformerConfiguration
    nonisolated let endpoint: OmniServerEndpoint
    private let runtime: OmniASRRuntime
    private var directoryLease: OmniModelDirectoryLease?

    private init(
        configuration: MeetingSortformerConfiguration,
        endpoint: OmniServerEndpoint,
        runtime: OmniASRRuntime,
        directoryLease: OmniModelDirectoryLease
    ) {
        self.configuration = configuration
        self.endpoint = endpoint
        self.runtime = runtime
        self.directoryLease = directoryLease
    }

    static func load() async throws -> MeetingSortformerModelLease {
        guard let launchConfiguration = OmniASRBackend.configuration(for: .sortformer) else {
            throw OmniASRRuntimeError.launchFailed("Native Sortformer is unavailable; the library fallback has been retired.")
        }
        guard let directoryLease = try await MeetingSortformerModelStorage.validatedModelDirectoryLease() else {
            throw MeetingVADModelError.modelNotDownloaded
        }
        let configuration = try MeetingSortformerConfiguration.load(from: directoryLease.directory)
        try Task.checkCancellation()
        let runtime = OmniASRRuntime(
            kind: .sortformer, modelDirectory: directoryLease.directory, configuration: launchConfiguration
        )
        do {
            let endpoint = try await runtime.prepare()
            try Task.checkCancellation()
            return MeetingSortformerModelLease(
                configuration: configuration, endpoint: endpoint, runtime: runtime, directoryLease: directoryLease
            )
        } catch {
            await runtime.retire()
            withExtendedLifetime(directoryLease) {}
            throw error
        }
    }

    func close() async {
        await runtime.retire()
        directoryLease = nil
    }

    deinit {
        // Note (Omni): the runtime itself retains a Reader until the actual child exits.
        let runtime = runtime
        Task { await runtime.retire() }
    }
}
