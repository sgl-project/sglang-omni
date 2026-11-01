import Foundation

nonisolated protocol MeetingSpeakerDiarizationEngine: Sendable {
    func diarize(
        asset: MeetingAudioAsset,
        options: MeetingSpeakerDiarizationOptions
    ) async throws -> [MeetingSpeakerTurn]

    func diarizeSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL: URL?,
        options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn]

    func diarizeFile(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn]
}

nonisolated extension MeetingSpeakerDiarizationEngine {
    func diarizeFile(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        try await diarizeSession(descriptors: descriptors, loadAsset: loadAsset,
                                continuousAudioURL: nil, options: options, progress: progress)
    }

    func diarizeSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL _: URL?,
        options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        var turns: [MeetingSpeakerTurn] = []
        let descriptorCount = max(descriptors.count, 1)
        await progress?(0)
        for (index, descriptor) in descriptors.enumerated() {
            try Task.checkCancellation()
            if let asset = await loadAsset(descriptor) {
                turns.append(contentsOf: try await diarize(asset: asset, options: options))
            }
            await progress?(Double(index + 1) / Double(descriptorCount))
        }
        return turns
    }
}

enum MeetingSpeakerDiarizationEngineFactory {
    nonisolated private static let sharedSortformerEngine = SortformerMeetingSpeakerDiarizationEngine()

    nonisolated static func makeDefault(defaults _: UserDefaults = .standard) -> (any MeetingSpeakerDiarizationEngine)? {
        sharedSortformerEngine
    }
}

/// Native HTTP is the ordinary backend; installed model selection remains independent of runtime availability.
actor SortformerMeetingSpeakerDiarizationEngine: MeetingSpeakerDiarizationEngine {
    private let httpEngine = OmniHTTPSortformerMeetingSpeakerDiarizationEngine()

    func diarize(asset: MeetingAudioAsset, options: MeetingSpeakerDiarizationOptions) async throws -> [MeetingSpeakerTurn] {
        try await httpEngine.diarize(asset: asset, options: options)
    }

    func diarizeSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL: URL?, options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        try await httpEngine.diarizeSession(
            descriptors: descriptors, loadAsset: loadAsset, continuousAudioURL: continuousAudioURL,
            options: options, progress: progress
        )
    }

    func diarizeFile(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        options: MeetingSpeakerDiarizationOptions, progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        try await httpEngine.diarizeFile(descriptors: descriptors, loadAsset: loadAsset, options: options, progress: progress)
    }
}
