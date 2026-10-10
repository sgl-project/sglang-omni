import Foundation

extension MLXTranscriber {
    func transcribeMeetingChunkResult(
        samples: [Float],
        sampleRate: Double
    ) async -> MLXBufferedTranscriptionResult? {
        do {
            return try await transcribeBufferedResult(samples: samples, sampleRate: sampleRate)
        } catch {
            VoxtLog.meetingError("Meeting MLX chunk transcription failed: \(error.localizedDescription)")
            return nil
        }
    }
}

extension RemoteASRTranscriber {
    struct MeetingConfiguration {
        let provider: RemoteASRProvider
        let configuration: RemoteProviderConfiguration
    }

    func currentMeetingConfiguration() -> MeetingConfiguration {
        let rawProvider = UserDefaults.standard.string(forKey: AppPreferenceKey.remoteASRSelectedProvider) ?? ""
        let provider = RemoteASRProvider(rawValue: rawProvider) ?? .openAIWhisper
        let rawConfigurations = UserDefaults.standard.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
        let configurations = RemoteModelConfigurationStore.loadConfiguration(
            providerID: provider.rawValue,
            from: rawConfigurations
        ).map { [provider.rawValue: $0] } ?? [:]
        let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: provider,
            stored: configurations
        )
        return MeetingConfiguration(provider: provider, configuration: configuration)
    }

    func transcribeMeetingAudioFile(_ fileURL: URL) async throws -> String {
        let meetingConfiguration = currentMeetingConfiguration()
        return try await transcribeDebugAudioFile(
            fileURL,
            provider: meetingConfiguration.provider,
            configuration: meetingConfiguration.configuration
        )
    }
}
