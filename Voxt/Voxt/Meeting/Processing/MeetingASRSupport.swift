// MeetingASRSupport.swift
// Provides Meeting ASRSupport for meeting transcript processing.

import Foundation

enum MeetingASRResolvedMode: Equatable, Sendable {
    case chunk(profile: MeetingChunkingProfile)
    case liveLocal(mode: MLXLiveMode)
    case liveRemote(provider: RemoteASRProvider)

    var chunkingProfile: MeetingChunkingProfile {
        switch self {
        case .chunk(let profile):
            return profile
        case .liveLocal, .liveRemote:
            return .realtime
        }
    }

    var usesLiveSessions: Bool {
        switch self {
        case .liveLocal, .liveRemote:
            return true
        case .chunk:
            return false
        }
    }

    var usesLocalVoiceActivityGate: Bool {
        if case .liveLocal = self {
            return true
        }
        return false
    }
}

struct MeetingASREngineContext: Equatable {
    let engine: TranscriptionEngine
    let historyModelDescription: String
    let resolvedMode: MeetingASRResolvedMode
    let needsModelInitialization: Bool
    var mlxModelRepo: String? = nil

    var chunkingProfile: MeetingChunkingProfile {
        resolvedMode.chunkingProfile
    }

    func resolvingChunkingMode(_ mode: MeetingChunkingMode) -> MeetingASREngineContext {
        guard case .chunk(let automaticProfile) = resolvedMode else {
            return self
        }
        return MeetingASREngineContext(
            engine: engine,
            historyModelDescription: historyModelDescription,
            resolvedMode: .chunk(profile: mode.resolvedProfile(automaticProfile: automaticProfile)),
            needsModelInitialization: needsModelInitialization,
            mlxModelRepo: mlxModelRepo
        )
    }
}

enum MeetingASRSupport {
    static func resolveContext(
        transcriptionEngine: TranscriptionEngine,
        mlxModelState: MLXModelManager.ModelState,
        mlxCurrentModelRepo: String,
        mlxIsCurrentModelLoaded: Bool,
        mlxDisplayTitle: (String) -> String,
        remoteProvider: RemoteASRProvider,
        remoteConfiguration: RemoteProviderConfiguration
    ) -> MeetingASREngineContext {
        switch transcriptionEngine {
        case .mlxAudio:
            let localMode = resolveLocalMode(repo: mlxCurrentModelRepo)
            return MeetingASREngineContext(
                engine: .mlxAudio,
                historyModelDescription: "\(mlxDisplayTitle(mlxCurrentModelRepo)) (\(mlxCurrentModelRepo))",
                resolvedMode: localMode,
                needsModelInitialization: !mlxIsCurrentModelLoaded && modelStateNeedsInitialization(mlxModelState),
                mlxModelRepo: mlxCurrentModelRepo
            )
        case .remote:
            let resolvedMode = resolveRemoteMode(
                provider: remoteProvider,
                configuration: remoteConfiguration
            )
            // Live and chunked meetings use the same configured provider model.
            let model = remoteConfiguration.hasUsableModel ? remoteConfiguration.model : remoteProvider.suggestedModel
            return MeetingASREngineContext(
                engine: .remote,
                historyModelDescription: "\(remoteProvider.title) (\(model))",
                resolvedMode: resolvedMode,
                needsModelInitialization: false
            )
        case .dictation:
            return MeetingASREngineContext(
                engine: .dictation,
                historyModelDescription: "Direct Dictation",
                resolvedMode: .chunk(profile: .quality),
                needsModelInitialization: false
            )
        }
    }

    static func resolveLocalMode(repo: String) -> MeetingASRResolvedMode {
        guard MLXModelCatalog.isAvailableModelRepo(repo) else {
            return .chunk(profile: .quality)
        }
        let liveMode = MLXModelCatalog.liveMode(for: repo)
        switch liveMode {
        case .nativeQwenLive, .nativeStreamingLive, .nativeNemotronLive:
            return .liveLocal(mode: liveMode)
        case .batchPreview:
            return .chunk(profile: .quality)
        }
    }

    static func resolveRemoteMode(
        provider: RemoteASRProvider,
        configuration: RemoteProviderConfiguration
    ) -> MeetingASRResolvedMode {
        if RemoteASRRealtimeSupport.usesRealtimeMeetingProfile(
            provider: provider,
            configuration: configuration
        ) {
            return .liveRemote(provider: provider)
        }

        switch provider {
        case .doubaoASR:
            return .chunk(profile: .quality)
        case .aliyunBailianASR:
            return .chunk(profile: .quality)
        case .openAIWhisper:
            return .chunk(profile: configuration.openAIChunkPseudoRealtimeEnabled ? .realtime : .quality)
        case .glmASR, .stepFunASR, .xiaomiMiMoASR, .googleGeminiASR:
            return .chunk(profile: .quality)
        }
    }

    private static func modelStateNeedsInitialization(_ state: MLXModelManager.ModelState) -> Bool {
        switch state {
        case .downloaded, .loading, .ready:
            return true
        case .notDownloaded, .downloading, .paused, .error:
            return false
        }
    }
}

extension RemoteASRRealtimeSupport {
    static func usesRealtimeMeetingProfile(
        provider: RemoteASRProvider,
        configuration: RemoteProviderConfiguration
    ) -> Bool {
        switch provider {
        case .openAIWhisper:
            return false
        case .doubaoASR:
            return true
        case .glmASR:
            return false
        case .aliyunBailianASR:
            return isAliyunRealtimeModel(configuration.model)
        case .stepFunASR:
            return false
        case .xiaomiMiMoASR:
            return false
        case .googleGeminiASR:
            return false
        }
    }
}
