// MeetingASRSupportTests.swift
// Provides Meeting ASRSupport Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class MeetingASRSupportTests: XCTestCase {
    private func assertChunkMode(
        _ context: MeetingASREngineContext,
        profile expectedProfile: MeetingChunkingProfile,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        switch context.resolvedMode {
        case .chunk(let profile):
            XCTAssertEqual(profile, expectedProfile, file: file, line: line)
        case .liveLocal(let mode):
            XCTFail("Expected chunk mode, got live local for \(mode)", file: file, line: line)
        case .liveRemote(let provider):
            XCTFail("Expected chunk mode, got live remote for \(provider)", file: file, line: line)
        }
    }

    private func assertLiveMode(
        _ context: MeetingASREngineContext,
        provider expectedProvider: RemoteASRProvider,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        switch context.resolvedMode {
        case .liveRemote(let provider):
            XCTAssertEqual(provider, expectedProvider, file: file, line: line)
        case .liveLocal(let mode):
            XCTFail("Expected live remote mode, got live local mode \(mode)", file: file, line: line)
        case .chunk(let profile):
            XCTFail("Expected live remote mode, got chunk mode \(profile)", file: file, line: line)
        }
    }

    func testWhisperMeetingFallsBackToMLXContext() {
        let context = MeetingASRSupport.resolveContext(
            transcriptionEngine: TranscriptionEngine.resolved(rawValue: "whisperKit"),
            mlxModelState: .ready,
            mlxCurrentModelRepo: "mlx-community/whisper-large-v3-turbo",
            mlxIsCurrentModelLoaded: true,
            mlxDisplayTitle: { _ in "Whisper Large v3 Turbo" },
            remoteProvider: .openAIWhisper,
            remoteConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "whisper-1", endpoint: "", apiKey: "")
        )

        XCTAssertEqual(context.engine, .mlxAudio)
        XCTAssertEqual(context.chunkingProfile, .quality)
        XCTAssertFalse(context.needsModelInitialization)
        XCTAssertEqual(
            context.historyModelDescription,
            "Whisper Large v3 Turbo (mlx-community/whisper-large-v3-turbo)"
        )
    }

    func testBatchWhisperModelUsesChunkPath() {
        let context = MeetingASRSupport.resolveContext(
            transcriptionEngine: .mlxAudio,
            mlxModelState: .ready,
            mlxCurrentModelRepo: "mlx-community/whisper-large-v3-turbo",
            mlxIsCurrentModelLoaded: true,
            mlxDisplayTitle: { _ in "Whisper Large v3 Turbo" },
            remoteProvider: .openAIWhisper,
            remoteConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "whisper-1", endpoint: "", apiKey: "")
        )

        XCTAssertEqual(context.engine, .mlxAudio)
        XCTAssertEqual(context.chunkingProfile, .quality)
        XCTAssertFalse(context.needsModelInitialization)
        assertChunkMode(context, profile: .quality)
    }

    func testVisibleQwenModelUsesNativeLocalLiveMode() {
        let mode = MeetingASRSupport.resolveLocalMode(repo: "mlx-community/Qwen3-ASR-0.6B-4bit")

        XCTAssertEqual(mode, .liveLocal(mode: .nativeQwenLive))
        XCTAssertTrue(mode.usesLiveSessions)
        XCTAssertTrue(mode.usesLocalVoiceActivityGate)
    }

    func testVisibleCohereMossAndNemotronUseTheirNativeLocalLiveModes() {
        let modes: [MeetingASRResolvedMode] = [
            MeetingASRSupport.resolveLocalMode(repo: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"),
            MeetingASRSupport.resolveLocalMode(repo: "OpenMOSS-Team/MOSS-Transcribe-Diarize"),
            MeetingASRSupport.resolveLocalMode(repo: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
        ]

        XCTAssertEqual(modes[0], .liveLocal(mode: .nativeStreamingLive))
        XCTAssertEqual(modes[1], .liveLocal(mode: .nativeStreamingLive))
        XCTAssertEqual(modes[2], .liveLocal(mode: .nativeNemotronLive))
        XCTAssertTrue(modes.allSatisfy(\.usesLocalVoiceActivityGate))
    }

    func testRemoteAndChunkModesDoNotUseLocalVoiceActivityGate() {
        XCTAssertFalse(
            MeetingASRResolvedMode.liveRemote(provider: .doubaoASR).usesLocalVoiceActivityGate
        )
        XCTAssertFalse(
            MeetingASRResolvedMode.chunk(profile: .quality).usesLocalVoiceActivityGate
        )
    }

    func testLocalLiveVoiceActivityGateStartsOnlyOnSpeechAndFinishesAfterEndpoint() {
        var gate = MeetingLocalLiveVoiceActivityGate()

        XCTAssertEqual(
            gate.consume(
                isSpeech: false,
                frameDuration: 0.2,
                hasActiveSession: false,
                endpointSilence: 0.75
            ),
            .buffer
        )
        XCTAssertEqual(
            gate.consume(
                isSpeech: true,
                frameDuration: 0.2,
                hasActiveSession: false,
                endpointSilence: 0.75
            ),
            .start
        )
        XCTAssertEqual(
            gate.consume(
                isSpeech: true,
                frameDuration: 0.2,
                hasActiveSession: true,
                endpointSilence: 0.75
            ),
            .append
        )
        XCTAssertEqual(
            gate.consume(
                isSpeech: false,
                frameDuration: 0.4,
                hasActiveSession: true,
                endpointSilence: 0.75
            ),
            .hold
        )
        XCTAssertEqual(
            gate.consume(
                isSpeech: false,
                frameDuration: 0.4,
                hasActiveSession: true,
                endpointSilence: 0.75
            ),
            .finish
        )
    }

    func testLocalLiveVoiceActivityGateCancelsPendingEndpointWhenSpeechResumes() {
        var gate = MeetingLocalLiveVoiceActivityGate()

        _ = gate.consume(
            isSpeech: false,
            frameDuration: 0.5,
            hasActiveSession: true,
            endpointSilence: 0.75
        )
        XCTAssertEqual(
            gate.consume(
                isSpeech: true,
                frameDuration: 0.1,
                hasActiveSession: true,
                endpointSilence: 0.75
            ),
            .append
        )
        XCTAssertEqual(gate.silenceDuration, 0)
    }


    func testOpenAIPseudoRealtimeUsesRealtimeProfile() {
        let context = MeetingASRSupport.resolveContext(
            transcriptionEngine: .remote,
            mlxModelState: .notDownloaded,
            mlxCurrentModelRepo: MLXModelManager.defaultModelRepo,
            mlxIsCurrentModelLoaded: false,
            mlxDisplayTitle: { _ in "" },
            remoteProvider: .openAIWhisper,
            remoteConfiguration: .init(
                providerID: RemoteASRProvider.openAIWhisper.rawValue,
                model: "gpt-4o-mini-transcribe",
                endpoint: "",
                apiKey: "token",
                openAIChunkPseudoRealtimeEnabled: true
            )
        )

        XCTAssertEqual(context.engine, .remote)
        XCTAssertEqual(context.chunkingProfile, .realtime)
        assertChunkMode(context, profile: .realtime)
    }

    func testGLMRemoteUsesQualityProfile() {
        let context = MeetingASRSupport.resolveContext(
            transcriptionEngine: .remote,
            mlxModelState: .notDownloaded,
            mlxCurrentModelRepo: MLXModelManager.defaultModelRepo,
            mlxIsCurrentModelLoaded: false,
            mlxDisplayTitle: { _ in "" },
            remoteProvider: .glmASR,
            remoteConfiguration: .init(
                providerID: RemoteASRProvider.glmASR.rawValue,
                model: "glm-asr-1",
                endpoint: "",
                apiKey: "token"
            )
        )

        XCTAssertEqual(context.chunkingProfile, .quality)
        assertChunkMode(context, profile: .quality)
    }

    func testAliyunRealtimeMeetingUsesLiveRemoteMode() {
        let context = MeetingASRSupport.resolveContext(
            transcriptionEngine: .remote,
            mlxModelState: .notDownloaded,
            mlxCurrentModelRepo: MLXModelManager.defaultModelRepo,
            mlxIsCurrentModelLoaded: false,
            mlxDisplayTitle: { _ in "" },
            remoteProvider: .aliyunBailianASR,
            remoteConfiguration: .init(
                providerID: RemoteASRProvider.aliyunBailianASR.rawValue,
                model: "fun-asr-realtime",
                endpoint: "",
                apiKey: "token"
            )
        )

        XCTAssertEqual(context.historyModelDescription, "\(RemoteASRProvider.aliyunBailianASR.title) (fun-asr-realtime)")
        assertLiveMode(context, provider: .aliyunBailianASR)
    }

    func testDoubaoMeetingUsesLiveRemoteModel() {
        let context = MeetingASRSupport.resolveContext(
            transcriptionEngine: .remote,
            mlxModelState: .notDownloaded,
            mlxCurrentModelRepo: MLXModelManager.defaultModelRepo,
            mlxIsCurrentModelLoaded: false,
            mlxDisplayTitle: { _ in "" },
            remoteProvider: .doubaoASR,
            remoteConfiguration: .init(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                model: DoubaoASRConfiguration.modelV2,
                endpoint: "",
                apiKey: "",
                appID: "app-id",
                accessToken: "token"
            )
        )

        XCTAssertEqual(context.historyModelDescription, "\(RemoteASRProvider.doubaoASR.title) (\(DoubaoASRConfiguration.modelV2))")
        assertLiveMode(context, provider: .doubaoASR)
    }
}
