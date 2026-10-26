// MeetingStartPlannerTests.swift
// Provides Meeting Start Planner Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class MeetingStartPlannerTests: XCTestCase {
    func testUnavailableSavedModelBlocksMeetingStart() {
        let repo = "unknown/model"
        for state: MLXModelManager.ModelState in [.downloaded, .ready, .loading] {
            let decision = MeetingStartPlanner.resolve(
                selectedEngine: .mlxAudio,
                selectedMLXRepo: repo,
                isSelectedMLXModelDownloaded: true,
                mlxModelState: state,
                remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "", endpoint: "", apiKey: "")
            )
            XCTAssertEqual(decision, .blocked(.recording(.mlxModelUnavailable(detail: OmniASRBackend.unavailabilityReason(for: repo)))))
        }
    }

    func testInstalledNativeCatalogModelsUsePlatformGateForMeetings() {
        let states: [MLXModelManager.ModelState] = [.downloaded, .ready, .loading]
        let repos: [String] = MLXModelCatalog.availableModels.map(\.id) + ["mlx-community/Parakeet-0.6B"]
        for repo in repos {
            for state in states {
                let decision = MeetingStartPlanner.resolve(
                    selectedEngine: .mlxAudio,
                    selectedMLXRepo: repo,
                    isSelectedMLXModelDownloaded: true,
                    mlxModelState: state,
                    remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "", endpoint: "", apiKey: "")
                )
                let expected: MeetingStartDecision = OmniASRBackend.isAppleSilicon
                    ? .start(.mlxAudio)
                    : .blocked(.recording(.mlxModelUnavailable(detail: OmniASRBackend.unavailabilityReason(for: repo))))
                XCTAssertEqual(decision, expected, repo)
            }
        }
    }

    func testWhisperMeetingFallsBackToMLXWhenAvailable() {
        let decision = MeetingStartPlanner.resolve(
            selectedEngine: TranscriptionEngine.resolved(rawValue: "whisperKit"),
            mlxModelState: .ready,
            remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "", endpoint: "", apiKey: "")
        )

        XCTAssertEqual(decision, .start(.mlxAudio))
    }

    func testWhisperMeetingUsesMLXAvailabilityRules() {
        let decision = MeetingStartPlanner.resolve(
            selectedEngine: TranscriptionEngine.resolved(rawValue: "whisperKit"),
            mlxModelState: .notDownloaded,
            remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "", endpoint: "", apiKey: "")
        )

        XCTAssertEqual(decision, .blocked(.recording(.mlxModelNotInstalled)))
    }

    func testMLXMeetingFollowsRecordingPlanner() {
        let decision = MeetingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            mlxModelState: .ready,
            remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "", endpoint: "", apiKey: "")
        )

        XCTAssertEqual(decision, .start(.mlxAudio))
    }





    func testMLXMeetingIgnoresDifferentRepoActiveDownload() {
        let decision = MeetingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            selectedMLXRepo: "mlx-community/whisper-small-mlx",
            activeMLXDownloadRepo: "mlx-community/Qwen3-ASR-0.6B-4bit",
            isSelectedMLXModelDownloaded: true,
            mlxModelState: .downloading(
                progress: 0.5,
                completed: 10,
                total: 20,
                currentFile: "weights.bin",
                completedFiles: 1,
                totalFiles: 2
            ),
            remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "", endpoint: "", apiKey: "")
        )

        XCTAssertEqual(decision, OmniASRBackend.isAppleSilicon ? .start(.mlxAudio) : .blocked(.recording(.mlxModelUnavailable(detail: OmniASRBackend.unavailabilityReason(for: "mlx-community/whisper-small-mlx")))))
    }

    func testMLXMeetingBlocksWhenSelectedModelIsPaused() {
        let decision = MeetingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            selectedMLXRepo: "mlx-community/Qwen3-ASR-0.6B-4bit",
            activeMLXDownloadRepo: "mlx-community/Qwen3-ASR-0.6B-4bit",
            mlxModelState: .paused(
                progress: 0.5,
                completed: 10,
                total: 20,
                currentFile: "weights.bin",
                completedFiles: 1,
                totalFiles: 2
            ),
            remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "", endpoint: "", apiKey: "")
        )

        XCTAssertEqual(decision, OmniASRBackend.isAppleSilicon ? .blocked(.recording(.mlxModelDownloading)) : .blocked(.recording(.mlxModelUnavailable(detail: OmniASRBackend.unavailabilityReason(for: MLXModelManager.defaultModelRepo)))))
    }

    func testRemoteMeetingRequiresConfiguredProvider() {
        let blocked = MeetingStartPlanner.resolve(
            selectedEngine: .remote,
            mlxModelState: .ready,
            remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "whisper-1", endpoint: "", apiKey: "")
        )
        XCTAssertEqual(blocked, .blocked(.remoteASRUnavailable))

        let allowed = MeetingStartPlanner.resolve(
            selectedEngine: .remote,
            mlxModelState: .ready,
            remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "whisper-1", endpoint: "", apiKey: "token")
        )
        XCTAssertEqual(allowed, .start(.remote))
    }

    func testDoubaoMeetingUsesConfiguredRealtimeProvider() {
        let allowed = MeetingStartPlanner.resolve(
            selectedEngine: .remote,
            mlxModelState: .ready,
            remoteASRConfiguration: .init(
                providerID: RemoteASRProvider.doubaoASR.rawValue,
                model: DoubaoASRConfiguration.modelV2,
                endpoint: "",
                apiKey: "",
                appID: "app-id",
                accessToken: "token"
            )
        )
        XCTAssertEqual(allowed, .start(.remote))
    }

    func testDictationMeetingIsBlocked() {
        let decision = MeetingStartPlanner.resolve(
            selectedEngine: .dictation,
            mlxModelState: .ready,
            remoteASRConfiguration: .init(providerID: RemoteASRProvider.openAIWhisper.rawValue, model: "whisper-1", endpoint: "", apiKey: "token")
        )

        XCTAssertEqual(decision, .blocked(.dictationUnsupported))
    }
}
