import XCTest
@testable import Voxt

final class IdleMemoryReclamationTests: XCTestCase {
    func testCanReclaimWhenAllRuntimeWorkIsIdleAndModelsAreUnloaded() {
        let state = makeState()
        XCTAssertEqual(state.disposition, .reclaim)
        XCTAssertTrue(state.canReclaim)
    }

    func testActiveUserFlowsBlockReclamation() {
        XCTAssertFalse(makeState(isRecordingSessionActive: true).canReclaim)
        XCTAssertFalse(makeState(isMeetingActive: true).canReclaim)
        XCTAssertFalse(makeState(hasPendingRecordingWork: true).canReclaim)
        XCTAssertFalse(makeState(hasPendingLLMWork: true).canReclaim)
    }

    func testLoadedOrActiveModelsBlockReclamation() {
        XCTAssertFalse(makeState(hasLoadedASRModel: true).canReclaim)
        XCTAssertFalse(makeState(hasActiveASRUse: true).canReclaim)
        XCTAssertFalse(makeState(hasLoadedLLMModel: true).canReclaim)
        XCTAssertFalse(makeState(hasActiveLLMInference: true).canReclaim)
    }

    func testPendingLoadsBlockReclamationBeforeAContainerExists() {
        var state = makeState()
        state.hasPendingASRLoad = true
        XCTAssertEqual(state.disposition, .retryAfterTransientWork)
        XCTAssertTrue(state.blockerSummary.contains("asr-loading"))
        state.hasPendingASRLoad = false
        state.hasPendingLLMLoad = true
        XCTAssertEqual(state.disposition, .retryAfterTransientWork)
        XCTAssertTrue(state.blockerSummary.contains("llm-loading"))
    }

    func testTranscriberWorkAndTerminationBlockReclamation() {
        XCTAssertFalse(makeState(isTranscriberRecording: true).canReclaim)
        XCTAssertFalse(makeState(isTranscriberFinalizing: true).canReclaim)
        XCTAssertFalse(makeState(isApplicationTerminating: true).canReclaim)
    }

    func testTransientBusinessWorkRetriesInsteadOfDroppingReclamation() {
        XCTAssertEqual(
            makeState(isMeetingActive: true).disposition,
            .retryAfterTransientWork
        )
        XCTAssertEqual(
            makeState(hasPendingLLMWork: true).disposition,
            .retryAfterTransientWork
        )
        XCTAssertEqual(
            makeState(hasActiveASRUse: true).disposition,
            .retryAfterTransientWork
        )
    }

    func testLoadedModelWaitsForUnloadCallbackWithoutPolling() {
        XCTAssertEqual(
            makeState(hasPendingLLMWork: true, hasLoadedASRModel: true).disposition,
            .waitForModelUnload
        )
        XCTAssertEqual(
            makeState(hasLoadedLLMModel: true).disposition,
            .waitForModelUnload
        )
    }

    func testRemoteVADFlowCanReclaimWithoutAnyLocalModelUnloadEvent() {
        XCTAssertEqual(makeState().disposition, .reclaim)
    }

    func testTerminationStopsReclamationWorker() {
        XCTAssertEqual(
            makeState(isApplicationTerminating: true).disposition,
            .stop
        )
    }

    func testModelUnloadTransitionSchedulesReclamation() {
        XCTAssertTrue(
            ModelUnloadReclamationNotificationPolicy.shouldNotify(
                wasLoaded: true,
                isLoaded: false,
                isApplicationTerminating: false
            )
        )
    }

    func testModelUnloadNotificationIgnoresNonTransitionsAndTermination() {
        XCTAssertFalse(
            ModelUnloadReclamationNotificationPolicy.shouldNotify(
                wasLoaded: false,
                isLoaded: false,
                isApplicationTerminating: false
            )
        )
        XCTAssertFalse(
            ModelUnloadReclamationNotificationPolicy.shouldNotify(
                wasLoaded: true,
                isLoaded: true,
                isApplicationTerminating: false
            )
        )
        XCTAssertFalse(
            ModelUnloadReclamationNotificationPolicy.shouldNotify(
                wasLoaded: true,
                isLoaded: false,
                isApplicationTerminating: true
            )
        )
    }

    func testBlockerSummaryNamesEveryActiveRuntimeCondition() {
        let state = makeState(
            isApplicationTerminating: true,
            isRecordingSessionActive: true,
            isMeetingActive: true,
            hasPendingRecordingWork: true,
            hasPendingLLMWork: true,
            hasLoadedASRModel: true,
            hasActiveASRUse: true,
            hasLoadedLLMModel: true,
            hasActiveLLMInference: true,
            isTranscriberRecording: true,
            isTranscriberFinalizing: true
        )

        XCTAssertEqual(
            state.blockerSummary,
            "application-terminating,recording-session,meeting-session,pending-recording-work,"
                + "pending-llm-work,asr-model-loaded,asr-active-use,llm-model-loaded,"
                + "llm-active-inference,transcriber-recording,transcriber-finalizing"
        )
        XCTAssertEqual(makeState().blockerSummary, "none")
    }

    private func makeState(
        isApplicationTerminating: Bool = false,
        isRecordingSessionActive: Bool = false,
        isMeetingActive: Bool = false,
        hasPendingRecordingWork: Bool = false,
        hasPendingLLMWork: Bool = false,
        hasLoadedASRModel: Bool = false,
        hasActiveASRUse: Bool = false,
        hasLoadedLLMModel: Bool = false,
        hasActiveLLMInference: Bool = false,
        isTranscriberRecording: Bool = false,
        isTranscriberFinalizing: Bool = false
    ) -> DeepIdleMemoryReclamationState {
        DeepIdleMemoryReclamationState(
            isApplicationTerminating: isApplicationTerminating,
            isRecordingSessionActive: isRecordingSessionActive,
            isMeetingActive: isMeetingActive,
            hasPendingRecordingWork: hasPendingRecordingWork,
            hasPendingLLMWork: hasPendingLLMWork,
            hasLoadedASRModel: hasLoadedASRModel,
            hasActiveASRUse: hasActiveASRUse,
            hasLoadedLLMModel: hasLoadedLLMModel,
            hasActiveLLMInference: hasActiveLLMInference,
            isTranscriberRecording: isTranscriberRecording,
            isTranscriberFinalizing: isTranscriberFinalizing
        )
    }
}
