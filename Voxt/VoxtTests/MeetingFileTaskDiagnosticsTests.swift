import Foundation
import XCTest
@testable import Voxt

@MainActor
final class MeetingFileTaskDiagnosticsTests: XCTestCase {
    func testDetailedFileTraceIsExplicitlyOptIn() {
        XCTAssertFalse(MeetingFileTrace.resolvedEnabled(environmentValue: nil))
        XCTAssertFalse(MeetingFileTrace.resolvedEnabled(environmentValue: "0"))
        XCTAssertFalse(MeetingFileTrace.resolvedEnabled(environmentValue: "OFF"))
        XCTAssertTrue(MeetingFileTrace.resolvedEnabled(environmentValue: "1"))
        XCTAssertTrue(MeetingFileTrace.resolvedEnabled(environmentValue: " true "))
    }

    func testTraceWithoutFileTaskContextDoesNotEvaluateDetails() {
        var evaluations = 0
        func details() -> String { evaluations += 1; return "unused" }
        MeetingFileTaskContext.$taskID.withValue(nil) {
            MeetingFileTrace.event("outside-file-task", details())
        }
        XCTAssertEqual(evaluations, 0)
    }

    func testTaskContextIsScopedAndInheritedByChildTasks() async {
        let id = UUID()
        XCTAssertNil(MeetingFileTaskContext.taskID)
        await MeetingFileTaskContext.$taskID.withValue(id) {
            XCTAssertEqual(MeetingFileTaskContext.taskID, id)
            let inherited = await Task { MeetingFileTaskContext.taskID }.value
            XCTAssertEqual(inherited, id)
            let capturedID = MeetingFileTaskContext.taskID
            let detached = await Task.detached {
                MeetingFileTaskContext.$taskID.withValue(capturedID) { MeetingFileTaskContext.taskID }
            }.value
            XCTAssertEqual(detached, id)
        }
        XCTAssertNil(MeetingFileTaskContext.taskID)
    }

    func testAdmissionFailuresHaveStableReasons() {
        XCTAssertTrue(MeetingFileTaskDiagnostics.errorSummary(
            MeetingLocalInferenceCoordinatorError.thermallyConstrained
        ).contains("reason=thermal-pressure"))
        XCTAssertTrue(MeetingFileTaskDiagnostics.errorSummary(
            MeetingLocalInferenceCoordinatorError.memoryConstrained
        ).contains("reason=memory-pressure"))
        XCTAssertTrue(MeetingFileTaskDiagnostics.errorSummary(
            MeetingLocalInferenceCoordinatorError.overloaded
        ).contains("reason=inference-queue-full"))
        XCTAssertTrue(MeetingFileTaskDiagnostics.errorSummary(CancellationError()).contains("reason=cancelled"))
    }

    func testDiagnosticsExcludeErrorDescriptionsAndPayloads() {
        let error = NSError(domain: "TestDecoder", code: 42, userInfo: [
            NSLocalizedDescriptionKey: "private transcript and /Users/person/video.mov",
            NSUnderlyingErrorKey: NSError(domain: "Secret", code: 1),
            "payload": "private request"
        ])
        XCTAssertEqual(
            MeetingFileTaskDiagnostics.errorSummary(error),
            "reason=operation-error, domain=TestDecoder, code=42"
        )
    }

    func testStageNamesAreUnambiguousAndPreserveStoredCoding() throws {
        let stages: [MeetingFileAnalysisStage] = [.preparing, .transcribing, .identifyingSpeakers, .saving]
        XCTAssertEqual(stages.map(\.diagnosticName), ["preparing", "transcribing", "identifyingSpeakers", "saving"])
        for stage in stages {
            XCTAssertFalse(stage.displayTitle.isEmpty)
            XCTAssertEqual(try JSONDecoder().decode(MeetingFileAnalysisStage.self, from: JSONEncoder().encode(stage)), stage)
        }
    }
}
