import AVFoundation
import XCTest
@testable import Voxt

@MainActor
final class MLXTranscriberIdleReleaseTests: XCTestCase {
    /// The Local API makes a transcriber only to transcribe files, so idle reclamation finds one that
    /// never captured. Reading `inputNode` there opens the default input device on the main actor,
    /// and a slow device then holds every Local API request until it answers.
    func testIdleReleaseOfATranscriberThatNeverCapturedLeavesTheAudioEngineAlone() {
        let engine = AudioEngineAccessRecorder()
        let transcriber = MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo),
            audioEngine: engine
        )

        XCTAssertTrue(transcriber.releaseIdleResources())

        XCTAssertEqual(engine.inputNodeReads, 0)
        XCTAssertEqual(engine.resets, 0)
    }

    /// Quitting Voxt after the Local API transcribed files must not open the input device either.
    func testShutdownOfATranscriberThatNeverCapturedLeavesTheAudioEngineAlone() async {
        let engine = AudioEngineAccessRecorder()
        let transcriber = MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo),
            audioEngine: engine
        )

        await transcriber.shutdownForApplicationTermination()

        XCTAssertEqual(engine.inputNodeReads, 0)
    }
}

private final class AudioEngineAccessRecorder: AVAudioEngine {
    private(set) var inputNodeReads = 0
    private(set) var resets = 0

    override var inputNode: AVAudioInputNode {
        inputNodeReads += 1
        return super.inputNode
    }

    override func reset() {
        resets += 1
        super.reset()
    }
}
