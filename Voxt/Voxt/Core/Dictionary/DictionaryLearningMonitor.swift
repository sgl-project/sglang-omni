// Pure learning policy; helper files separate prompt, text scope and edit comparison.
import Foundation

struct AutomaticDictionaryLearningRequest: Equatable {
    let insertedText: String
    let baselineContext: String
    let finalContext: String
    let baselineChangedFragment: String
    let finalChangedFragment: String
    let editRatio: Double
}

struct AutomaticDictionaryLearningObservationState: Equatable {
    let baselineText: String
    var latestText: String
    var didObserveChange: Bool
    var lastChangeElapsedSeconds: TimeInterval?
    var consecutiveMissingSnapshots: Int

    init(baselineText: String) {
        self.baselineText = baselineText
        self.latestText = baselineText
        self.didObserveChange = false
        self.lastChangeElapsedSeconds = nil
        self.consecutiveMissingSnapshots = 0
    }
}

enum AutomaticDictionaryLearningObservationDecision: Equatable {
    case continueObserving
    case stopWithoutAnalysis
    case settleForAnalysis(finalText: String)
}

enum AutomaticDictionaryLearningObservationScheduleDecision: Equatable {
    case schedule
    case skipTextNotInjected
    case skipNonTranscriptionOutput
    case skipFeatureDisabled
    case skipEmptyText
    case skipAutoKeyPress
}

enum AutomaticDictionaryLearningMonitor {
    enum RequestOutcome: Equatable {
        case ready(AutomaticDictionaryLearningRequest)
        case skipped(reason: String)
    }

    static let startupDelayNanoseconds: UInt64 = 900_000_000

    static let pollIntervalNanoseconds: UInt64 = 1_000_000_000

    static let initialSnapshotRetryCount = 3

    static let initialSnapshotRetryNanoseconds: UInt64 = 500_000_000

    static let observationWindowSeconds: TimeInterval = 30

    static let idleSettleSeconds: TimeInterval = 4

    static let maxConsecutiveMissingSnapshotsBeforeStop = 3

    static let maxConsecutiveMissingSnapshotsAfterObservedChange = 3

    static let maximumEditRatio = 0.8

    static func observationScheduleDecision(
        didInject: Bool,
        isTranscriptionOutput: Bool,
        isFeatureEnabled: Bool,
        insertedText: String,
        didTriggerAutoKeyPress: Bool
    ) -> AutomaticDictionaryLearningObservationScheduleDecision {
        guard didInject else { return .skipTextNotInjected }
        guard isTranscriptionOutput else { return .skipNonTranscriptionOutput }
        guard isFeatureEnabled else { return .skipFeatureDisabled }
        guard !insertedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return .skipEmptyText
        }
        guard !didTriggerAutoKeyPress else { return .skipAutoKeyPress }
        return .schedule
    }

    static func makeLearningRequest(
        insertedText rawInsertedText: String,
        baselineText rawBaselineText: String,
        finalText rawFinalText: String
    ) -> RequestOutcome {
        let insertedText = rawInsertedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let baselineText = rawBaselineText.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalText = rawFinalText.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !insertedText.isEmpty, !baselineText.isEmpty, !finalText.isEmpty else {
            return .skipped(reason: "empty inserted/baseline/final text")
        }

        if let insertedScopedOutcome = insertedScopedLearningRequest(
            insertedText: insertedText,
            baselineText: baselineText,
            finalText: finalText
        ) {
            return insertedScopedOutcome
        }

        let primaryOutcome = scopedLearningRequest(
            insertedText: insertedText,
            baselineText: baselineText,
            finalText: finalText
        )
        if case .ready = primaryOutcome {
            return primaryOutcome
        }

        if let fallbackOutcome = lineScopedLearningRequest(
            insertedText: insertedText,
            baselineText: baselineText,
            finalText: finalText
        ),
           case .ready = fallbackOutcome {
            return fallbackOutcome
        }

        return primaryOutcome
    }

    private static func insertedScopedLearningRequest(
        insertedText: String,
        baselineText: String,
        finalText: String
    ) -> RequestOutcome? {
        guard baselineText != finalText else {
            return .skipped(reason: "baseline and final text are identical")
        }
        guard let baselineInsertedRange = insertedRange(of: insertedText, in: baselineText) else {
            return nil
        }

        let baselineScopedText = scopedTextForInsertedRange(
            in: baselineText,
            insertedRange: baselineInsertedRange,
            fallback: insertedText
        )
        let finalScopedText = extractFinalScopedText(
            insertedText: insertedText,
            baselineScopedText: baselineScopedText,
            finalText: finalText
        )

        guard !finalScopedText.isEmpty else {
            return nil
        }

        let changeWindow = changedRangeWindow(
            baselineText: baselineScopedText,
            finalText: finalScopedText
        )
        guard changeWindow.hasMeaningfulChange else {
            return .skipped(reason: "changed fragment has no meaningful terms")
        }
        guard let insertedScopedRange = insertedRange(of: insertedText, in: baselineScopedText) else {
            return .skipped(reason: "inserted text not found inside baseline snapshot")
        }
        guard changeIntersectsInsertedText(
            baselineChangeRange: changeWindow.baselineRange,
            insertedRange: insertedScopedRange
        ) else {
            return .skipped(reason: "detected edit does not intersect inserted text")
        }
        guard !isPureAppendAfterBaseline(baseline: baselineScopedText, final: finalScopedText) else {
            return .skipped(reason: "detected edit does not intersect inserted text")
        }
        let editRatio = editRatio(
            inserted: insertedText,
            baseline: baselineScopedText,
            final: finalScopedText
        )
        guard editRatio <= maximumEditRatio else {
            return .skipped(
                reason: "edit ratio \(String(format: "%.3f", editRatio)) exceeded limit \(String(format: "%.3f", maximumEditRatio))"
            )
        }

        return .ready(
            AutomaticDictionaryLearningRequest(
                insertedText: insertedText,
                baselineContext: baselineScopedText,
                finalContext: finalScopedText,
                baselineChangedFragment: changeWindow.baselineFragment,
                finalChangedFragment: changeWindow.finalFragment,
                editRatio: editRatio
            )
        )
    }

    private static func scopedLearningRequest(
        insertedText: String,
        baselineText: String,
        finalText: String
    ) -> RequestOutcome {
        guard baselineText != finalText else {
            return .skipped(reason: "baseline and final text are identical")
        }

        let changeWindow = changedRangeWindow(baselineText: baselineText, finalText: finalText)
        guard changeWindow.hasMeaningfulChange else {
            return .skipped(reason: "changed fragment has no meaningful terms")
        }
        guard let insertedRange = insertedRange(of: insertedText, in: baselineText) else {
            return .skipped(reason: "inserted text not found inside baseline snapshot")
        }
        guard changeIntersectsInsertedText(
            baselineChangeRange: changeWindow.baselineRange,
            insertedRange: insertedRange
        ) else {
            return .skipped(reason: "detected edit does not intersect inserted text")
        }
        guard !isPureAppendAfterBaseline(baseline: baselineText, final: finalText) else {
            return .skipped(reason: "detected edit does not intersect inserted text")
        }
        let editRatio = editRatio(
            inserted: insertedText,
            baseline: baselineText,
            final: finalText
        )
        guard editRatio <= maximumEditRatio else {
            return .skipped(
                reason: "edit ratio \(String(format: "%.3f", editRatio)) exceeded limit \(String(format: "%.3f", maximumEditRatio))"
            )
        }

        let baselineContextRange = union(
            lhs: insertedRange,
            rhs: changeWindow.baselineRange,
            upperBound: baselineText.count
        )
        let finalAnchorRange = NSRange(
            location: min(changeWindow.finalRange.location, max(finalText.count - 1, 0)),
            length: changeWindow.finalRange.length
        )
        let baselineContext = contextualSnippet(
            in: baselineText,
            focusRange: baselineContextRange,
            radius: 72
        )
        let finalContext = contextualSnippet(
            in: finalText,
            focusRange: finalAnchorRange,
            radius: 72
        )

        return .ready(
            AutomaticDictionaryLearningRequest(
                insertedText: insertedText,
                baselineContext: baselineContext,
                finalContext: finalContext,
                baselineChangedFragment: changeWindow.baselineFragment,
                finalChangedFragment: changeWindow.finalFragment,
                editRatio: editRatio
            )
        )
    }

    private static func lineScopedLearningRequest(
        insertedText: String,
        baselineText: String,
        finalText: String
    ) -> RequestOutcome? {
        guard let insertedRange = insertedRange(of: insertedText, in: baselineText) else {
            return nil
        }

        let baselineNSString = baselineText as NSString
        let baselineLineRange = baselineNSString.lineRange(for: insertedRange)
        let baselineLine = baselineNSString.substring(with: baselineLineRange)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !baselineLine.isEmpty else {
            return nil
        }

        guard let finalLine = bestMatchingLine(
            primaryTarget: insertedText,
            secondaryTarget: baselineLine,
            within: finalText
        ) else {
            return nil
        }

        return scopedLearningRequest(
            insertedText: insertedText,
            baselineText: baselineLine,
            finalText: finalLine
        )
    }

    static func observeMissingSnapshot(
        state: inout AutomaticDictionaryLearningObservationState
    ) -> AutomaticDictionaryLearningObservationDecision {
        state.consecutiveMissingSnapshots += 1

        if !state.didObserveChange,
           state.consecutiveMissingSnapshots >= maxConsecutiveMissingSnapshotsBeforeStop {
            return .stopWithoutAnalysis
        }

        if state.didObserveChange,
           state.consecutiveMissingSnapshots >= maxConsecutiveMissingSnapshotsAfterObservedChange,
           let lastChangeElapsedSeconds = state.lastChangeElapsedSeconds,
           lastChangeElapsedSeconds >= idleSettleSeconds {
            if shouldContinueObservingForPotentialReplacement(
                baselineText: state.baselineText,
                currentFinalText: state.latestText
            ) {
                return .continueObserving
            }
            return .settleForAnalysis(finalText: state.latestText)
        }

        return .continueObserving
    }

    static func observeSnapshot(
        text: String,
        elapsedSinceLastChange: TimeInterval?,
        state: inout AutomaticDictionaryLearningObservationState
    ) -> AutomaticDictionaryLearningObservationDecision {
        state.consecutiveMissingSnapshots = 0

        guard text != state.latestText else {
            guard state.didObserveChange,
                  let elapsedSinceLastChange,
                  elapsedSinceLastChange >= idleSettleSeconds else {
                return .continueObserving
            }

            if shouldContinueObservingForPotentialReplacement(
                baselineText: state.baselineText,
                currentFinalText: state.latestText
            ) {
                return .continueObserving
            }
            return .settleForAnalysis(finalText: state.latestText)
        }

        state.latestText = text
        state.didObserveChange = true
        state.lastChangeElapsedSeconds = 0
        return .continueObserving
    }

    static func shouldFinalizeWhileFocused(
        decision: AutomaticDictionaryLearningObservationDecision
    ) -> Bool {
        switch decision {
        case .continueObserving, .settleForAnalysis:
            return false
        case .stopWithoutAnalysis:
            return true
        }
    }

    static func shouldContinueObservingForPotentialReplacement(
        baselineText: String,
        currentFinalText: String
    ) -> Bool {
        let changeWindow = changedRangeWindow(
            baselineText: baselineText,
            finalText: currentFinalText
        )
        let baselineMeaningful = DictionaryStore.normalizeTerm(changeWindow.baselineFragment)
        let finalMeaningful = DictionaryStore.normalizeTerm(changeWindow.finalFragment)
        if !baselineMeaningful.isEmpty && finalMeaningful.isEmpty {
            return true
        }
        return changeWindow.containsDeletionOnlyChangeGroup
    }
}
