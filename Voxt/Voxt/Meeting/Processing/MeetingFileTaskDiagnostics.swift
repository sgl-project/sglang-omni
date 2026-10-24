import Foundation

/// Keep diagnostic reasons stable and omit arbitrary error descriptions/userInfo:
/// providers may include transcript text, local paths or request payloads there.
nonisolated enum MeetingFileTaskDiagnostics {
    static func errorSummary(_ error: Error) -> String {
        let reason: String
        if error is CancellationError {
            reason = "cancelled"
        } else if let safety = error as? MeetingLocalInferenceCoordinatorError {
            switch safety {
            case .thermallyConstrained: reason = "thermal-pressure"
            case .memoryConstrained: reason = "memory-pressure"
            case .overloaded: reason = "inference-queue-full"
            }
        } else {
            reason = "operation-error"
        }
        let nsError = error as NSError
        let domain = String(nsError.domain.prefix(160)).map { character in
            character.isLetter || character.isNumber || ".-_".contains(character) ? character : "_"
        }
        return "reason=\(reason), domain=\(String(domain)), code=\(nsError.code)"
    }
}

extension MeetingFileAnalysisStage {
    var diagnosticName: String {
        switch self {
        case .preparing: return "preparing"
        case .transcribing: return "transcribing"
        case .identifyingSpeakers: return "identifyingSpeakers"
        case .saving: return "saving"
        }
    }

    var displayTitle: String {
        switch self {
        case .preparing: return AppLocalization.localizedString("Preparing audio…")
        case .transcribing: return AppLocalization.localizedString("Transcribing meeting…")
        case .identifyingSpeakers: return AppLocalization.localizedString("Identifying speakers…")
        case .saving: return AppLocalization.localizedString("Saving analysis…")
        }
    }
}
