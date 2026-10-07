import Foundation

// Provider-specific text accumulation stays separate; every terminal result is
// latched, and cancellation is never converted into a successful partial result.
actor AliyunQwenResponseState {
    private var committed: [String] = []
    private var partial = ""
    private var finishRequested = false
    private var sessionFinished = false
    private var completionError: Error?
    private let onError: @Sendable (Error) -> Void

    init(onError: @escaping @Sendable (Error) -> Void = { _ in }) {
        self.onError = onError
    }

    func markFinishRequested() {
        finishRequested = true
    }

    func markSessionFinished() {
        sessionFinished = true
    }

    func markCompletedWithError(_ error: Error) {
        if sessionFinished {
            return
        }
        if completionError == nil {
            completionError = error
            onError(error)
        }
    }

    func setPartial(_ value: String) -> String {
        guard !self.sessionFinished, completionError == nil else { return mergedText() }
        partial = value
        return mergedText()
    }

    func commit(_ value: String) -> String {
        guard !self.sessionFinished, completionError == nil else { return mergedText() }
        if committed.last != value {
            committed.append(value)
        }
        partial = ""
        return mergedText()
    }

    func waitForFinalResult(timeoutSeconds: TimeInterval) async throws -> String {
        defer { sessionFinished = true }
        try Task.checkCancellation()
        let deadline = Date().addingTimeInterval(max(timeoutSeconds, 0))
        while !sessionFinished, completionError == nil, Date() < deadline {
            try await Task.sleep(for: .milliseconds(120))
        }
        try Task.checkCancellation()
        if let completionError {
            throw completionError
        }
        if finishRequested, !partial.isEmpty {
            if committed.last != partial {
                committed.append(partial)
            }
            partial = ""
        }
        return mergedText()
    }

    func currentText() -> String {
        mergedText()
    }

    private func mergedText() -> String {
        var values = committed
        if !partial.isEmpty {
            values.append(partial)
        }
        return values.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

actor StepFunResponseState {
    private var committed: [String] = []
    private var partialByItem: [String: String] = [:]
    private var finishRequested = false
    private var finishRequestedAt: Date?
    private var sessionFinished = false
    private var completionError: Error?
    private let onError: @Sendable (Error) -> Void

    init(onError: @escaping @Sendable (Error) -> Void = { _ in }) {
        self.onError = onError
    }

    func markFinishRequested() {
        finishRequested = true
        finishRequestedAt = Date()
    }

    func markSessionFinished() {
        sessionFinished = true
    }

    func markCompletedWithError(_ error: Error) {
        if sessionFinished {
            return
        }
        if completionError == nil {
            completionError = error
            onError(error)
        }
    }

    func appendDelta(_ value: String, itemID: String?) -> String {
        guard !self.sessionFinished, completionError == nil else { return mergedText() }
        let key = itemID ?? "_default"
        partialByItem[key, default: ""] += value
        return mergedText()
    }

    func commit(_ value: String, itemID: String?) -> String {
        guard !self.sessionFinished, completionError == nil else { return mergedText() }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, committed.last != trimmed {
            committed.append(trimmed)
        }
        partialByItem[itemID ?? "_default"] = nil
        if finishRequested {
            sessionFinished = true
        }
        return mergedText()
    }

    func waitForFinalResult(timeoutSeconds: TimeInterval) async throws -> String {
        defer { sessionFinished = true }
        try Task.checkCancellation()
        let deadline = Date().addingTimeInterval(max(timeoutSeconds, 0))
        while !sessionFinished, completionError == nil, Date() < deadline {
            if let finishRequestedAt,
               Date().timeIntervalSince(finishRequestedAt) >= 2.0 {
                break
            }
            try await Task.sleep(for: .milliseconds(120))
        }
        try Task.checkCancellation()
        if let completionError {
            throw completionError
        }
        if finishRequested {
            let partial = partialByItem.values.joined()
            if !partial.isEmpty, committed.last != partial {
                committed.append(partial)
            }
            partialByItem.removeAll()
        }
        return mergedText()
    }

    func currentText() -> String {
        mergedText()
    }

    private func mergedText() -> String {
        var values = committed
        let partial = partialByItem.values.joined()
        if !partial.isEmpty {
            values.append(partial)
        }
        return values.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

actor AliyunFunResponseState {
    private var committedSegments: [String] = []
    private var livePartial = ""
    private var finishRequested = false
    private var taskFinished = false
    private var completionError: Error?
    private let onError: @Sendable (Error) -> Void

    init(onError: @escaping @Sendable (Error) -> Void = { _ in }) {
        self.onError = onError
    }

    func markFinishRequested() {
        finishRequested = true
    }

    func markTaskFinished() {
        taskFinished = true
    }

    func markCompletedWithError(_ error: Error) {
        guard !taskFinished else { return }
        if completionError == nil {
            completionError = error
            onError(error)
        }
    }

    func updateWithSentence(_ text: String, isSentenceEnd: Bool) -> String {
        guard !self.taskFinished, completionError == nil else { return joinedText() }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return joinedText()
        }
        if isSentenceEnd {
            if committedSegments.last != trimmed {
                committedSegments.append(trimmed)
            }
            livePartial = ""
        } else {
            livePartial = trimmed
        }
        return joinedText()
    }

    func waitForFinalResult(timeoutSeconds: TimeInterval) async throws -> String {
        defer { taskFinished = true }
        try Task.checkCancellation()
        let deadline = Date().addingTimeInterval(max(timeoutSeconds, 0))
        while !taskFinished, completionError == nil, Date() < deadline {
            try await Task.sleep(for: .milliseconds(120))
        }
        try Task.checkCancellation()
        if let completionError {
            throw completionError
        }
        if finishRequested, !livePartial.isEmpty {
            if committedSegments.last != livePartial {
                committedSegments.append(livePartial)
            }
            livePartial = ""
        }
        return joinedText()
    }

    func currentText() -> String {
        joinedText()
    }

    private func joinedText() -> String {
        var segments = committedSegments
        if !livePartial.isEmpty {
            segments.append(livePartial)
        }
        return segments.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

actor GeminiLiveResponseState {
    private var committed: [String] = []
    private var interim = ""
    private var finishRequested = false
    private var finishRequestedAt: Date?
    private var sessionFinished = false
    private var completionError: Error?
    private let onError: @Sendable (Error) -> Void

    init(onError: @escaping @Sendable (Error) -> Void = { _ in }) {
        self.onError = onError
    }

    func markFinishRequested() {
        finishRequested = true
        finishRequestedAt = Date()
    }

    func markSessionFinished() {
        sessionFinished = true
    }

    func markCompletedWithError(_ error: Error) {
        if sessionFinished {
            return
        }
        if completionError == nil {
            completionError = error
            onError(error)
        }
    }

    func setInterim(_ value: String) -> String {
        guard !self.sessionFinished, completionError == nil else { return mergedText() }
        interim = value
        return mergedText()
    }

    func commit(_ value: String) -> String {
        guard !self.sessionFinished, completionError == nil else { return mergedText() }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, committed.last != trimmed {
            committed.append(trimmed)
        }
        interim = ""
        return mergedText()
    }

    func waitForFinalResult(timeoutSeconds: TimeInterval) async throws -> String {
        defer { sessionFinished = true }
        try Task.checkCancellation()
        let deadline = Date().addingTimeInterval(max(timeoutSeconds, 0))
        while !sessionFinished, completionError == nil, Date() < deadline {
            // The live transcribe API documents no session-end event, so a short
            // grace window after audioStreamEnd is what actually ends the wait.
            if let finishRequestedAt, Date().timeIntervalSince(finishRequestedAt) >= 2.5 {
                break
            }
            try await Task.sleep(for: .milliseconds(120))
        }
        try Task.checkCancellation()
        if let completionError {
            throw completionError
        }
        if finishRequested, !interim.isEmpty {
            if committed.last != interim {
                committed.append(interim)
            }
            interim = ""
        }
        return mergedText()
    }

    func currentText() -> String {
        mergedText()
    }

    private func mergedText() -> String {
        var values = committed
        if !interim.isEmpty {
            values.append(interim)
        }
        return GeminiLiveTranscriptJoining.join(values)
    }
}

actor DoubaoResponseState {
    private var text = ""
    private var isFinal = false
    private var completionError: Error?
    private var isSocketClosed = false
    private let onError: @Sendable (Error) -> Void

    init(onError: @escaping @Sendable (Error) -> Void = { _ in }) {
        self.onError = onError
    }

    func replace(text newText: String, isFinal: Bool) -> String {
        guard !self.isFinal, !isSocketClosed, completionError == nil else { return text }
        text = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        if isFinal {
            self.isFinal = true
        }
        return text
    }

    func markFinal() {
        isFinal = true
    }

    func markCompletedWithError(_ error: Error) {
        guard !isFinal, !isSocketClosed else { return }
        if completionError == nil {
            completionError = error
            onError(error)
        }
    }

    func markSocketClosed() {
        isSocketClosed = true
    }

    func waitForFinalResult(timeoutSeconds: TimeInterval) async throws -> String {
        defer { isFinal = true }
        try Task.checkCancellation()
        let deadline = Date().addingTimeInterval(max(timeoutSeconds, 0))
        while !isFinal, !isSocketClosed, completionError == nil, Date() < deadline {
            try await Task.sleep(for: .milliseconds(120))
        }
        try Task.checkCancellation()
        if let completionError {
            throw completionError
        }
        return text
    }

    func currentText() -> String {
        text
    }
}
