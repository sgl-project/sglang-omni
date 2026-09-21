// SPDX-License-Identifier: Apache-2.0
import Foundation

extension AppModel {
    var reviewContent: String { mode == .ask ? answerText : resultDraft }
    var reviewDestination: InsertionTarget? { mode == .ask ? answerTarget : reviewTarget }
    var canResumeVoice: Bool { !isBusy && (mode != .ask || answerText.isEmpty) }

    func toggleVoiceCapture() {
        if phase == .recording { finish() }
        else if canResumeVoice { start() }
    }

    func newQuestion() {
        guard mode == .ask, !isBusy else { return }
        saveReviewEdits()
        discardRetryRecording()
        questionText = ""; answerText = ""; answerHistoryID = nil
        rawText = ""; liveText = ""; liveStatus = ""; error = ""; notice = ""
        target = nil; draftOperation = nil
        editorFolded = false
        presentVoicePanel()
    }

    func closeVoicePanel() {
        if isBusy { cancel() }
        if isReviewingResult { finishReview() } else { dismissVoicePanel() }
    }

    func prepareEditSelection() {
        if isEditingEntireField { notice = "" }
        isEditingEntireField = false
        target?.replacementRange = nil
        guard mode == .edit else { return }
        if let draft = draftOperation, draft.range.length == 0, !draft.text.isEmpty {
            let range = NSRange(location: 0, length: (draft.text as NSString).length)
            draftOperation = DraftSelection(text: draft.text, range: range)
            draftSelection = range
            isEditingEntireField = true
        } else if draftOperation == nil, let destination = target,
                  destination.selectedText.isEmpty, let value = destination.value, !value.isEmpty,
                  destination.element != nil, destination.range != nil {
            target?.replacementRange = CFRange(location: 0, length: (value as NSString).length)
            isEditingEntireField = true
        }
        if isEditingEntireField { notice = L("panel.editingAllText") }
    }

    var canInsertReview: Bool {
        isReviewingResult && !isBusy && reviewDestination != nil
            && !reviewContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func saveReviewEdits() {
        resultText = reviewContent
        for (id, text) in [(resultHistoryID, resultDraft), (answerHistoryID, answerText)] {
            if let index = store.history.firstIndex(where: { $0.id == id }) { store.history[index].text = text }
        }
    }

    func copyReviewedResult() {
        guard isReviewingResult, !isBusy else { return }
        saveReviewEdits()
        copyResult()
    }

    func insertReviewedResult() {
        guard canInsertReview, let destination = reviewDestination else { return }
        saveReviewEdits()
        let text = reviewContent
        let historyID = mode == .ask ? answerHistoryID : resultHistoryID
        let token = UUID(); generation = token
        phase = .processing; error = ""
        presentVoicePanel()
        task = Task {
            do {
                try await insertText(text, destination, true)
                guard generation == token else { return }
                notice = L("notice.inserted", destination.applicationName)
                phase = .idle; reviewTarget = nil; answerTarget = nil
                presentVoicePanel()
            } catch {
                guard generation == token else { return }
                phase = .idle
                editorFolded = false
                self.error = L("notice.readyToCopy", error.localizedDescription)
                if let historyID { store.note(error.localizedDescription, on: historyID) }
                Diagnostics.record("insert.failed", ["destination": destination.bundleID,
                                                     "reason": Diagnostics.code(of: error)])
                presentVoicePanel()
            }
        }
    }

    func clearReviewedResult() {
        guard isReviewingResult, !isBusy else { return }
        saveReviewEdits()
        discardRetryRecording()
        resultDraft = ""; draftSelection = NSRange(location: 0, length: 0)
        resultHistoryID = nil; draftOperation = nil; unappliedResult = ""
        error = ""; notice = ""
        focusDraft?()
    }

    func finishReview() {
        guard isReviewingResult, !isBusy else { return }
        saveReviewEdits()
        resultDraft = ""; draftSelection = NSRange(location: 0, length: 0)
        questionText = ""; answerText = ""; answerHistoryID = nil; answerTarget = nil
        editorFolded = false; resultHistoryID = nil
        isReviewingResult = false; reviewTarget = nil; draftOperation = nil; unappliedResult = ""
        dismissVoicePanel()
    }

}
