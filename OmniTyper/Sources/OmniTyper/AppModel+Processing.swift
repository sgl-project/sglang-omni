// SPDX-License-Identifier: Apache-2.0
import Foundation

extension AppModel {
    var recordingStyle: String {
        let preferences = isBusy ? sessionPreferences : store.preferences
        return verbatimDictation ? "verbatim" : store.rules.first { $0.bundleID == target?.bundleID }?.style ?? preferences.style
    }

    var requiresTextAPI: Bool { mode != .dictate || recordingStyle != "verbatim" }

    func payload(audio: URL?, text: String? = nil) throws -> [String: Any] {
        let preferences = sessionPreferences
        let rule = store.rules.first { $0.bundleID == target?.bundleID }
        let instructions = try Preferences.combinedInstructions(preferences.instructions, rule?.instructions ?? "")
        guard store.dictionary.count <= 200, store.dictionary.allSatisfy(\.isValid) else {
            throw Failure("error.dictionaryInvalid")
        }
        let selectedText = (mode == .edit || mode == .ask)
            ? try draftOperation?.selectedText() ?? (mode == .edit ? target?.editingText : target?.selectedText) ?? "" : ""
        guard mode != .edit || !selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Failure("error.editNeedsSelection")
        }
        guard selectedText.unicodeScalars.count <= 12_000, !selectedText.contains("\0") else {
            throw Failure("error.selectionTooLong")
        }
        var request: [String: Any] = [
            "op": audio == nil ? "process" : "transcribe",
            "asr_model": preferences.asrModel,
            "mode": mode.rawValue, "language": preferences.language,
            "target_language": preferences.targetLanguage, "style": recordingStyle,
            "instructions": instructions,
            "dictionary": store.dictionary.map { ["spoken": $0.spoken, "written": $0.written] },
            "selected_text": selectedText, "app_name": lastApp
        ]
        if requiresTextAPI {
            request.merge(try preferences.textSettings.payload(apiKey: sessionAPIKey)) { _, new in new }
        }
        if let audio { request["audio_path"] = audio.path }
        if let text { request["text"] = text }
        return request
    }

    func run(audio: URL, duration: Double, allowInsertion: Bool) {
        guard !isSelectingMode, !isReviewingResult || draftOperation != nil else { return }
        phase = .processing; error = ""; notice = ""; rawText = ""
        presentVoicePanel()
        let token = UUID(); generation = token
        let request = Result { try payload(audio: audio) }
        let requestMode = mode
        let capturedTarget = target
        let draft = draftOperation
        let preferences = sessionPreferences
        let recording = FailedRecording(url: audio, duration: duration, mode: requestMode, verbatim: verbatimDictation,
                                        target: capturedTarget, appName: lastApp, draft: draft)
        task = Task {
            do {
                var payload = try request.get()
                if let preloadTask { _ = try await preloadTask.value }
                var streamingWarning = ""
                if let stream = speechStream {
                    liveStatus = L("status.finalizing")
                    do {
                        let transcript = try await stream.finish()
                        payload["op"] = "process"
                        payload["audio_path"] = nil
                        payload["text"] = transcript
                        if requestMode == .ask { questionText = transcript }
                    } catch {
                        guard generation == token, !Task.isCancelled else { throw CancellationError() }
                        streamingWarning = L("notice.streamRecovered")
                    }
                    stream.cancel(); speechStream = nil
                }
                try Task.checkCancellation()
                liveStatus = payload["op"] as? String == "process" ? L("status.processingText") : L("status.transcribing")
                let response = try await worker.request(payload, python: preferences.pythonExecutable)
                guard generation == token, !Task.isCancelled else { try? FileManager.default.removeItem(at: audio); return }
                guard let text = response["text"] as? String else {
                    throw WorkerFailure(message: L("worker.incomplete"), rawText: response["raw_text"] as? String)
                }
                let raw = response["raw_text"] as? String ?? text
                let warning = [streamingWarning, response["warning"] as? String ?? ""].filter { !$0.isEmpty }.joined(separator: " ")
                resultText = text; rawText = raw
                if requestMode == .ask { questionText = raw }
                let review = preferences.recordingPresentation == .edit || requestMode == .ask || draft != nil
                if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    notice = L("notice.noSpeech")
                    try? FileManager.default.removeItem(at: audio)
                    if review {
                        if !isReviewingResult { reviewTarget = allowInsertion ? capturedTarget : nil }
                        isReviewingResult = true
                        editorFolded = false
                    }
                } else {
                    var editedText = text
                    if let draft, requestMode != .ask {
                        unappliedResult = text
                        editedText = try draft.replacing(with: text, in: resultDraft)
                        unappliedResult = ""
                    }
                    let entry = HistoryEntry(mode: requestMode, appName: lastApp, rawText: raw, text: editedText,
                                             duration: duration, warning: warning.isEmpty ? nil : warning)
                    if requestMode == .ask { answerHistoryID = entry.id }
                    else { resultHistoryID = review ? entry.id : nil }
                    if let retentionError = store.add(entry, recording: audio) {
                        retryRecording = recording
                        self.error = retentionError
                    }
                    notice = warning
                    if review {
                        resultText = editedText
                        if requestMode == .ask {
                            answerText = text
                            answerTarget = allowInsertion ? (capturedTarget ?? answerTarget ?? reviewTarget) : nil
                        } else {
                            resultDraft = editedText
                            draftSelection = NSRange(location: (draft?.range.location ?? 0) + (text as NSString).length, length: 0)
                            if draft == nil { reviewTarget = allowInsertion ? capturedTarget : nil }
                        }
                        isReviewingResult = true
                        editorFolded = false
                    } else if allowInsertion, preferences.autoPaste, requestMode != .ask, let capturedTarget {
                        do {
                            try await insertText(text, capturedTarget, false)
                            guard generation == token else { return }
                            if notice.isEmpty { notice = L("notice.inserted", capturedTarget.applicationName) }
                        } catch {
                            notice = L("notice.readyToCopy", error.localizedDescription)
                            store.note(error.localizedDescription, on: entry.id)
                            Diagnostics.record("insert.failed", ["destination": capturedTarget.bundleID,
                                                                 "reason": Diagnostics.code(of: error)])
                            showMainWindow?()
                        }
                    } else {
                        if notice.isEmpty { notice = requestMode == .ask ? L("notice.answerReady") : L("notice.textReady") }
                        showMainWindow?()
                    }
                }
                guard generation == token else { return }
                target = nil; draftOperation = nil; isEditingEntireField = false; phase = .idle
                if self.error.isEmpty && !isReviewingResult { dismissVoicePanel() }
                else { presentVoicePanel() }
                releaseIdleModel()
            } catch {
                guard generation == token else { try? FileManager.default.removeItem(at: audio); return }
                speechStream?.cancel(); speechStream = nil
                retryRecording = recording
                phase = .idle
                setError(error)
                Diagnostics.record("dictation.failed", ["reason": Diagnostics.code(of: error)])
                if let raw = (error as? WorkerFailure)?.rawText, !raw.isEmpty {
                    rawText = raw; resultText = raw
                    if requestMode == .ask { questionText = raw }
                    notice = L("notice.textFailed")
                }
                presentVoicePanel()
                releaseIdleModel()
            }
        }
    }

    var canRetry: Bool { retryRecording != nil && !isBusy && (!isReviewingResult || retryRecording?.draft != nil) }
    func retryLast(mode requestedMode: VoiceMode? = nil, verbatim: Bool? = nil) {
        guard canRetry, let recording = retryRecording else { return }
        retryRecording = nil
        target = recording.target; mode = requestedMode ?? recording.mode; lastApp = recording.appName
        draftOperation = recording.draft
        verbatimDictation = verbatim ?? recording.verbatim
        sessionPreferences = store.preferences
        sessionAPIKey = textAPIKey
        run(audio: recording.url, duration: recording.duration, allowInsertion: false)
    }

    func retry(_ entry: HistoryEntry) {
        guard phase == .idle, !isReviewingResult, let audio = store.audioURL(for: entry) else { return }
        if entry.mode == .edit || entry.mode == .ask {
            error = L("error.retryNeedsRecording")
            return
        }
        do {
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent("OmniTyper-\(UUID()).wav")
            try FileManager.default.copyItem(at: audio, to: copy)
            discardRetryRecording()
            target = nil; mode = entry.mode; verbatimDictation = false
            lastApp = entry.appName; sessionPreferences = store.preferences
            sessionAPIKey = textAPIKey
            run(audio: copy, duration: entry.duration, allowInsertion: false)
        } catch { self.error = error.localizedDescription }
    }

    func discardRetryRecording() {
        if let retryRecording { try? FileManager.default.removeItem(at: retryRecording.url) }
        retryRecording = nil
    }
}
