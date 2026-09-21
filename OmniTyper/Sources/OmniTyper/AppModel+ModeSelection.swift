// SPDX-License-Identifier: Apache-2.0
import AppKit

extension AppModel {
    var canSelectMode: Bool {
        !isSelectingMode && (phase == .idle || (!isReviewingResult && (phase == .starting || phase == .recording)))
    }
    var showsEditor: Bool {
        !isSelectingMode && ((isReviewingResult && !editorFolded)
            || (isBusy ? sessionPreferences : store.preferences).recordingPresentation == .edit)
    }
    var usesCompactPanel: Bool {
        !showsEditor && (error.isEmpty || needsEditSelection)
    }
    var compactSelectionGuidance: String? {
        guard !isSelectingMode, usesCompactPanel, mode == .edit else { return nil }
        if needsEditSelection { return error }
        return isEditingEntireField ? L("panel.editingAllText") : nil
    }

    var selectionDisplayTarget: CGFloat {
        selectionExpansion * selectionExpansion * (3 - 2 * selectionExpansion)
    }

    func shortcutPressed(pointerX: CGFloat = NSEvent.mouseLocation.x, pointerY: CGFloat = NSEvent.mouseLocation.y) {
        if phase == .recording { finish(); return }
        guard phase == .idle, !isSelectingMode else { return }
        isReselectingMode = false
        selectionAllowsModeChange = true
        sessionPreferences = store.preferences
        positionModeSelector(at: NSPoint(x: pointerX, y: pointerY))
        error = ""; notice = ""
        isSelectingMode = true
        presentVoicePanel()
        modeSelectionTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 16_000_000)
                guard !Task.isCancelled, let self, self.isSelectingMode else { return }
                self.updateModeSelection(pointerX: NSEvent.mouseLocation.x, pointerY: NSEvent.mouseLocation.y)
            }
        }
    }

    func beginPanelSelection(at pointer: NSPoint) {
        guard !isSelectingMode, phase != .processing else { return }
        selectionAllowsModeChange = canSelectMode
        positionModeSelector(at: pointer)
        isReselectingMode = true
        isSelectingMode = true
        presentVoicePanel()
    }

    private func positionModeSelector(at pointer: NSPoint) {
        selectionInitialMode = mode
        selectionInitialExpansion = showsEditor ? 1 : 0
        selectionExpansion = selectionInitialExpansion
        selectionPointerX = pointer.x; selectionPointerY = pointer.y
        selectionPointerMoved = false
        selectionTrackX = pointer.x - ModeSelectionPanel.centerX(for: mode)
        selectionPanelBottom = pointer.y - VoicePanel.selectorSize.height / 2
        selectionExpansionStartY = selectionPanelBottom + VoicePanel.selectorSize.height
        selectionBarOrigin = NSPoint(x: selectionTrackX, y: selectionPanelBottom)
    }

    func endPanelSelection(at pointer: NSPoint) {
        guard isSelectingMode, isReselectingMode else { return }
        updateModeSelection(pointerX: pointer.x, pointerY: pointer.y)
        isSelectingMode = false
        isReselectingMode = false
        store.preferences.recordingPresentation = selectionExpansion >= 0.5 ? .edit : .simple
        sessionPreferences.recordingPresentation = store.preferences.recordingPresentation
        editorFolded = selectionExpansion < 0.5
        if selectionAllowsModeChange { selectMode(mode) }
        presentVoicePanel()
    }

    func foldEditor() {
        guard !isSelectingMode else { return }
        if isReviewingResult && !isBusy && !reviewContent.isEmpty { copyReviewedResult() }
        editorFolded = true
        store.preferences.recordingPresentation = .simple
        sessionPreferences.recordingPresentation = .simple
        presentVoicePanel()
    }

    func updateModeSelection(pointerX: CGFloat, pointerY: CGFloat? = nil) {
        guard isSelectingMode else { return }
        if pointerX != selectionPointerX || (pointerY ?? selectionPointerY) != selectionPointerY { selectionPointerMoved = true }
        if selectionAllowsModeChange && selectionPointerMoved, let selection = VoiceMode.allCases.first(where: { candidate in
            let centerX = selectionBarOrigin.x + ModeSelectionPanel.centerX(for: candidate)
            let centerY = selectionBarOrigin.y + VoicePanel.selectorSize.height / 2
            let radius = ModeSelectionPanel.markerDiameter / 2 * (candidate == mode ? ModeSelectionPanel.selectedScale : 1)
            return hypot(pointerX - centerX, (pointerY ?? selectionPointerY) - centerY) <= radius
        }) {
            if mode != selection { mode = selection }
        }
        if let pointerY {
            let origin = selectionInitialExpansion == 0 ? selectionExpansionStartY : selectionPointerY
            let expansion = min(max(selectionInitialExpansion
                + (pointerY - origin) / Self.presentationSelectionDistance, 0), 1)
            if selectionExpansion != expansion { selectionExpansion = expansion; showVoicePanel?() }
        }
    }

    func shortcutReleased(pointerX: CGFloat = NSEvent.mouseLocation.x, pointerY: CGFloat = NSEvent.mouseLocation.y) {
        if isSelectingMode && !isReselectingMode {
            updateModeSelection(pointerX: pointerX, pointerY: pointerY)
            store.preferences.recordingPresentation = selectionExpansion >= 0.5 ? .edit : .simple
            editorFolded = selectionExpansion < 0.5
            isSelectingMode = false
            modeSelectionTask?.cancel(); modeSelectionTask = nil
            verbatimDictation = false
            start()
        }
    }
}
