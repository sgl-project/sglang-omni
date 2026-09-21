// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

struct DraftSelection {
    let text: String
    let range: NSRange

    func selectedText() throws -> String {
        guard range.location >= 0, range.length >= 0, range.location <= (text as NSString).length,
              range.length <= (text as NSString).length - range.location else { throw Failure("sys.badSelection") }
        let start = text.utf16.index(text.utf16.startIndex, offsetBy: range.location)
        let end = text.utf16.index(start, offsetBy: range.length)
        guard start.samePosition(in: text.unicodeScalars) != nil,
              end.samePosition(in: text.unicodeScalars) != nil else { throw Failure("sys.badSelection") }
        return (text as NSString).substring(with: range)
    }

    func replacing(with output: String, in currentText: String) throws -> String {
        guard text == currentText else { throw Failure("panel.draftChanged") }
        _ = try selectedText()
        return (text as NSString).replacingCharacters(in: range, with: output)
    }
}

struct ResultEditor: NSViewRepresentable {
    @ObservedObject var model: AppModel

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = ResultTextView.scrollableTextView()
        let text = scroll.documentView as! ResultTextView
        text.isRichText = false
        text.isAutomaticQuoteSubstitutionEnabled = false
        text.isAutomaticDashSubstitutionEnabled = false
        text.allowsUndo = true
        text.font = .systemFont(ofSize: 14)
        text.textContainerInset = NSSize(width: 8, height: 8)
        text.delegate = context.coordinator
        model.focusDraft = { [weak text] in
            guard let text else { return }
            text.window?.makeFirstResponder(text)
        }
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        let text = scroll.documentView as! NSTextView
        context.coordinator.update(text)
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        let model: AppModel
        var updating = false
        init(model: AppModel) { self.model = model }

        func update(_ text: NSTextView) {
            updating = true
            defer { updating = false }
            text.setAccessibilityLabel(L("panel.result"))
            text.isEditable = !model.isBusy && model.isReviewingResult
            text.isSelectable = !model.isBusy
            if text.string != model.resultDraft {
                text.textStorage?.setAttributedString(NSAttributedString(string: model.resultDraft, attributes: [
                    .font: NSFont.systemFont(ofSize: 14), .foregroundColor: NSColor.labelColor
                ]))
                text.undoManager?.removeAllActions()
            }
            let length = (text.string as NSString).length
            let location = min(max(model.draftSelection.location, 0), length)
            let range = NSRange(location: location, length: min(max(model.draftSelection.length, 0), length - location))
            if text.selectedRange() != range { text.setSelectedRange(range); text.scrollRangeToVisible(range) }
            text.typingAttributes = [.font: NSFont.systemFont(ofSize: 14), .foregroundColor: NSColor.labelColor]
            if let text = text as? ResultTextView {
                text.placeholder = model.liveText.isEmpty ? L("panel.editorPlaceholder") : model.liveText
                text.needsDisplay = true
            }
        }

        func textDidChange(_ notification: Notification) {
            guard !updating, let text = notification.object as? NSTextView else { return }
            model.resultDraft = text.string
            model.draftSelection = text.selectedRange()
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            guard !updating, !model.isBusy, let text = notification.object as? NSTextView else { return }
            if model.draftSelection != text.selectedRange() { model.draftSelection = text.selectedRange() }
        }
    }
}

final class ResultTextView: NSTextView {
    var placeholder = ""

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard string.isEmpty, !placeholder.isEmpty, let textContainer else { return }
        let storage = NSTextStorage(string: placeholder, attributes: [
            .font: font ?? NSFont.systemFont(ofSize: 14), .foregroundColor: NSColor.placeholderTextColor
        ])
        let layout = NSLayoutManager()
        let container = NSTextContainer(size: textContainer.containerSize)
        container.lineFragmentPadding = textContainer.lineFragmentPadding
        storage.addLayoutManager(layout)
        layout.addTextContainer(container)
        layout.drawGlyphs(forGlyphRange: layout.glyphRange(for: container), at: textContainerOrigin)
    }
}
