import AppKit

/// Temporary text writes own an observed change count, not the whole clipboard.
/// Newer observed copies invalidate restoration even for an identical string.
/// This retains text-only snapshots; NSPasteboard has no cross-process CAS.
@MainActor
final class PasteboardTextWriter {
    @MainActor
    struct Restoration {
        fileprivate let id: UUID
        fileprivate let pasteboard: NSPasteboard
        fileprivate let changeCount: Int
        fileprivate let previousText: String?
    }

    private var pending: Restoration?

    func write(_ text: String, to pasteboard: NSPasteboard, restorePrevious: Bool) -> Restoration? {
        let previousText: String?
        if !restorePrevious {
            previousText = nil
        } else if let pending,
                  pending.pasteboard.name == pasteboard.name,
                  pending.changeCount == pasteboard.changeCount {
            // Overlapping pastes inherit the user's original text, not the
            // previous temporary result that happens to be on the clipboard.
            previousText = pending.previousText
        } else {
            previousText = readStringFromPasteboard(pasteboard)
        }

        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        guard restorePrevious else { pending = nil; return nil }
        let restoration = Restoration(
            id: UUID(), pasteboard: pasteboard,
            changeCount: pasteboard.changeCount, previousText: previousText
        )
        pending = restoration
        return restoration
    }

    @discardableResult
    func restoreIfUnchanged(_ restoration: Restoration) -> Bool {
        guard pending?.id == restoration.id else { return false }
        pending = nil
        let pasteboard = restoration.pasteboard
        guard pasteboard.changeCount == restoration.changeCount else { return false }
        pasteboard.clearContents()
        if let previousText = restoration.previousText, !previousText.isEmpty {
            pasteboard.setString(previousText, forType: .string)
        }
        return true
    }
}
