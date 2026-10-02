import Foundation

@main
private enum RevisionConfirmationTests {
    @MainActor
    static func main() async throws {
        let before = TextDraft(value: "前", selection: NSRange(location: 1, length: 0))
        var reads = 0
        let anchor = EditableTextAnchor(text: "听写", before: before, read: {
            reads += 1
            return TextDraft(value: "前听写", selection: NSRange(location: reads == 1 ? 1 : 3, length: 0))
        }, validateIdentity: {}, select: { _ in fatalError("Must not select during confirmation") },
        makePaste: { fatalError("Must not paste during confirmation") }, deleteSelection: { fatalError("Must not delete") })
        let confirmed = await anchor.confirmInsertion()
        precondition(confirmed, "Text and caret can arrive in separate accessibility updates")
        anchor.cancel()
        let changed = EditableTextAnchor(text: "听写", before: before,
                                         read: { TextDraft(value: "用户改动", selection: NSRange(location: 4, length: 0)) },
                                         validateIdentity: {}, select: { _ in fatalError() },
                                         makePaste: { fatalError() }, deleteSelection: { fatalError() })
        let rejected = await changed.confirmInsertion()
        precondition(!rejected, "Changed content must not create a correction anchor")
        changed.cancel()
        print("PASS: delayed caret confirmation without accepting changed text")
    }
}
