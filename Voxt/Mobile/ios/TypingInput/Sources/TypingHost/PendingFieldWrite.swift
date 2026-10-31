import Foundation

/// One text-proxy write awaiting acknowledgement from the host.
public struct PendingFieldWrite: Sendable {
    public let id: UUID
    public let origin: FieldContext
    public let expected: FieldContext
    private let omitted: FieldContext

    public init(id: UUID, origin: FieldContext, replacing count: Int, text: String) {
        self.id = id
        self.origin = origin
        let prefix = String(origin.before.dropLast(count))
        expected = .init(documentID: origin.documentID, before: prefix + text, after: origin.after, selected: "")
        omitted = .init(documentID: origin.documentID, before: prefix, after: origin.after, selected: "")
    }

    public func canUnmarkForAcknowledgement(_ current: FieldContext?) -> Bool {
        // A replacement deletes the exposed old reading before marking the new
        // one. A host which omits that mark reports the stripped prefix, not origin.
        current == origin || current == omitted
    }
}
