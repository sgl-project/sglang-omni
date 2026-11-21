// The mark on the keyboard events the helper posts itself (⌘V, ⌘C,
// Return): their eventSourceUserData. The hotkey tap lets marked events
// through without judging them, so a paste never triggers a shortcut.
//
// The mark is VoxtNew's own, "VOXT": Auralink, the same author's app with
// the same helper design, marks its events "AURA" (0x41555241). Two apps
// running together each judge the other's posted events as typing, and
// neither lets the other's through unjudged.

public enum InjectedEvent {
    /// eventSourceUserData of the events VoxtNew posts: "VOXT" in ASCII.
    public static let marker: Int64 = 0x564F_5854

    /// Auralink's mark, for tests and diagnostics only: not ours.
    public static let auralinkMarker: Int64 = 0x4155_5241

    /// Whether an event's eventSourceUserData marks it as VoxtNew's own.
    public static func isOurs(_ userData: Int64) -> Bool { userData == marker }
}
