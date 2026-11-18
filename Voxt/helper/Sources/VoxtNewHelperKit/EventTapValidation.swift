// macOS can create an event tap while stripping the keyboard events the
// app is not authorized to see (Accessibility granted, Input Monitoring
// not): the tap is nonnil, modifier flags still arrive, plain keys never
// do. A created tap is therefore not proof of a working one (Voxt's #2996
// hit exactly this). The check reads the new tap's granted mask back from
// the system's tap list and refuses a partial or unverifiable tap.

import CoreGraphics
import Foundation

public enum EventTapValidation {
    /// The taps currently installed, or nil when the list cannot be read.
    /// Reading the list needs no permission and installs no monitor.
    public static func currentTaps() -> [CGEventTapInformation]? {
        var count: UInt32 = 0
        guard CGGetEventTapList(0, nil, &count) == .success else { return nil }
        guard count > 0 else { return [] }
        var taps = [CGEventTapInformation](repeating: CGEventTapInformation(), count: Int(count))
        guard CGGetEventTapList(count, &taps, &count) == .success else { return nil }
        return Array(taps.prefix(Int(count)))
    }

    /// Whether exactly one new tap of this process at this location kept
    /// every requested event. Another process's tap, another location, a
    /// listen-only tap or a pre-existing one proves nothing about ours;
    /// missing or ambiguous metadata is not a ready tap either.
    public static func newTapHasRequiredEvents(
        eventMask: CGEventMask,
        location: CGEventTapLocation,
        processIdentifier: pid_t,
        previousTapIDs: Set<UInt32>,
        installedTaps: [CGEventTapInformation]
    ) -> Bool {
        let candidates = installedTaps.filter {
            $0.tappingProcess == processIdentifier &&
                $0.tapPoint == location &&
                $0.options == .defaultTap &&
                !previousTapIDs.contains($0.eventTapID)
        }
        guard candidates.count == 1, let installed = candidates.first else { return false }
        return installed.eventsOfInterest & eventMask == eventMask
    }
}
