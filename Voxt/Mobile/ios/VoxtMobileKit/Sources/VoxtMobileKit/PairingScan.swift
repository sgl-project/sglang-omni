import Foundation

/// The pairing screen's in-app QR scanner. Decoded text is checked by `PairingBootstrap`,
/// the same parser a Camera-app scan of the link goes through.
public struct PairingScan: Equatable, Sendable {
    public enum State: Equatable, Sendable { case starting, scanning, pairing, paired, cameraDenied }
    public enum Camera: Sendable { case notDetermined, authorized, denied }

    public private(set) var state: State = .starting
    /// One short line under the viewfinder: a wrong code, an expired code, or a failed pairing.
    public private(set) var hint: String?

    public init() {}

    public mutating func camera(_ access: Camera) {
        switch access {
        case .notDetermined: if state != .pairing && state != .paired { state = .starting }
        case .denied: if state != .pairing && state != .paired { state = .cameraDenied }
        case .authorized: if state == .starting || state == .cameraDenied { state = .scanning }
        }
    }

    /// Returns the pairing link to open, or nil when the text isn't a usable Voxt code.
    public mutating func decoded(_ text: String, now: Date = Date()) -> URL? {
        guard state == .scanning else { return nil }
        guard let url = URL(string: text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            hint = "This isn't a Voxt pairing code."
            return nil
        }
        do {
            _ = try PairingBootstrap(url: url, now: now)
        } catch PairingBootstrap.Error.expired {
            hint = "This code has expired. Show a new one in Voxt Connector."
            return nil
        } catch {
            hint = "This isn't a Voxt pairing code."
            return nil
        }
        hint = nil
        state = .pairing
        return url
    }

    /// `error` is the message to show when pairing failed; nil when it succeeded.
    public mutating func pairingFinished(error: String?) {
        guard state == .pairing else { return }
        if let error {
            state = .scanning
            hint = error
        } else {
            state = .paired
            hint = nil
        }
    }
}
