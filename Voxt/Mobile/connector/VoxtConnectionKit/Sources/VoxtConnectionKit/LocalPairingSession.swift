import CryptoKit
import Foundation

/// Serves phones that found the Connector on the same Wi-Fi during one Add iPhone window.
/// One request at a time; no answer before the window expires counts as Don't Allow; Cancel
/// drops a pending request. The user sees the same 6-digit code on the phone and in the sheet.
public actor LocalPairingCoordinator {
    private let invitation: URL
    private let expiresAt: Date
    private let onRequest: @Sendable (_ code: String, _ deviceName: String) async -> Bool
    private var busy = false
    private var cancelled = false
    private let pending = PendingDecision()

    public init(invitation: URL, expiresAt: Date, onRequest: @escaping @Sendable (String, String) async -> Bool) {
        self.invitation = invitation
        self.expiresAt = expiresAt
        self.onRequest = onRequest
    }

    public func serve(_ channel: some LocalPairingChannel) async {
        guard !cancelled, Date() < expiresAt, !busy else {
            try? await channel.send(LocalPairingWire.denied(reason: !cancelled && Date() >= expiresAt ? .expired : nil))
            return
        }
        busy = true
        defer { busy = false }
        do {
            let hello = try await channel.readLine()
            let phoneKey = try LocalPairingWire.decodeHello(hello)
            let device = ((try? JSONSerialization.jsonObject(with: hello) as? [String: Any])?["device"] as? String)
                .map { String($0.prefix(64)) } ?? "iPhone"
            let macKey = Curve25519.KeyAgreement.PrivateKey()
            try await channel.send(LocalPairingWire.macHello(publicKey: macKey.publicKey.rawRepresentation))
            let crypto = try LocalPairingCrypto(privateKey: macKey, peerPublicKey: phoneKey, isPhone: false)
            let allowed = await decide(code: crypto.code, device: device)
            if allowed, !cancelled, Date() < expiresAt {
                try await channel.send(LocalPairingWire.approved(sealed: crypto.seal(invitation)))
            } else {
                // The phone tells "Don't Allow" from "no answer before the window closed".
                let expired = !cancelled && Date() >= expiresAt
                try await channel.send(LocalPairingWire.denied(reason: expired ? .expired : nil))
            }
        } catch {
            try? await channel.send(LocalPairingWire.denied())
        }
    }

    /// Add iPhone was cancelled or closed: a pending request is answered Don't Allow.
    public func cancel() {
        cancelled = true
        pending.resolve(false)
    }

    private func decide(code: String, device: String) async -> Bool {
        let onRequest = onRequest, expiresAt = expiresAt, pending = pending
        return await withCheckedContinuation { continuation in
            pending.arm(continuation)
            let ask = Task { pending.resolve(await onRequest(code, device)) }
            pending.onResolve = { ask.cancel() }
            Task {
                try? await Task.sleep(for: .seconds(max(0, expiresAt.timeIntervalSinceNow)))
                pending.resolve(false)
            }
        }
    }
}

/// Resumes the decision exactly once: the user's answer, the expiry, or Cancel.
private final class PendingDecision: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Bool, Never>?
    var onResolve: (() -> Void)?

    func arm(_ continuation: CheckedContinuation<Bool, Never>) {
        lock.withLock { self.continuation = continuation }
    }

    func resolve(_ value: Bool) {
        let (waiting, hook) = lock.withLock { () -> (CheckedContinuation<Bool, Never>?, (() -> Void)?) in
            defer { continuation = nil; onResolve = nil }
            return (continuation, onResolve)
        }
        guard let waiting else { return }
        hook?()
        waiting.resume(returning: value)
    }
}
