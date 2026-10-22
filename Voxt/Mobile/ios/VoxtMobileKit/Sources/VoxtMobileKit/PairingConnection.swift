import Foundation

/// One saved pairing's request lifetime. Revocation is synchronous on the app's
/// main actor, so a late resolver result cannot revive an old pairing.
@MainActor public final class PairingConnection {
    private let server: ServerConfig
    private let resolve: (@Sendable () async throws -> URL)?
    private let reset: @Sendable () async -> Void
    private let close: @Sendable () async -> Void
    private var resetting: Task<Void, Never>?
    private var lastEndpoint: URL?
    private var revoked = false
    private var closing: Task<Void, Never>?
    /// Shared by every client of this pairing so consecutive dictations skip the pre-checks.
    private let preflight = PreflightCache()

    public init(server: ServerConfig, resolve: (@Sendable () async throws -> URL)? = nil,
                reset: @escaping @Sendable () async -> Void = {},
                close: @escaping @Sendable () async -> Void = {}) {
        self.server = server
        self.resolve = resolve
        self.close = close
        self.reset = reset
    }

    deinit {
        if closing == nil {
            let close = close
            Task { await close() }
        }
    }

    public func endpoint() async throws -> URL {
        await resetting?.value
        try Task.checkCancellation()
        guard !revoked else { throw CancellationError() }
        let url: URL
        if let resolve { url = try await resolve() }
        else {
            guard server.tailcat == nil else { throw VoxtClient.Error.transportUnavailable }
            url = server.baseURL
        }
        try Task.checkCancellation()
        guard !revoked else { throw CancellationError() }
        lastEndpoint = url
        return url
    }

    public func client(session: URLSession = .shared, allowCloudModels: @escaping @MainActor @Sendable () -> Bool = { false },
                       customPrompt: @escaping @MainActor @Sendable (CompositionOptions.Mode) -> String? = { _ in nil }) -> VoxtClient {
        VoxtClient(server: server, session: session, resolveBaseURL: { [self] in try await endpoint() },
                   onTransportFailure: { [self] url in await recover(failedEndpoint: url) }, allowCloudModels: allowCloudModels,
                   customPrompt: customPrompt, preflight: preflight)
    }

    private func recover(failedEndpoint: URL) async {
        guard !revoked else { return }
        if let resetting { await resetting.value; return }
        guard let lastEndpoint, lastEndpoint.scheme == failedEndpoint.scheme,
              lastEndpoint.host == failedEndpoint.host, lastEndpoint.port == failedEndpoint.port else { return }
        self.lastEndpoint = nil
        preflight.invalidate()
        let task = Task { await reset() }
        resetting = task
        await task.value
        resetting = nil
    }

    @discardableResult public func invalidate() -> Task<Void, Never> {
        if let closing { return closing }
        revoked = true
        let close = close
        let closing = Task { await close() }
        self.closing = closing
        return closing
    }
}
