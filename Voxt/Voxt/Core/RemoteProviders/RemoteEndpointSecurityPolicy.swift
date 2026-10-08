// RemoteEndpointSecurityPolicy.swift
// Validates user-supplied remote endpoints before credentials are attached.

import Foundation

enum RemoteEndpointSecurityPolicy {
    static func hasExplicitCredentials(_ configuration: RemoteProviderConfiguration) -> Bool {
        !configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !configuration.accessToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !configuration.appID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || configuration.hasStoredCredential(for: .apiKey)
            || configuration.hasStoredCredential(for: .accessToken)
            || configuration.hasStoredCredential(for: .appID)
    }

    static func hasLLMCredentials(
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration
    ) -> Bool {
        provider == .codex || hasExplicitCredentials(configuration)
    }

    static func validationMessage(
        endpoint: String,
        hasCredentials: Bool,
        allowsWebSocket: Bool = false
    ) -> String? {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard let components = URLComponents(string: trimmed),
              let scheme = components.scheme?.lowercased(),
              let host = components.host,
              !host.isEmpty,
              (["https", "http"] + (allowsWebSocket ? ["wss", "ws"] : [])).contains(scheme)
        else {
            return AppLocalization.localizedString(
                allowsWebSocket
                    ? "Endpoint must be a valid HTTP, HTTPS, WS, or WSS URL."
                    : "Endpoint must be a valid HTTP or HTTPS URL."
            )
        }
        if components.user != nil || components.password != nil {
            return AppLocalization.localizedString("Endpoint must not contain a username or password.")
        }
        if (scheme == "http" || scheme == "ws"), hasCredentials, !isLoopbackHost(host) {
            return AppLocalization.localizedString("Insecure endpoints can receive credentials only on this Mac. Use HTTPS or WSS for remote hosts.")
        }
        return nil
    }

    static func sanitizedForLog(_ endpoint: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "<default>" }
        guard var components = URLComponents(string: trimmed),
              components.scheme != nil,
              components.host != nil
        else { return "<invalid endpoint>" }
        components.user = nil
        components.password = nil
        components.query = nil
        components.fragment = nil
        return components.string ?? "<invalid endpoint>"
    }

    private static func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        return normalized == "localhost" || normalized == "127.0.0.1" || normalized == "::1"
    }
}
