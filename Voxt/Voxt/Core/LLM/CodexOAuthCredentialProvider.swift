// CodexOAuthCredentialProvider.swift
// Provides Codex OAuth Credential Provider for LLM requests and response handling.

import Darwin
import Foundation

struct CodexOAuthCredentialProvider {
    struct Credential {
        let accessToken: String
        let accountID: String?
    }

    enum CredentialError: LocalizedError {
        case authFileNotFound(String)
        case authFilePermissionDenied(String)
        case missingTokens
        case invalidAuthFile(String)
        case refreshFailed(Int, String)

        var errorDescription: String? {
            switch self {
            case .authFileNotFound(let path):
                return AppLocalization.format("Codex auth not found at %@. Run `codex login` first.", path)
            case .authFilePermissionDenied(let path):
                return AppLocalization.format("Codex auth.json permission denied at %@. Click Choose and select auth.json to grant Voxt access.", path)
            case .missingTokens:
                return AppLocalization.localizedString("Codex auth.json has no ChatGPT OAuth tokens. Run `codex login` first.")
            case .invalidAuthFile(let detail):
                return AppLocalization.format("Invalid Codex auth.json: %@", detail)
            case .refreshFailed(let status, let body):
                return AppLocalization.format("Codex token refresh failed (HTTP %d). %@ Run `codex login` again.", status, body)
            }
        }
    }

    private struct AuthFile: Codable {
        var authMode: String?
        var openAIAPIKey: String?
        var tokens: Tokens?
        var lastRefresh: String?

        enum CodingKeys: String, CodingKey {
            case authMode = "auth_mode"
            case openAIAPIKey = "OPENAI_API_KEY"
            case tokens
            case lastRefresh = "last_refresh"
        }
    }

    private struct Tokens: Codable {
        var idToken: String?
        var accessToken: String?
        var refreshToken: String?
        var accountID: String?

        enum CodingKeys: String, CodingKey {
            case idToken = "id_token"
            case accessToken = "access_token"
            case refreshToken = "refresh_token"
            case accountID = "account_id"
        }
    }

    private struct RefreshResponse: Codable {
        var idToken: String?
        var accessToken: String?
        var refreshToken: String?

        enum CodingKeys: String, CodingKey {
            case idToken = "id_token"
            case accessToken = "access_token"
            case refreshToken = "refresh_token"
        }
    }

    private let fileManager: FileManager
    private let environment: [String: String]
    private let userHomeDirectory: String
    private let authFilePathOverride: String
    private let authFileBookmark: Data?
    private let refreshURL = URL(string: "https://auth.openai.com/oauth/token")!
    private let clientID = "app_EMoamEEZ73f0CkXaXp7hrann"

    init(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        userHomeDirectory: String? = nil,
        authFilePath: String = "",
        authFileBookmark: Data? = nil
    ) {
        self.fileManager = fileManager
        self.environment = environment
        self.userHomeDirectory = userHomeDirectory ?? Self.defaultUserHomeDirectory(
            environment: environment,
            fileManager: fileManager
        )
        self.authFilePathOverride = authFilePath
        self.authFileBookmark = authFileBookmark
    }

    func authorizationHeaders() async throws -> [String: String] {
        let credential = try await credential()
        var headers = [
            "Authorization": "Bearer \(credential.accessToken)",
            "User-Agent": "codex_cli_rs/0.0.0 (Voxt)",
            "originator": "codex_cli_rs"
        ]
        if let accountID = credential.accountID, !accountID.isEmpty {
            headers["ChatGPT-Account-ID"] = accountID
        }
        return headers
    }

    func credential() async throws -> Credential {
        let access = authFileAccess()
        let url = access?.url ?? URL(fileURLWithPath: authFilePath(), isDirectory: false)
        defer { withExtendedLifetime(access) {} }
        var auth = try readAuthFile(url: url)
        guard let tokens = auth.tokens,
              let accessToken = tokens.accessToken,
              !accessToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw CredentialError.missingTokens
        }

        if !isTokenExpired(accessToken, lastRefresh: auth.lastRefresh) {
            return Credential(
                accessToken: accessToken,
                accountID: tokens.accountID ?? chatGPTAccountID(from: accessToken)
            )
        }

        guard let refreshToken = tokens.refreshToken,
              !refreshToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw CredentialError.missingTokens
        }

        let refreshed = try await refreshTokens(refreshToken: refreshToken)
        let newAccessToken = refreshed.accessToken ?? accessToken
        auth.tokens = Tokens(
            idToken: refreshed.idToken ?? tokens.idToken,
            accessToken: newAccessToken,
            refreshToken: refreshed.refreshToken ?? refreshToken,
            accountID: tokens.accountID
        )
        auth.lastRefresh = ISO8601DateFormatter().string(from: Date())
        try writeAuthFile(auth, url: url)

        return Credential(
            accessToken: newAccessToken,
            accountID: auth.tokens?.accountID ?? chatGPTAccountID(from: newAccessToken)
        )
    }

    func authFilePath() -> String {
        let overridePath = authFilePathOverride.trimmingCharacters(in: .whitespacesAndNewlines)
        if !overridePath.isEmpty {
            return expandedPath(overridePath)
        }
        if let codexHome = environment["CODEX_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !codexHome.isEmpty {
            return (expandedPath(codexHome) as NSString).appendingPathComponent("auth.json")
        }
        return (userHomeDirectory as NSString).appendingPathComponent(".codex/auth.json")
    }

    private func authFileAccess() -> SecurityScopedBookmarkSupport.Access? {
        SecurityScopedBookmarkSupport.accessFileURL(
            bookmarkData: authFileBookmark,
            fallbackPath: authFilePath()
        )
    }

    private static func defaultUserHomeDirectory(
        environment: [String: String],
        fileManager: FileManager
    ) -> String {
        if let path = posixUserHomeDirectory(), !path.isEmpty {
            return path
        }
        if let home = environment["HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !home.isEmpty {
            return home
        }
        return fileManager.homeDirectoryForCurrentUser.path
    }

    private static func posixUserHomeDirectory() -> String? {
        guard let passwd = getpwuid(getuid()),
              let directory = passwd.pointee.pw_dir else {
            return nil
        }
        return String(cString: directory)
    }

    private func expandedPath(_ path: String) -> String {
        if path == "~" {
            return userHomeDirectory
        }
        if path.hasPrefix("~/") {
            return (userHomeDirectory as NSString).appendingPathComponent(String(path.dropFirst(2)))
        }
        return path
    }

    private func readAuthFile(url: URL) throws -> AuthFile {
        let path = url.path
        guard fileManager.fileExists(atPath: path) else {
            throw CredentialError.authFileNotFound(path)
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(AuthFile.self, from: data)
        } catch let error as CredentialError {
            throw error
        } catch where Self.isPermissionDenied(error) {
            throw CredentialError.authFilePermissionDenied(path)
        } catch {
            throw CredentialError.invalidAuthFile(error.localizedDescription)
        }
    }

    private func writeAuthFile(_ auth: AuthFile, url: URL) throws {
        let path = url.path
        do {
            var object: [String: Any] = [:]
            if let existingData = try? Data(contentsOf: url),
               let existingObject = try? JSONSerialization.jsonObject(with: existingData) as? [String: Any] {
                object = existingObject
            }

            if let authMode = auth.authMode {
                object["auth_mode"] = authMode
            }
            if let openAIAPIKey = auth.openAIAPIKey {
                object["OPENAI_API_KEY"] = openAIAPIKey
            }
            if let tokens = auth.tokens {
                var tokenObject = object["tokens"] as? [String: Any] ?? [:]
                if let idToken = tokens.idToken {
                    tokenObject["id_token"] = idToken
                }
                if let accessToken = tokens.accessToken {
                    tokenObject["access_token"] = accessToken
                }
                if let refreshToken = tokens.refreshToken {
                    tokenObject["refresh_token"] = refreshToken
                }
                if let accountID = tokens.accountID {
                    tokenObject["account_id"] = accountID
                }
                object["tokens"] = tokenObject
            }
            if let lastRefresh = auth.lastRefresh {
                object["last_refresh"] = lastRefresh
            }

            let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
            try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
            try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path)
        } catch where Self.isPermissionDenied(error) {
            throw CredentialError.authFilePermissionDenied(path)
        } catch {
            throw CredentialError.invalidAuthFile(error.localizedDescription)
        }
    }

    private static func isPermissionDenied(_ error: Error) -> Bool {
        let nsError = error as NSError
        if nsError.domain == NSCocoaErrorDomain,
           nsError.code == NSFileReadNoPermissionError || nsError.code == NSFileWriteNoPermissionError {
            return true
        }
        if nsError.domain == NSPOSIXErrorDomain,
           nsError.code == Int(EACCES) || nsError.code == Int(EPERM) {
            return true
        }
        if let underlying = nsError.userInfo[NSUnderlyingErrorKey] as? Error {
            return isPermissionDenied(underlying)
        }
        return false
    }

    private func isTokenExpired(_ accessToken: String, lastRefresh: String?) -> Bool {
        if let expiration = jwtExpiration(accessToken),
           expiration <= Date().timeIntervalSince1970 + 60 {
            return true
        }
        guard let lastRefresh else { return false }
        guard let date = ISO8601DateFormatter().date(from: lastRefresh) else { return false }
        return Date().timeIntervalSince(date) > 8 * 24 * 60 * 60
    }

    private func refreshTokens(refreshToken: String) async throws -> RefreshResponse {
        var request = URLRequest(url: refreshURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 20
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "client_id": clientID,
            "grant_type": "refresh_token",
            "refresh_token": refreshToken
        ])

        let (data, response) = try await VoxtNetworkSession.active.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw CredentialError.refreshFailed(-1, "")
        }
        guard (200...299).contains(http.statusCode) else {
            let body = String(data: data.prefix(240), encoding: .utf8) ?? ""
            throw CredentialError.refreshFailed(http.statusCode, body)
        }
        return try JSONDecoder().decode(RefreshResponse.self, from: data)
    }

    private func jwtExpiration(_ token: String) -> TimeInterval? {
        guard let payload = jwtPayload(token) else {
            return nil
        }
        if let expiration = payload["exp"] as? TimeInterval {
            return expiration
        }
        if let expiration = payload["exp"] as? NSNumber {
            return expiration.doubleValue
        }
        return nil
    }

    private func chatGPTAccountID(from token: String) -> String? {
        guard let payload = jwtPayload(token),
              let auth = payload["https://api.openai.com/auth"] as? [String: Any],
              let accountID = auth["chatgpt_account_id"] as? String
        else {
            return nil
        }
        return accountID
    }

    private func jwtPayload(_ token: String) -> [String: Any]? {
        let parts = token.split(separator: ".")
        guard parts.count >= 2 else { return nil }
        var encoded = String(parts[1])
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let padding = encoded.count % 4
        if padding > 0 {
            encoded += String(repeating: "=", count: 4 - padding)
        }
        guard let data = Data(base64Encoded: encoded),
              let object = try? JSONSerialization.jsonObject(with: data),
              let payload = object as? [String: Any]
        else {
            return nil
        }
        return payload
    }
}
