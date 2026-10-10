// RemoteProviderConfiguration.swift
// Provides Remote Provider Configuration for remote provider configuration.

import Foundation

fileprivate nonisolated struct RemoteStoredCredentialPresence: Codable, Hashable {
    var apiKey: Bool
    var appID: Bool
    var accessToken: Bool

    init(apiKey: String, appID: String, accessToken: String) {
        self.apiKey = Self.isPresent(apiKey)
        self.appID = Self.isPresent(appID)
        self.accessToken = Self.isPresent(accessToken)
    }

    private static func isPresent(_ value: String) -> Bool {
        !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var isEmpty: Bool {
        !apiKey && !appID && !accessToken
    }
}

struct RemoteProviderConfiguration: Codable, Identifiable, Hashable {
    nonisolated enum CredentialField: CaseIterable, Hashable {
        case apiKey
        case appID
        case accessToken
    }

    let providerID: String
    var model: String
    var endpoint: String
    var apiKey: String
    var appID: String
    var accessToken: String
    fileprivate var storedCredentialPresence: RemoteStoredCredentialPresence?
    var searchEnabled: Bool
    var openAIChunkPseudoRealtimeEnabled: Bool
    var openAIReasoningEffort: String
    var openAITextVerbosity: String
    var openAIMaxOutputTokens: Int?
    var doubaoDictionaryMode: String
    var doubaoEnableRequestHotwords: Bool
    var doubaoEnableRequestCorrections: Bool
    var ollamaResponseFormat: String
    var ollamaJSONSchema: String
    var ollamaThinkMode: String
    var ollamaKeepAlive: String
    var ollamaLogprobsEnabled: Bool
    var ollamaTopLogprobs: Int?
    var ollamaOptionsJSON: String
    var omlxResponseFormat: String
    var omlxJSONSchema: String
    var omlxIncludeUsageStreamOptions: Bool
    var omlxExtraBodyJSON: String
    var codexAuthFilePath: String
    var codexAuthFileBookmark: Data?
    var codexFastModeEnabled: Bool
    var aliyunASRSettings: AliyunASRModelSettings
    var generationSettings: LLMGenerationSettings

    var id: String { providerID }

    nonisolated func hasStoredCredential(for field: CredentialField) -> Bool {
        guard sensitiveValue(for: field).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        return credentialPresence(for: field)
    }

    /// Carries credential intent from the metadata-only configuration shown by
    /// the settings UI. An unedited empty field means "keep the stored value";
    /// an edited empty field means "clear it".
    nonisolated func applyingCredentialEditIntent(
        from original: RemoteProviderConfiguration,
        editedFields: Set<CredentialField>
    ) -> RemoteProviderConfiguration {
        var updated = self
        var presence = RemoteStoredCredentialPresence(
            apiKey: apiKey,
            appID: appID,
            accessToken: accessToken
        )

        for field in CredentialField.allCases where !editedFields.contains(field) {
            let shouldPreserve = original.credentialPresence(for: field)
            switch field {
            case .apiKey:
                presence.apiKey = shouldPreserve
            case .appID:
                presence.appID = shouldPreserve
            case .accessToken:
                presence.accessToken = shouldPreserve
            }
        }
        updated.storedCredentialPresence = presence
        return updated
    }

    private nonisolated func credentialPresence(for field: CredentialField) -> Bool {
        if !sensitiveValue(for: field).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return true
        }
        switch field {
        case .apiKey:
            return storedCredentialPresence?.apiKey == true
        case .appID:
            return storedCredentialPresence?.appID == true
        case .accessToken:
            return storedCredentialPresence?.accessToken == true
        }
    }

    private nonisolated func sensitiveValue(for field: CredentialField) -> String {
        switch field {
        case .apiKey:
            return apiKey
        case .appID:
            return appID
        case .accessToken:
            return accessToken
        }
    }

    var hasUsableModel: Bool {
        !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var isConfigured: Bool {
        if RemoteLLMProvider(rawValue: providerID)?.apiKeyIsOptional == true {
            return hasUsableModel
        }
        let hasAPIKey =
            !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            storedCredentialPresence?.apiKey == true
        let hasAppID =
            !appID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            storedCredentialPresence?.appID == true
        let hasAccessToken =
            !accessToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            storedCredentialPresence?.accessToken == true
        if providerID == RemoteASRProvider.doubaoASR.rawValue {
            return hasUsableModel && hasAppID && hasAccessToken
        }
        return hasUsableModel && (hasAPIKey || hasAccessToken)
    }

    var doubaoDictionaryModeValue: DoubaoDictionaryMode {
        DoubaoDictionaryMode(rawValue: doubaoDictionaryMode) ?? .requestScoped
    }

    var ollamaResponseFormatValue: OllamaResponseFormat {
        OllamaResponseFormat(rawValue: ollamaResponseFormat) ?? .plain
    }

    var ollamaThinkModeValue: OllamaThinkMode {
        OllamaThinkMode(rawValue: ollamaThinkMode) ?? .off
    }

    var omlxResponseFormatValue: OMLXResponseFormat {
        OMLXResponseFormat(rawValue: omlxResponseFormat) ?? .plain
    }

    var resolvedCodexAuthFilePath: String {
        codexAuthFilePath.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var openAIReasoningEffortValue: OpenAIReasoningEffort {
        OpenAIReasoningEffort(rawValue: openAIReasoningEffort) ?? .automatic
    }

    var openAITextVerbosityValue: OpenAITextVerbosity {
        OpenAITextVerbosity(rawValue: openAITextVerbosity) ?? .automatic
    }

    init(
        providerID: String,
        model: String,
        endpoint: String,
        apiKey: String,
        appID: String = "",
        accessToken: String = "",
        searchEnabled: Bool = false,
        openAIChunkPseudoRealtimeEnabled: Bool = false,
        openAIReasoningEffort: String = OpenAIReasoningEffort.automatic.rawValue,
        openAITextVerbosity: String = OpenAITextVerbosity.automatic.rawValue,
        openAIMaxOutputTokens: Int? = nil,
        doubaoDictionaryMode: String = DoubaoDictionaryMode.requestScoped.rawValue,
        doubaoEnableRequestHotwords: Bool = true,
        doubaoEnableRequestCorrections: Bool = true,
        ollamaResponseFormat: String = OllamaResponseFormat.plain.rawValue,
        ollamaJSONSchema: String = "",
        ollamaThinkMode: String = OllamaThinkMode.off.rawValue,
        ollamaKeepAlive: String = "",
        ollamaLogprobsEnabled: Bool = false,
        ollamaTopLogprobs: Int? = nil,
        ollamaOptionsJSON: String = "",
        omlxResponseFormat: String = OMLXResponseFormat.plain.rawValue,
        omlxJSONSchema: String = "",
        omlxIncludeUsageStreamOptions: Bool = false,
        omlxExtraBodyJSON: String = "",
        codexAuthFilePath: String = "",
        codexAuthFileBookmark: Data? = nil,
        codexFastModeEnabled: Bool = false,
        aliyunASRSettings: AliyunASRModelSettings = AliyunASRModelSettings(),
        generationSettings: LLMGenerationSettings? = nil
    ) {
        self.providerID = providerID
        self.model = model
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.appID = appID
        self.accessToken = accessToken
        storedCredentialPresence = RemoteStoredCredentialPresence(
            apiKey: apiKey,
            appID: appID,
            accessToken: accessToken
        )
        self.searchEnabled = searchEnabled
        self.openAIChunkPseudoRealtimeEnabled = openAIChunkPseudoRealtimeEnabled
        self.openAIReasoningEffort = openAIReasoningEffort
        self.openAITextVerbosity = openAITextVerbosity
        self.openAIMaxOutputTokens = openAIMaxOutputTokens
        self.doubaoDictionaryMode = doubaoDictionaryMode
        self.doubaoEnableRequestHotwords = doubaoEnableRequestHotwords
        self.doubaoEnableRequestCorrections = doubaoEnableRequestCorrections
        self.ollamaResponseFormat = ollamaResponseFormat
        self.ollamaJSONSchema = ollamaJSONSchema
        self.ollamaThinkMode = ollamaThinkMode
        self.ollamaKeepAlive = ollamaKeepAlive
        self.ollamaLogprobsEnabled = ollamaLogprobsEnabled
        self.ollamaTopLogprobs = ollamaTopLogprobs
        self.ollamaOptionsJSON = ollamaOptionsJSON
        self.omlxResponseFormat = omlxResponseFormat
        self.omlxJSONSchema = omlxJSONSchema
        self.omlxIncludeUsageStreamOptions = omlxIncludeUsageStreamOptions
        self.omlxExtraBodyJSON = omlxExtraBodyJSON
        self.codexAuthFilePath = codexAuthFilePath
        self.codexAuthFileBookmark = codexAuthFileBookmark
        self.codexFastModeEnabled = codexFastModeEnabled
        self.aliyunASRSettings = aliyunASRSettings
        self.generationSettings = generationSettings ?? LLMGenerationSettings.legacy(
            providerID: providerID,
            openAIReasoningEffort: openAIReasoningEffort,
            openAIMaxOutputTokens: openAIMaxOutputTokens,
            ollamaResponseFormat: ollamaResponseFormat,
            ollamaThinkMode: ollamaThinkMode,
            ollamaLogprobsEnabled: ollamaLogprobsEnabled,
            ollamaTopLogprobs: ollamaTopLogprobs,
            ollamaOptionsJSON: ollamaOptionsJSON,
            omlxResponseFormat: omlxResponseFormat,
            omlxExtraBodyJSON: omlxExtraBodyJSON
        )
    }

    enum CodingKeys: String, CodingKey {
        case providerID
        case model
        case endpoint
        case apiKey
        case appID
        case accessToken
        case storedCredentialPresence
        case searchEnabled
        case openAIChunkPseudoRealtimeEnabled
        case openAIReasoningEffort
        case openAITextVerbosity
        case openAIMaxOutputTokens
        case doubaoDictionaryMode
        case doubaoEnableRequestHotwords
        case doubaoEnableRequestCorrections
        case ollamaResponseFormat
        case ollamaJSONSchema
        case ollamaThinkMode
        case ollamaKeepAlive
        case ollamaLogprobsEnabled
        case ollamaTopLogprobs
        case ollamaOptionsJSON
        case omlxResponseFormat
        case omlxJSONSchema
        case omlxIncludeUsageStreamOptions
        case omlxExtraBodyJSON
        case codexAuthFilePath
        case codexAuthFileBookmark
        case codexFastModeEnabled
        case aliyunASRSettings
        case generationSettings
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        providerID = try container.decode(String.self, forKey: .providerID)
        model = try container.decodeIfPresent(String.self, forKey: .model) ?? ""
        endpoint = try container.decodeIfPresent(String.self, forKey: .endpoint) ?? ""
        apiKey = try container.decodeIfPresent(String.self, forKey: .apiKey) ?? ""
        appID = try container.decodeIfPresent(String.self, forKey: .appID) ?? ""
        accessToken = try container.decodeIfPresent(String.self, forKey: .accessToken) ?? ""
        storedCredentialPresence = try container.decodeIfPresent(
            RemoteStoredCredentialPresence.self,
            forKey: .storedCredentialPresence
        )
        let defaultSearchEnabled = RemoteLLMProvider(rawValue: providerID)?.defaultSearchEnabled ?? false
        searchEnabled = try container.decodeIfPresent(Bool.self, forKey: .searchEnabled) ?? defaultSearchEnabled
        openAIChunkPseudoRealtimeEnabled = try container.decodeIfPresent(Bool.self, forKey: .openAIChunkPseudoRealtimeEnabled) ?? false
        openAIReasoningEffort = try container.decodeIfPresent(String.self, forKey: .openAIReasoningEffort) ?? OpenAIReasoningEffort.automatic.rawValue
        openAITextVerbosity = try container.decodeIfPresent(String.self, forKey: .openAITextVerbosity) ?? OpenAITextVerbosity.automatic.rawValue
        openAIMaxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .openAIMaxOutputTokens)
        doubaoDictionaryMode = try container.decodeIfPresent(String.self, forKey: .doubaoDictionaryMode) ?? DoubaoDictionaryMode.requestScoped.rawValue
        doubaoEnableRequestHotwords = try container.decodeIfPresent(Bool.self, forKey: .doubaoEnableRequestHotwords) ?? true
        doubaoEnableRequestCorrections = try container.decodeIfPresent(Bool.self, forKey: .doubaoEnableRequestCorrections) ?? true
        ollamaResponseFormat = try container.decodeIfPresent(String.self, forKey: .ollamaResponseFormat) ?? OllamaResponseFormat.plain.rawValue
        ollamaJSONSchema = try container.decodeIfPresent(String.self, forKey: .ollamaJSONSchema) ?? ""
        ollamaThinkMode = try container.decodeIfPresent(String.self, forKey: .ollamaThinkMode) ?? OllamaThinkMode.off.rawValue
        ollamaKeepAlive = try container.decodeIfPresent(String.self, forKey: .ollamaKeepAlive) ?? ""
        ollamaLogprobsEnabled = try container.decodeIfPresent(Bool.self, forKey: .ollamaLogprobsEnabled) ?? false
        ollamaTopLogprobs = try container.decodeIfPresent(Int.self, forKey: .ollamaTopLogprobs)
        ollamaOptionsJSON = try container.decodeIfPresent(String.self, forKey: .ollamaOptionsJSON) ?? ""
        omlxResponseFormat = try container.decodeIfPresent(String.self, forKey: .omlxResponseFormat) ?? OMLXResponseFormat.plain.rawValue
        omlxJSONSchema = try container.decodeIfPresent(String.self, forKey: .omlxJSONSchema) ?? ""
        omlxIncludeUsageStreamOptions = try container.decodeIfPresent(Bool.self, forKey: .omlxIncludeUsageStreamOptions) ?? false
        omlxExtraBodyJSON = try container.decodeIfPresent(String.self, forKey: .omlxExtraBodyJSON) ?? ""
        codexAuthFilePath = try container.decodeIfPresent(String.self, forKey: .codexAuthFilePath) ?? ""
        codexAuthFileBookmark = try container.decodeIfPresent(Data.self, forKey: .codexAuthFileBookmark)
        codexFastModeEnabled = try container.decodeIfPresent(Bool.self, forKey: .codexFastModeEnabled) ?? false
        aliyunASRSettings = try container.decodeIfPresent(AliyunASRModelSettings.self, forKey: .aliyunASRSettings) ?? AliyunASRModelSettings()
        generationSettings = try container.decodeIfPresent(LLMGenerationSettings.self, forKey: .generationSettings)
            ?? LLMGenerationSettings.legacy(
                providerID: providerID,
                openAIReasoningEffort: openAIReasoningEffort,
                openAIMaxOutputTokens: openAIMaxOutputTokens,
                ollamaResponseFormat: ollamaResponseFormat,
                ollamaThinkMode: ollamaThinkMode,
                ollamaLogprobsEnabled: ollamaLogprobsEnabled,
                ollamaTopLogprobs: ollamaTopLogprobs,
                ollamaOptionsJSON: ollamaOptionsJSON,
                omlxResponseFormat: omlxResponseFormat,
                omlxExtraBodyJSON: omlxExtraBodyJSON
            )
    }

    nonisolated var withoutSensitiveValues: RemoteProviderConfiguration {
        var sanitized = self
        let detectedPresence = RemoteStoredCredentialPresence(
            apiKey: apiKey,
            appID: appID,
            accessToken: accessToken
        )
        sanitized.storedCredentialPresence = detectedPresence.isEmpty
            ? storedCredentialPresence
            : detectedPresence
        sanitized.apiKey = ""
        sanitized.appID = ""
        sanitized.accessToken = ""
        return sanitized
    }
}

/// A provider configuration whose credentials have already been resolved for
/// a network request. Request builders accept this type so metadata-only
/// configurations cannot reach Authorization header construction directly.
struct RemoteProviderRuntimeConfiguration {
    let value: RemoteProviderConfiguration

    fileprivate init(value: RemoteProviderConfiguration) {
        self.value = value
    }
}

enum RemoteModelConfigurationStore {
    private static let currentCredentialMigrationVersion = 1

    enum SaveError: LocalizedError, Equatable {
        case secureStorageUnavailable
        case metadataEncodingFailed

        var errorDescription: String? {
            switch self {
            case .secureStorageUnavailable:
                return AppLocalization.localizedString(
                    "Configuration could not be saved securely. Check Keychain access and try again."
                )
            case .metadataEncodingFailed:
                return AppLocalization.localizedString(
                    "Configuration could not be encoded. Please try again."
                )
            }
        }
    }

    enum SensitiveValueLoading {
        case metadataOnly
        case includeStoredValues
    }

    enum RuntimeCredentialError: LocalizedError, Equatable {
        case missing
        case corrupted
        case secureStorageUnavailable

        var errorDescription: String? {
            switch self {
            case .missing:
                return AppLocalization.localizedString(
                    "Stored provider credentials are missing. Reopen the provider settings and save them again."
                )
            case .corrupted:
                return AppLocalization.localizedString(
                    "Stored provider credentials are invalid. Reopen the provider settings and save them again."
                )
            case .secureStorageUnavailable:
                return AppLocalization.localizedString(
                    "Stored provider credentials are temporarily unavailable. Try again after unlocking your Mac."
                )
            }
        }
    }

    private enum SensitiveField: String, CaseIterable {
        case apiKey
        case appID
        case accessToken
    }

    private nonisolated struct StoredSensitiveValues: Codable {
        var apiKey: String
        var appID: String
        var accessToken: String

        init(configuration: RemoteProviderConfiguration) {
            apiKey = configuration.apiKey
            appID = configuration.appID
            accessToken = configuration.accessToken
        }

        var isEmpty: Bool {
            [apiKey, appID, accessToken].allSatisfy {
                $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
        }

        func applying(to configuration: RemoteProviderConfiguration) -> RemoteProviderConfiguration {
            var resolved = configuration
            resolved.apiKey = apiKey
            resolved.appID = appID
            resolved.accessToken = accessToken
            resolved.storedCredentialPresence = RemoteStoredCredentialPresence(
                apiKey: apiKey,
                appID: appID,
                accessToken: accessToken
            )
            return resolved
        }
    }

    private nonisolated enum CredentialMigrationResolution {
        case values(StoredSensitiveValues)
        case missing
        case unavailable
    }

    static func loadConfigurations(
        from raw: String,
        sensitiveValueLoading: SensitiveValueLoading = .includeStoredValues
    ) -> [String: RemoteProviderConfiguration] {
        let items = decodedConfigurations(from: raw).map(normalizedCompatibilityValues(for:))
        return Dictionary(uniqueKeysWithValues: items.map { item in
            let resolved = switch sensitiveValueLoading {
            case .metadataOnly:
                resolvedSensitiveValuePresence(for: item)
            case .includeStoredValues:
                resolvedSensitiveValues(for: item)
            }
            return (resolved.providerID, resolved)
        })
    }

    static func loadConfiguration(
        providerID: String,
        from raw: String,
        sensitiveValueLoading: SensitiveValueLoading = .includeStoredValues
    ) -> RemoteProviderConfiguration? {
        guard let item = decodedConfigurations(from: raw)
            .map(normalizedCompatibilityValues(for:))
            .first(where: { $0.providerID == providerID })
        else {
            return nil
        }

        switch sensitiveValueLoading {
        case .metadataOnly:
            return resolvedSensitiveValuePresence(for: item)
        case .includeStoredValues:
            return resolvedSensitiveValues(for: item)
        }
    }

    /// Resolves a metadata-only configuration into a request-safe value.
    /// Metadata never contains a credential-shaped placeholder.
    private static func configurationForRuntimeRequest(
        _ configuration: RemoteProviderConfiguration
    ) throws -> RemoteProviderConfiguration {
        try configurationByResolvingStoredCredentials(configuration)
    }

    /// Resolves only credential fields whose metadata says they are stored and
    /// whose current value is empty. Inline replacements and explicit clears
    /// therefore remain authoritative on a field-by-field basis.
    private static func configurationByResolvingStoredCredentials(
        _ configuration: RemoteProviderConfiguration
    ) throws -> RemoteProviderConfiguration {
        guard let expectedPresence = configuration.storedCredentialPresence else {
            return configuration
        }
        let inlinePresence = RemoteStoredCredentialPresence(
            apiKey: configuration.apiKey,
            appID: configuration.appID,
            accessToken: configuration.accessToken
        )
        let fieldsToResolve = SensitiveField.allCases.filter { field in
            switch field {
            case .apiKey:
                return expectedPresence.apiKey && !inlinePresence.apiKey
            case .appID:
                return expectedPresence.appID && !inlinePresence.appID
            case .accessToken:
                return expectedPresence.accessToken && !inlinePresence.accessToken
            }
        }
        guard !fieldsToResolve.isEmpty else {
            return configuration
        }

        let bundledAccount = bundledKeychainAccount(providerID: configuration.providerID)
        let stored: String
        do {
            guard let value = try VoxtSecureStorage.protectedString(for: bundledAccount) else {
                throw RuntimeCredentialError.missing
            }
            stored = value
        } catch let error as RuntimeCredentialError {
            throw error
        } catch {
            throw RuntimeCredentialError.secureStorageUnavailable
        }

        guard let data = stored.data(using: .utf8),
              let values = try? JSONDecoder().decode(StoredSensitiveValues.self, from: data)
        else {
            throw RuntimeCredentialError.corrupted
        }

        var resolved = configuration
        for field in fieldsToResolve {
            let value: String
            switch field {
            case .apiKey:
                value = values.apiKey
            case .appID:
                value = values.appID
            case .accessToken:
                value = values.accessToken
            }
            guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw RuntimeCredentialError.missing
            }
            setSensitiveValue(value, for: field, in: &resolved)
        }
        return resolved
    }

    static func runtimeConfiguration(
        for configuration: RemoteProviderConfiguration
    ) throws -> RemoteProviderRuntimeConfiguration {
        RemoteProviderRuntimeConfiguration(
            value: try configurationForRuntimeRequest(configuration)
        )
    }

    static func saveConfigurations(_ values: [String: RemoteProviderConfiguration]) -> String {
        let items = values.values.sorted(by: { $0.providerID < $1.providerID })
        for item in items {
            persistSensitiveValues(for: item)
        }
        return encodeConfigurations(items.map(\.withoutSensitiveValues))
    }

    static func saveConfiguration(
        _ configuration: RemoteProviderConfiguration,
        updating raw: String
    ) -> Result<String, SaveError> {
        let resolvedConfiguration: RemoteProviderConfiguration
        do {
            resolvedConfiguration = try configurationByResolvingStoredCredentials(configuration)
        } catch {
            return .failure(.secureStorageUnavailable)
        }

        var items = decodedConfigurations(from: raw).map(normalizedCompatibilityValues(for:))
        let sanitized = resolvedConfiguration.withoutSensitiveValues

        if let existingIndex = items.firstIndex(where: { $0.providerID == configuration.providerID }) {
            items[existingIndex] = sanitized
        } else {
            items.append(sanitized)
        }

        guard let encoded = encodedConfigurations(items.sorted(by: { $0.providerID < $1.providerID })) else {
            return .failure(.metadataEncodingFailed)
        }
        guard persistSensitiveValues(for: resolvedConfiguration) else {
            return .failure(.secureStorageUnavailable)
        }
        return .success(encoded)
    }

    static func migrateLegacyStoredSecrets(defaults: UserDefaults = .standard) {
        guard defaults.integer(forKey: AppPreferenceKey.remoteCredentialMigrationVersion)
            < currentCredentialMigrationVersion
        else {
            return
        }

        let asrCompleted = migrateLegacyStoredSecrets(
            defaultsKey: AppPreferenceKey.remoteASRProviderConfigurations,
            defaults: defaults
        )
        let llmCompleted = migrateLegacyStoredSecrets(
            defaultsKey: AppPreferenceKey.remoteLLMProviderConfigurations,
            defaults: defaults
        )
        guard asrCompleted, llmCompleted else { return }
        defaults.set(
            currentCredentialMigrationVersion,
            forKey: AppPreferenceKey.remoteCredentialMigrationVersion
        )
    }

    // Temporary compatibility migration for persisted legacy LLM endpoints.
    // Remove this after the legacy upgrade window closes and all supported users
    // have moved through a version that rewrites old `/models` and
    // `/chat/completions` URLs to `/responses`.
    static func migrateLegacyLLMEndpoints(defaults: UserDefaults = .standard) {
        let defaultsKey = AppPreferenceKey.remoteLLMProviderConfigurations
        let raw = defaults.string(forKey: defaultsKey) ?? ""
        guard !raw.isEmpty else { return }

        let decoded = decodedConfigurations(from: raw)
        let migrated = encodeConfigurations(decoded.map(normalizedCompatibilityValues(for:)))
        if migrated != raw {
            defaults.set(migrated, forKey: defaultsKey)
        }
    }

    @discardableResult
    private static func migrateLegacyStoredSecrets(defaultsKey: String, defaults: UserDefaults) -> Bool {
        let raw = defaults.string(forKey: defaultsKey) ?? ""
        guard !raw.isEmpty else { return true }
        let decoded = decodedConfigurations(from: raw)
        let normalized = decoded.map(normalizedCompatibilityValues(for:))
        var changed = false
        var completed = true
        let migrated = normalized.map { configuration in
            if hasInlineSensitiveValues(configuration) {
                guard persistSensitiveValues(for: configuration) else {
                    completed = false
                    return configuration
                }
                changed = true
                return configuration.withoutSensitiveValues
            }

            // An explicit empty presence mask means the user never configured a
            // credential (or intentionally cleared it). Do not touch Keychain.
            if configuration.storedCredentialPresence?.isEmpty == true {
                return configuration
            }

            switch credentialMigrationResolution(for: configuration) {
            case .values(let values):
                // An empty bundled tombstone is not a configured credential. It
                // only needs to replace a stale positive presence mask.
                guard !values.isEmpty || configuration.storedCredentialPresence != nil else {
                    return configuration
                }
                let resolved = values.applying(to: configuration).withoutSensitiveValues
                if resolved != configuration {
                    changed = true
                }
                return resolved
            case .missing:
                // No metadata and no stored value means this provider was never
                // configured, so there is nothing to migrate.
                guard configuration.storedCredentialPresence != nil else {
                    return configuration
                }
                // A previous build may have left a positive mask after losing
                // the Keychain item. Persist one definitive empty mask so the UI
                // asks for the credential once and future launches do no work.
                var configurationWithoutPresence = configuration
                configurationWithoutPresence.storedCredentialPresence = RemoteStoredCredentialPresence(
                    apiKey: "",
                    appID: "",
                    accessToken: ""
                )
                let resolved = configurationWithoutPresence.withoutSensitiveValues
                if resolved != configuration {
                    changed = true
                }
                return resolved
            case .unavailable:
                // Never turn an inconclusive Keychain read into data loss. Leave
                // the migration pending and retry on a later launch.
                completed = false
                return configuration
            }
        }

        if changed {
            let encoded = encodeConfigurations(migrated)
            if encoded != raw {
                defaults.set(encoded, forKey: defaultsKey)
            }
        }
        return completed
    }

    nonisolated private static func decodedConfigurations(from raw: String) -> [RemoteProviderConfiguration] {
        guard let data = raw.data(using: .utf8), !data.isEmpty else {
            return []
        }
        do {
            return try JSONDecoder().decode([RemoteProviderConfiguration].self, from: data)
        } catch {
            return []
        }
    }

    nonisolated private static func encodeConfigurations(_ items: [RemoteProviderConfiguration]) -> String {
        encodedConfigurations(items) ?? ""
    }

    nonisolated private static func encodedConfigurations(
        _ items: [RemoteProviderConfiguration]
    ) -> String? {
        guard let data = try? JSONEncoder().encode(items) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    nonisolated private static func resolvedSensitiveValues(for configuration: RemoteProviderConfiguration) -> RemoteProviderConfiguration {
        let bundledAccount = bundledKeychainAccount(providerID: configuration.providerID)
        if let stored = VoxtSecureStorage.string(for: bundledAccount),
           let data = stored.data(using: .utf8),
           let values = try? JSONDecoder().decode(StoredSensitiveValues.self, from: data) {
            return values.applying(to: configuration)
        }

        var resolved = configuration
        var foundLegacyValue = false
        for field in SensitiveField.allCases {
            let keychainValue = VoxtSecureStorage.string(
                for: legacyKeychainAccount(providerID: configuration.providerID, field: field)
            )
            let currentValue = sensitiveValue(for: field, in: configuration)
            let finalValue = keychainValue ?? currentValue
            foundLegacyValue = foundLegacyValue || keychainValue != nil
            setSensitiveValue(finalValue, for: field, in: &resolved)
        }

        if foundLegacyValue || hasInlineSensitiveValues(resolved) {
            _ = persistSensitiveValues(for: resolved)
        }
        return resolved
    }

    nonisolated private static func credentialMigrationResolution(
        for configuration: RemoteProviderConfiguration
    ) -> CredentialMigrationResolution {
        let bundledAccount = bundledKeychainAccount(providerID: configuration.providerID)
        switch VoxtSecureStorage.migrationValue(for: bundledAccount) {
        case .value(let stored):
            if let data = stored.data(using: .utf8),
               let values = try? JSONDecoder().decode(StoredSensitiveValues.self, from: data) {
                return .values(values)
            }
            // A corrupt bundled item is unusable, but an older per-field copy
            // may still be recoverable below.
        case .missing:
            break
        case .unavailable:
            return .unavailable
        }

        var resolved = configuration
        for field in SensitiveField.allCases {
            setSensitiveValue("", for: field, in: &resolved)
        }
        var foundLegacyValue = false
        for field in SensitiveField.allCases {
            let account = legacyKeychainAccount(providerID: configuration.providerID, field: field)
            switch VoxtSecureStorage.migrationValue(for: account) {
            case .value(let value):
                guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continue
                }
                foundLegacyValue = true
                setSensitiveValue(value, for: field, in: &resolved)
            case .missing:
                continue
            case .unavailable:
                return .unavailable
            }
        }

        guard foundLegacyValue else {
            return .missing
        }
        guard persistSensitiveValues(for: resolved) else {
            return .unavailable
        }
        return .values(StoredSensitiveValues(configuration: resolved))
    }

    nonisolated private static func resolvedSensitiveValuePresence(for configuration: RemoteProviderConfiguration) -> RemoteProviderConfiguration {
        var resolved = configuration.withoutSensitiveValues
        if let presence = configuration.storedCredentialPresence {
            applySensitiveValuePresence(presence, to: &resolved)
            return resolved
        }

        let bundledAccount = bundledKeychainAccount(providerID: configuration.providerID)
        if let stored = VoxtSecureStorage.string(for: bundledAccount),
           let data = stored.data(using: .utf8),
           let values = try? JSONDecoder().decode(StoredSensitiveValues.self, from: data) {
            applySensitiveValuePresence(
                RemoteStoredCredentialPresence(
                    apiKey: values.apiKey,
                    appID: values.appID,
                    accessToken: values.accessToken
                ),
                to: &resolved
            )
            return resolved
        }

        for field in SensitiveField.allCases {
            let hasKeychainValue = VoxtSecureStorage.hasString(
                for: legacyKeychainAccount(providerID: configuration.providerID, field: field)
            )
            let currentValue = sensitiveValue(for: field, in: configuration)
            let hasValue = hasKeychainValue || !currentValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            guard hasValue else { continue }
            var presence = resolved.storedCredentialPresence ?? RemoteStoredCredentialPresence(
                apiKey: "",
                appID: "",
                accessToken: ""
            )
            switch field {
            case .apiKey:
                presence.apiKey = true
            case .appID:
                presence.appID = true
            case .accessToken:
                presence.accessToken = true
            }
            resolved.storedCredentialPresence = presence
        }
        return resolved
    }

    nonisolated private static func applySensitiveValuePresence(
        _ presence: RemoteStoredCredentialPresence,
        to configuration: inout RemoteProviderConfiguration
    ) {
        configuration.apiKey = ""
        configuration.appID = ""
        configuration.accessToken = ""
        configuration.storedCredentialPresence = presence
    }

    @discardableResult
    nonisolated private static func persistSensitiveValues(
        for configuration: RemoteProviderConfiguration
    ) -> Bool {
        let values = StoredSensitiveValues(configuration: configuration)
        let bundledAccount = bundledKeychainAccount(providerID: configuration.providerID)
        guard let data = try? JSONEncoder().encode(values),
              let encoded = String(data: data, encoding: .utf8)
        else {
            return false
        }
        do {
            try VoxtSecureStorage.setProtectedString(encoded, for: bundledAccount)
        } catch {
            return false
        }

        // The bundled item is authoritative as soon as it is written. In
        // particular, an encoded empty value acts as a tombstone so a failed
        // cleanup cannot make an old per-field credential visible again.
        var removedAllLegacyItems = true
        for field in SensitiveField.allCases {
            if !VoxtSecureStorage.removeValueWithoutUserInteraction(
                for: legacyKeychainAccount(providerID: configuration.providerID, field: field)
            ) {
                removedAllLegacyItems = false
            }
        }

        // Once every legacy account is gone, an empty tombstone is no longer
        // needed. Failure to remove it is harmless: it still resolves to the
        // same empty credential set and can be retried on a later save.
        if values.isEmpty, removedAllLegacyItems {
            _ = VoxtSecureStorage.removeValueWithoutUserInteraction(for: bundledAccount)
        }
        return true
    }

    nonisolated private static func hasInlineSensitiveValues(_ configuration: RemoteProviderConfiguration) -> Bool {
        SensitiveField.allCases.contains { field in
            !sensitiveValue(for: field, in: configuration)
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty
        }
    }

    nonisolated private static func bundledKeychainAccount(providerID: String) -> String {
        "remote-provider.\(providerID).credentials"
    }

    nonisolated private static func legacyKeychainAccount(providerID: String, field: SensitiveField) -> String {
        "remote-provider.\(providerID).\(field.rawValue)"
    }

    nonisolated private static func sensitiveValue(
        for field: SensitiveField,
        in configuration: RemoteProviderConfiguration
    ) -> String {
        switch field {
        case .apiKey:
            return configuration.apiKey
        case .appID:
            return configuration.appID
        case .accessToken:
            return configuration.accessToken
        }
    }

    nonisolated private static func setSensitiveValue(
        _ value: String,
        for field: SensitiveField,
        in configuration: inout RemoteProviderConfiguration
    ) {
        switch field {
        case .apiKey:
            configuration.apiKey = value
        case .appID:
            configuration.appID = value
        case .accessToken:
            configuration.accessToken = value
        }
    }
}
