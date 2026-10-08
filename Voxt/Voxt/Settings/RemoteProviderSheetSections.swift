// RemoteProviderSheetSections.swift
// Provides Remote Provider Sheet Sections for settings screens.

import SwiftUI

extension RemoteProviderConfigurationSheet {
    private var ollamaOptionsJSONPlaceholder: String {
        """
        {
          "num_ctx": 8192,
          "seed": 42,
          "repeat_penalty": 1.1,
          "stop": ["<|im_end|>"]
        }
        """
    }

    private var ollamaJSONSchemaPlaceholder: String {
        """
        {
          "type": "object",
          "properties": {
            "answer": { "type": "string" }
          },
          "required": ["answer"]
        }
        """
    }

    private var omlxJSONSchemaPlaceholder: String {
        """
        {
          "type": "object",
          "properties": {
            "answer": { "type": "string" }
          },
          "required": ["answer"]
        }
        """
    }

    private var omlxExtraBodyJSONPlaceholder: String {
        """
        {
          "top_k": 40,
          "min_p": 0.05
        }
        """
    }

    private var extraBodyJSONPlaceholder: String {
        """
        {
          "reasoning": { "effort": "low" },
          "response_format": { "type": "json_object" }
        }
        """
    }

    var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("Model"))
                .font(.subheadline)
                .foregroundStyle(.secondary)
            SettingsMenuPicker(
                selection: providerModelSelectionBinding,
                options: providerModelMenuOptions,
                selectedTitle: providerModelSelectedTitle,
                width: 240
            )

            if shouldShowCustomProviderModelField {
                Text(AppLocalization.localizedString("Custom Model ID (Optional)"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                TextField(customProviderModelPlaceholder, text: $customModelID)
                    .textFieldStyle(.plain)
                    .settingsFieldSurface(minHeight: 34)
            }
        }
    }

    var endpointAndKeySection: some View {
        Group {
            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Endpoint (Optional)"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                TextField(endpointFieldPlaceholder, text: $endpoint)
                    .textFieldStyle(.plain)
                    .settingsFieldSurface(minHeight: 34)
                if !endpointPresets.isEmpty {
                    HStack(spacing: 10) {
                        Menu {
                            ForEach(endpointPresets, id: \.id) { preset in
                                Button(preset.title) {
                                    endpoint = preset.url
                                }
                            }
                        } label: {
                            HStack(spacing: 5) {
                                Text(AppLocalization.localizedString("Apply Preset"))
                                Image(systemName: "chevron.down")
                                    .font(.system(size: 10, weight: .semibold))
                            }
                        }
                        .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 30))

                        if !endpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Button(AppLocalization.localizedString("Clear")) {
                                endpoint = ""
                            }
                            .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 30))
                        }

                        Spacer()
                    }
                }
            }

            if isCodexLLMProvider {
                VStack(alignment: .leading, spacing: 8) {
                    Text(AppLocalization.localizedString("Codex Credentials, Voxt uses local Codex OAuth credentials."))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(AppLocalization.localizedString("For first-time setup, manually choose the auth.json configuration file once."))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    Text(AppLocalization.localizedString("Auth File"))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    HStack(alignment: .center, spacing: 10) {
                        Text(codexAuthFileDisplayPath)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color(nsColor: .textBackgroundColor).opacity(0.75))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.secondary.opacity(0.12), lineWidth: 1)
                            )

                        Button(AppLocalization.localizedString("Choose")) {
                            chooseCodexAuthFile()
                        }
                        .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 30))

                        if hasCustomCodexAuthFilePath {
                            Button(AppLocalization.localizedString("Clear")) {
                                clearCodexAuthFileSelection()
                            }
                            .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 30))
                        }
                    }

                    if let codexAuthFileSelectionError, !codexAuthFileSelectionError.isEmpty {
                        Text(codexAuthFileSelectionError)
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Text(apiKeyFieldTitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 10) {
                        SecureField(apiKeyFieldPlaceholder, text: apiKeyInput)
                            .textFieldStyle(.plain)
                            .settingsFieldSurface(minHeight: 34)
                        if shouldOfferStoredCredentialClear(for: .apiKey) {
                            Button(AppLocalization.localizedString("Clear")) {
                                clearStoredCredential(.apiKey)
                            }
                            .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 30))
                        }
                    }
                }
            }

            if let endpointPresetHintText {
                Text(endpointPresetHintText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    var doubaoCredentialsSection: some View {
        Group {
            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("App ID"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    TextField(AppLocalization.localizedString("App ID"), text: appIDInput)
                        .textFieldStyle(.plain)
                        .settingsFieldSurface(minHeight: 34)
                    if shouldOfferStoredCredentialClear(for: .appID) {
                        Button(AppLocalization.localizedString("Clear")) {
                            clearStoredCredential(.appID)
                        }
                        .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 30))
                    }
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Access Token"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    SecureField(AppLocalization.localizedString("Paste access token"), text: accessTokenInput)
                        .textFieldStyle(.plain)
                        .settingsFieldSurface(minHeight: 34)
                    if shouldOfferStoredCredentialClear(for: .accessToken) {
                        Button(AppLocalization.localizedString("Clear")) {
                            clearStoredCredential(.accessToken)
                        }
                        .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 30))
                    }
                }
            }
        }
    }

    var doubaoDictionarySection: some View {
        RemoteProviderDoubaoDictionarySection(
            mode: $doubaoDictionaryMode,
            enableRequestHotwords: $doubaoEnableRequestHotwords,
            enableRequestCorrections: $doubaoEnableRequestCorrections
        )
    }

    @ViewBuilder
    var aliyunASRConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(AppLocalization.localizedString("Aliyun ASR Model Settings"))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if aliyunASRModelCapabilities.supportsInlineVocabulary {
                Text(AppLocalization.localizedString("High-priority terms from the current Voxt dictionary and contextual phrases will be sent as instant hotwords for this model."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if aliyunASRModelCapabilities.supportsLanguageHints {
                Text(AppLocalization.format("Language hints follow the languages selected in Speech Recognition Language settings. This model accepts up to %d hint(s).", aliyunASRModelCapabilities.maximumLanguageHints))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if aliyunASRModelCapabilities.family == .qwen3ASR {
                Text(AppLocalization.localizedString("This model uses the main language hint only and does not expose hotwords in the realtime API."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if aliyunASRModelCapabilities.supportsMaxSentenceSilence {
                aliyunNumericField(
                    title: AppLocalization.localizedString("Sentence Silence (ms)"),
                    placeholder: "1300",
                    text: $aliyunMaxSentenceSilenceMillisecondsText
                )
            }

            if aliyunASRModelCapabilities.supportsServerVAD {
                RemoteProviderConfigurationDisclosureSection(
                    title: AppLocalization.localizedString("Voice Activity Detection"),
                    isExpanded: $generationAdvancedExpanded,
                    content: {
                        VStack(alignment: .leading, spacing: 12) {
                            if aliyunASRModelCapabilities.supportsManualCommit {
                                Toggle(AppLocalization.localizedString("Manual Commit"), isOn: $aliyunUseManualCommit)
                                    .toggleStyle(.switch)
                            }

                            if !aliyunUseManualCommit || !aliyunASRModelCapabilities.supportsManualCommit {
                                HStack(alignment: .top, spacing: 12) {
                                    aliyunNumericField(
                                        title: AppLocalization.localizedString("Threshold"),
                                        placeholder: "0.35",
                                        text: $aliyunServerVADThresholdText
                                    )
                                    aliyunNumericField(
                                        title: AppLocalization.localizedString("Silence (ms)"),
                                        placeholder: "800",
                                        text: $aliyunServerVADSilenceDurationMillisecondsText
                                    )
                                }
                            }
                        }
                        .padding(.top, 8)
                    }
                )
            }

            if aliyunASRModelCapabilities.supportsSemanticPunctuation {
                Toggle(AppLocalization.localizedString("Semantic Punctuation"), isOn: $aliyunSemanticPunctuationEnabled)
                    .toggleStyle(.switch)
            }
            if aliyunASRModelCapabilities.supportsPunctuationPrediction {
                Toggle(AppLocalization.localizedString("Punctuation Prediction"), isOn: $aliyunPunctuationPredictionEnabled)
                    .toggleStyle(.switch)
            }
            if aliyunASRModelCapabilities.supportsInverseTextNormalization {
                Toggle(AppLocalization.localizedString("Inverse Text Normalization"), isOn: $aliyunInverseTextNormalizationEnabled)
                    .toggleStyle(.switch)
            }
            if aliyunASRModelCapabilities.supportsDisfluencyRemoval {
                Toggle(AppLocalization.localizedString("Remove Disfluencies"), isOn: $aliyunDisfluencyRemovalEnabled)
                    .toggleStyle(.switch)
            }
        }
    }

    private func aliyunNumericField(title: String, placeholder: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            TextField(placeholder, text: text)
                .textFieldStyle(.plain)
                .settingsFieldSurface(width: 160, minHeight: 34)
        }
    }

    @ViewBuilder
    var stepFunASRConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("StepFun ASR Model Settings"))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if stepFunASRModelCapabilities.supportsHotwords {
                Text(AppLocalization.localizedString("High-priority terms from the current Voxt dictionary and contextual phrases will be sent as StepFun hotwords for this model."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text(AppLocalization.localizedString("This StepFun model does not expose hotwords. Dictionary terms are not sent as a hotword list."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if stepFunASRModelCapabilities.supportsPrompt {
                Text(AppLocalization.localizedString("This model also accepts a recognition context prompt generated from the current dictionary and contextual phrases."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text(AppLocalization.localizedString("Language follows the languages selected in Speech Recognition Language settings."))
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    var openAIChunkSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(AppLocalization.localizedString("Chunk Pseudo Realtime Preview"), isOn: $openAIChunkPseudoRealtimeEnabled)
                .toggleStyle(.switch)
            Text(AppLocalization.localizedString("Enable segmented OpenAI ASR preview during recording. This roughly doubles usage."))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    var searchSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(AppLocalization.localizedString("Search"), isOn: $searchEnabled)
                .toggleStyle(.switch)

            if let provider = llmProviderForPicker {
                Text(provider.searchToggleDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    var codexConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(AppLocalization.localizedString("Fast Mode"), isOn: $codexFastModeEnabled)
                .toggleStyle(.switch)
            Text(AppLocalization.localizedString("Requests Codex with priority service tier when the backend supports Fast mode."))
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    var advancedGenerationSettingsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(AppLocalization.localizedString("Generation Settings"))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if shouldShowGenerationThinking {
                VStack(alignment: .leading, spacing: 8) {
                    Text(AppLocalization.localizedString("Think"))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    SettingsMenuPicker(
                        selection: $generationThinkingMode,
                        options: generationThinkingModeMenuOptions,
                        selectedTitle: generationThinkingModeSelectedTitle,
                        width: 240
                    )

                    if LLMThinkingMode(rawValue: generationThinkingMode) == .effort {
                        if generationThinkingEffortMenuOptions.isEmpty {
                            TextField(AppLocalization.localizedString("e.g. low"), text: $generationThinkingEffort)
                                .textFieldStyle(.plain)
                                .settingsFieldSurface(width: 160, minHeight: 34)
                        } else {
                            SettingsMenuPicker(
                                selection: $generationThinkingEffort,
                                options: generationThinkingEffortMenuOptions,
                                selectedTitle: generationThinkingEffortSelectedTitle,
                                width: 180
                            )
                        }
                    }

                    if LLMThinkingMode(rawValue: generationThinkingMode) == .budget {
                        TextField(AppLocalization.localizedString("Budget tokens"), text: $generationThinkingBudgetText)
                            .textFieldStyle(.plain)
                            .settingsFieldSurface(width: 160, minHeight: 34)
                    }
                }
            }

            if generationCapabilities?.supportsMaxOutputTokens == true {
                generationNumericField(
                    title: AppLocalization.localizedString("Max Output Tokens (Optional)"),
                    placeholder: AppLocalization.localizedString("e.g. 4096"),
                    text: $generationMaxOutputTokensText
                )
            }

            if generationCapabilities?.supportsTemperature == true ||
                generationCapabilities?.supportsTopP == true {
                HStack(alignment: .top, spacing: 12) {
                    if generationCapabilities?.supportsTemperature == true {
                        generationNumericField(
                            title: AppLocalization.localizedString("Temperature"),
                            placeholder: "0.2",
                            text: $generationTemperatureText
                        )
                    }
                    if generationCapabilities?.supportsTopP == true {
                        generationNumericField(
                            title: AppLocalization.localizedString("Top P"),
                            placeholder: "0.9",
                            text: $generationTopPText
                        )
                    }
                }
            }

            if shouldShowGenerationAdvancedControls {
                RemoteProviderConfigurationDisclosureSection(
                    title: AppLocalization.localizedString("Advanced"),
                    isExpanded: $generationAdvancedExpanded,
                    content: {
                        VStack(alignment: .leading, spacing: 12) {
                            if generationCapabilities?.supportsResponseFormat == true {
                                generationResponseFormatSection
                            }

                            HStack(alignment: .top, spacing: 12) {
                                if generationCapabilities?.supportsTopK == true {
                                    generationNumericField(
                                        title: AppLocalization.localizedString("Top K"),
                                        placeholder: "40",
                                        text: $generationTopKText
                                    )
                                }
                                if generationCapabilities?.supportsMinP == true {
                                    generationNumericField(
                                        title: AppLocalization.localizedString("Min P"),
                                        placeholder: "0.05",
                                        text: $generationMinPText
                                    )
                                }
                            }

                            if generationCapabilities?.supportsSeed == true {
                                generationNumericField(
                                    title: AppLocalization.localizedString("Seed"),
                                    placeholder: "42",
                                    text: $generationSeedText
                                )
                            }

                            if generationCapabilities?.supportsPenalties == true {
                                generationPenaltyFields
                            }

                            if generationCapabilities?.supportsLogprobs == true {
                                generationLogprobsSection
                            }

                            if generationCapabilities?.supportsStopSequences == true {
                                generationStopSection
                            }
                        }
                        .padding(.top, 8)
                    }
                )
            }

            if shouldShowGenerationExpertControls {
                RemoteProviderConfigurationDisclosureSection(
                    title: AppLocalization.localizedString("Expert"),
                    isExpanded: $generationExpertExpanded,
                    content: {
                        VStack(alignment: .leading, spacing: 12) {
                            if generationCapabilities?.supportsExtraBody == true {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text(AppLocalization.localizedString("Extra Body JSON"))
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                    placeholderPromptEditor(
                                        text: $generationExtraBodyJSON,
                                        placeholder: extraBodyJSONPlaceholder,
                                        height: 112
                                    )
                                    Text(AppLocalization.localizedString("Merged into the request body after Voxt defaults, so matching keys override generated values."))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }

                            if generationCapabilities?.supportsExtraOptions == true {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text(AppLocalization.localizedString("Options JSON"))
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                    placeholderPromptEditor(
                                        text: $generationExtraOptionsJSON,
                                        placeholder: ollamaOptionsJSONPlaceholder,
                                        height: 112
                                    )
                                    Text(AppLocalization.localizedString("Merged into provider options after Voxt defaults, so matching keys override generated values."))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                        .padding(.top, 8)
                    }
                )
            }
        }
    }

    var openAILLMConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(AppLocalization.localizedString("OpenAI Options"))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Verbosity"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                SettingsMenuPicker(
                    selection: $openAITextVerbosity,
                    options: openAITextVerbosityMenuOptions,
                    selectedTitle: openAITextVerbositySelectedTitle,
                    width: 240
                )
            }
        }
    }

    var ollamaConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(AppLocalization.localizedString("Ollama Options"))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                Text(AppLocalization.localizedString("Keep Alive"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                TextField(AppLocalization.localizedString("e.g. 5m"), text: $ollamaKeepAlive)
                    .textFieldStyle(.plain)
                    .settingsFieldSurface(minHeight: 34)
            }
        }
    }

    var omlxConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(AppLocalization.localizedString("oMLX Options"))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                Toggle(AppLocalization.localizedString("Include Usage In Stream Events"), isOn: $omlxIncludeUsageStreamOptions)
                    .toggleStyle(.switch)
                Text(AppLocalization.localizedString("Adds stream_options.include_usage for OpenAI-compatible streaming responses."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    var generationResponseFormatSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("Response Format"))
                .font(.subheadline)
                .foregroundStyle(.secondary)
            SettingsMenuPicker(
                selection: $generationResponseFormat,
                options: generationResponseFormatMenuOptions,
                selectedTitle: generationResponseFormatSelectedTitle,
                width: 240
            )

            if shouldShowGenerationJSONSchemaField {
                Text(AppLocalization.localizedString("JSON Schema"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                placeholderPromptEditor(
                    text: isOllamaLLMProvider ? $ollamaJSONSchema : $omlxJSONSchema,
                    placeholder: isOllamaLLMProvider ? ollamaJSONSchemaPlaceholder : omlxJSONSchemaPlaceholder,
                    height: 96
                )
            }
        }
    }

    var generationPenaltyFields: some View {
        VStack(alignment: .leading, spacing: 10) {
            if isStepFunLLMProvider {
                generationNumericField(
                    title: AppLocalization.localizedString("Frequency Penalty"),
                    placeholder: "0",
                    text: $generationFrequencyPenaltyText
                )
            } else {
                HStack(alignment: .top, spacing: 12) {
                    generationNumericField(
                        title: AppLocalization.localizedString("Presence Penalty"),
                        placeholder: "0",
                        text: $generationPresencePenaltyText
                    )
                    generationNumericField(
                        title: AppLocalization.localizedString("Frequency Penalty"),
                        placeholder: "0",
                        text: $generationFrequencyPenaltyText
                    )
                }
                generationNumericField(
                    title: AppLocalization.localizedString("Repetition Penalty"),
                    placeholder: "1.1",
                    text: $generationRepetitionPenaltyText
                )
            }
        }
    }

    var generationLogprobsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(AppLocalization.localizedString("Logprobs"), isOn: $generationLogprobsEnabled)
                .toggleStyle(.switch)

            if generationLogprobsEnabled {
                generationNumericField(
                    title: AppLocalization.localizedString("Top Logprobs (Optional)"),
                    placeholder: "5",
                    text: $generationTopLogprobsText
                )
            }
        }
    }

    var generationStopSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("Stop Sequences"))
                .font(.subheadline)
                .foregroundStyle(.secondary)
            placeholderPromptEditor(
                text: $generationStopText,
                placeholder: AppLocalization.localizedString("One stop sequence per line"),
                height: 72
            )
        }
    }

    func generationNumericField(
        title: String,
        placeholder: String,
        text: Binding<String>
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            TextField(placeholder, text: text)
                .textFieldStyle(.plain)
                .settingsFieldSurface(width: 140, minHeight: 34)
        }
    }

    var actionSection: some View {
        SettingsDialogActionRow {
            if operations.isTestingConnection {
                ProgressView()
                    .controlSize(.small)
            }
            Button(AppLocalization.localizedString("Test")) {
                testConnection()
            }
            .buttonStyle(SettingsPillButtonStyle())
            .disabled(operations.isTestingConnection)

        } trailing: {
            Button(AppLocalization.localizedString("Cancel")) {
                close()
            }
            .buttonStyle(SettingsPillButtonStyle())
            .keyboardShortcut(.cancelAction)

            Button(AppLocalization.localizedString("Save")) {
                saveConfiguration()
            }
            .buttonStyle(SettingsPrimaryButtonStyle())
            .keyboardShortcut(.defaultAction)
        }
    }

    @ViewBuilder
    private func placeholderPromptEditor(
        text: Binding<String>,
        placeholder: String,
        height: CGFloat
    ) -> some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: text)
                .settingsPromptEditor(height: height, contentPadding: 6)

            if text.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(placeholder)
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .lineSpacing(4)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .allowsHitTesting(false)
            }
        }
    }
}

private struct RemoteProviderConfigurationDisclosureSection<Content: View>: View {
    let title: String
    @Binding var isExpanded: Bool
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.14)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 12, height: 12)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))

                    Text(title)
                        .font(.subheadline)
                        .foregroundStyle(.primary)

                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isExpanded {
                content()
                    .transition(.opacity.combined(with: .move(edge: .top)))
                    .clipped()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
