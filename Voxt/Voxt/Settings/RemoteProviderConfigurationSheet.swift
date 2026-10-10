// RemoteProviderConfigurationSheet.swift
// Provides Remote Provider Configuration Sheet for settings screens.

import SwiftUI
import Foundation

struct RemoteProviderConfigurationSheet: View {
    @Environment(\.dismiss) var dismiss
    @AppStorage(AppPreferenceKey.remoteASRSelectedProvider) var selectedRemoteASRProviderRaw = RemoteASRProvider.openAIWhisper.rawValue
    @AppStorage(AppPreferenceKey.remoteLLMSelectedProvider) var selectedRemoteLLMProviderRaw = RemoteLLMProvider.openAI.rawValue

    let providerTitle: String
    let credentialHint: String?
    let showsDoubaoFields: Bool
    let testTarget: RemoteProviderTestTarget
    let configuration: RemoteProviderConfiguration
    let onSave: (RemoteProviderConfiguration) -> Result<Void, RemoteModelConfigurationStore.SaveError>
    var cornerRadius: CGFloat = SettingsUIStyle.dialogCornerRadius
    var onClose: (() -> Void)? = nil

    @State var selectedProviderModel = ""
    @State var customModelID = ""
    @State var endpoint = ""
    @State var apiKey = ""
    @State var appID = ""
    @State var accessToken = ""
    @State var editedCredentialFields: Set<RemoteProviderConfiguration.CredentialField> = []
    @State var searchEnabled = false
    @State var openAIChunkPseudoRealtimeEnabled = false
    @State var openAITextVerbosity = OpenAITextVerbosity.automatic.rawValue
    @State var generationMaxOutputTokensText = ""
    @State var generationTemperatureText = ""
    @State var generationTopPText = ""
    @State var generationTopKText = ""
    @State var generationMinPText = ""
    @State var generationSeedText = ""
    @State var generationStopText = ""
    @State var generationPresencePenaltyText = ""
    @State var generationFrequencyPenaltyText = ""
    @State var generationRepetitionPenaltyText = ""
    @State var generationLogprobsEnabled = false
    @State var generationTopLogprobsText = ""
    @State var generationResponseFormat = LLMResponseFormat.plain.rawValue
    @State var generationThinkingMode = LLMThinkingMode.providerDefault.rawValue
    @State var generationThinkingEffort = ""
    @State var generationThinkingBudgetText = ""
    @State var generationExtraBodyJSON = ""
    @State var generationExtraOptionsJSON = ""
    @State var generationAdvancedExpanded = false
    @State var generationExpertExpanded = false
    @State var doubaoDictionaryMode = DoubaoDictionaryMode.requestScoped.rawValue
    @State var doubaoEnableRequestHotwords = true
    @State var doubaoEnableRequestCorrections = true
    @State var aliyunMaxSentenceSilenceMillisecondsText = "1300"
    @State var aliyunServerVADThresholdText = "0.35"
    @State var aliyunServerVADSilenceDurationMillisecondsText = "800"
    @State var aliyunUseManualCommit = false
    @State var aliyunSemanticPunctuationEnabled = true
    @State var aliyunPunctuationPredictionEnabled = true
    @State var aliyunInverseTextNormalizationEnabled = true
    @State var aliyunDisfluencyRemovalEnabled = false
    @State var ollamaResponseFormat = OllamaResponseFormat.plain.rawValue
    @State var ollamaJSONSchema = ""
    @State var ollamaThinkMode = OllamaThinkMode.off.rawValue
    @State var ollamaKeepAlive = ""
    @State var ollamaLogprobsEnabled = false
    @State var ollamaTopLogprobsText = ""
    @State var ollamaOptionsJSON = ""
    @State var omlxResponseFormat = OMLXResponseFormat.plain.rawValue
    @State var omlxJSONSchema = ""
    @State var omlxIncludeUsageStreamOptions = false
    @State var omlxExtraBodyJSON = ""
    @State var codexAuthFilePath = ""
    @State var codexAuthFileBookmark: Data?
    @State var codexFastModeEnabled = false
    @State var codexAuthFileSelectionError: String?
    @StateObject var operations = RemoteProviderSheetOperations()
    @State var operationToastMessage = ""
    @State var operationToastDismissTask: Task<Void, Never>?

    private var dialogWidth: CGFloat {
        SettingsUIStyle.modelConfigurationDialogWidth
    }

    private var dialogMaxHeight: CGFloat {
        SettingsUIStyle.modelConfigurationDialogMaxHeight
    }

    private var scrollContentMaxHeight: CGFloat {
        SettingsUIStyle.modelConfigurationScrollMaxHeight
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(AppLocalization.format("Configure %@", providerTitle))
                .font(.title3.weight(.semibold))

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    modelSection

                    if !isDoubaoASRTest {
                        endpointAndKeySection
                    }

                    if isCodexLLMProvider {
                        codexConfigurationSection
                    }

                    if showsSearchSection {
                        searchSection
                    }

                    if llmProviderForPicker != nil && !isCodexLLMProvider {
                        advancedGenerationSettingsSection
                    }

                    if usesOpenAIResponsesOptions {
                        openAILLMConfigurationSection
                    }

                    if isOllamaLLMProvider {
                        ollamaConfigurationSection
                    }

                    if isOMLXLLMProvider {
                        omlxConfigurationSection
                    }

                    if showsDoubaoFields {
                        doubaoCredentialsSection
                        doubaoDictionarySection
                    }

                    if isAliyunASRProvider {
                        aliyunASRConfigurationSection
                    }

                    if isStepFunASRProvider {
                        stepFunASRConfigurationSection
                    }

                    if isOpenAIASRTest {
                        openAIChunkSection
                    }

                    if let credentialHint, !credentialHint.isEmpty {
                        Text(credentialHint)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    if let activeProviderNotice {
                        Text(activeProviderNotice)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .padding(.trailing, 4)
            }
            .frame(maxHeight: scrollContentMaxHeight)

            actionSection

            if let result = operations.connectionResult, !result.message.isEmpty {
                Text(result.message)
                    .font(.caption)
                    .foregroundStyle(result.succeeded ? .green : .orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .settingsDialogChrome(width: dialogWidth, maxHeight: dialogMaxHeight, cornerRadius: cornerRadius, onClose: close)
        .overlay(alignment: .top) {
            if !operationToastMessage.isEmpty {
                ModelDebugToast(message: operationToastMessage) {
                    dismissOperationToast()
                }
                .padding(.top, 10)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: operationToastMessage)
        .onChange(of: operations.modelOptions) { _, options in
            if let options, options == operations.modelOptions, !pickerModelOptionIDs.contains(selectedProviderModel) {
                configureModelSelection()
            }
        }
        .onChange(of: operations.connectionResult) { _, result in
            if let result, result == operations.connectionResult, !result.succeeded { showOperationToast(result.message) }
        }
        .onDisappear {
            operations.cancel()
            operationToastDismissTask?.cancel()
        }
        .onAppear {
            operations.activate()
            configureModelSelection()
            customModelID = configuration.model
            endpoint = initialEndpointValue()
            apiKey = configuration.apiKey
            appID = configuration.appID
            accessToken = configuration.accessToken
            editedCredentialFields.removeAll()
            searchEnabled = configuration.searchEnabled
            openAIChunkPseudoRealtimeEnabled = configuration.openAIChunkPseudoRealtimeEnabled
            openAITextVerbosity = configuration.openAITextVerbosity
            configureGenerationSettingsState()
            doubaoDictionaryMode = configuration.doubaoDictionaryMode
            doubaoEnableRequestHotwords = configuration.doubaoEnableRequestHotwords
            doubaoEnableRequestCorrections = configuration.doubaoEnableRequestCorrections
            aliyunMaxSentenceSilenceMillisecondsText = String(configuration.aliyunASRSettings.maxSentenceSilenceMilliseconds)
            aliyunServerVADThresholdText = Self.formatOptionalDouble(configuration.aliyunASRSettings.serverVADThreshold)
            aliyunServerVADSilenceDurationMillisecondsText = String(configuration.aliyunASRSettings.serverVADSilenceDurationMilliseconds)
            aliyunUseManualCommit = configuration.aliyunASRSettings.useManualCommit
            aliyunSemanticPunctuationEnabled = configuration.aliyunASRSettings.semanticPunctuationEnabled
            aliyunPunctuationPredictionEnabled = configuration.aliyunASRSettings.punctuationPredictionEnabled
            aliyunInverseTextNormalizationEnabled = configuration.aliyunASRSettings.inverseTextNormalizationEnabled
            aliyunDisfluencyRemovalEnabled = configuration.aliyunASRSettings.disfluencyRemovalEnabled
            ollamaResponseFormat = configuration.ollamaResponseFormat
            ollamaJSONSchema = configuration.ollamaJSONSchema
            ollamaThinkMode = configuration.ollamaThinkMode
            ollamaKeepAlive = configuration.ollamaKeepAlive
            ollamaLogprobsEnabled = configuration.ollamaLogprobsEnabled
            ollamaTopLogprobsText = configuration.ollamaTopLogprobs.map(String.init) ?? ""
            ollamaOptionsJSON = configuration.ollamaOptionsJSON
            omlxResponseFormat = configuration.omlxResponseFormat
            omlxJSONSchema = configuration.omlxJSONSchema
            omlxIncludeUsageStreamOptions = configuration.omlxIncludeUsageStreamOptions
            omlxExtraBodyJSON = configuration.omlxExtraBodyJSON
            codexAuthFilePath = configuration.codexAuthFilePath
            codexAuthFileBookmark = configuration.codexAuthFileBookmark
            codexFastModeEnabled = configuration.codexFastModeEnabled
            loadCodexModelOptionsIfNeeded()
        }
    }

    func close() {
        operations.cancel()
        operationToastDismissTask?.cancel()
        if let onClose {
            onClose()
        } else {
            dismiss()
        }
    }

}
