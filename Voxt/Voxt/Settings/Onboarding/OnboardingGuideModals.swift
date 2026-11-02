import SwiftUI
import AppKit

extension OnboardingGuideView {
    @ViewBuilder
    var onboardingModalOverlay: some View {
        if isMicrophoneDialogPresented {
            onboardingModalScrim {
                MicrophonePriorityDialog(
                    state: microphoneState,
                    mode: .selectionOnly,
                    onUseNow: { uid in
                        focusMicrophone(uid: uid)
                        isMicrophoneDialogPresented = false
                    },
                    cornerRadius: OnboardingGuideStyle.modalCornerRadius,
                    onClose: {
                        isMicrophoneDialogPresented = false
                    }
                )
            }
        } else if isUserMainLanguageDialogPresented {
            onboardingModalScrim {
                UserMainLanguageSelectionSheet(
                    selectedCodes: selectedUserMainLanguageCodes,
                    localeIdentifier: interfaceLanguage.localeIdentifier,
                    cornerRadius: OnboardingGuideStyle.modalCornerRadius,
                    onClose: {
                        isUserMainLanguageDialogPresented = false
                    }
                ) { updatedCodes in
                    userMainLanguageCodesRaw = UserMainLanguageOption.storageValue(for: updatedCodes)
                }
            }
        } else if isModelStorageDialogPresented {
            onboardingModalScrim {
                modelStorageDialog
            }
        } else if isTranslationSetupPresented && editingLLMProvider == nil {
            onboardingModalScrim { translationSetupDialog }
        } else if let provider = editingASRProvider {
            onboardingModalScrim {
                RemoteProviderConfigurationSheet(
                    providerTitle: provider.title,
                    credentialHint: asrCredentialHint(for: provider),
                    showsDoubaoFields: provider == .doubaoASR,
                    testTarget: .asr(provider),
                    configuration: RemoteModelConfigurationStore.resolvedASRConfiguration(
                        provider: provider,
                        from: remoteASRProviderConfigurationsRaw
                    ),
                    onSave: saveRemoteASRConfiguration(_:),
                    cornerRadius: OnboardingGuideStyle.modalCornerRadius,
                    onClose: {
                        editingASRProvider = nil
                    }
                )
            }
        } else if let provider = editingLLMProvider {
            onboardingModalScrim {
                RemoteProviderConfigurationSheet(
                    providerTitle: provider.title,
                    credentialHint: nil,
                    showsDoubaoFields: false,
                    testTarget: .llm(provider),
                    configuration: RemoteModelConfigurationStore.resolvedLLMConfiguration(
                        provider: provider,
                        from: remoteLLMProviderConfigurationsRaw
                    ),
                    onSave: saveRemoteLLMConfiguration(_:),
                    cornerRadius: OnboardingGuideStyle.modalCornerRadius,
                    onClose: {
                        editingLLMProvider = nil
                    }
                )
            }
        } else if let shortcut = editingShortcut {
            onboardingModalScrim {
                shortcutSheet(for: shortcut)
            }
        }
    }

    private func onboardingModalScrim<Content: View>(
        @ViewBuilder content: () -> Content
    ) -> some View {
        ZStack {
            OnboardingGuideStyle.modalScrim
                .contentShape(Rectangle())

            content()
        }
        .frame(width: Self.windowSize.width, height: Self.windowSize.height)
        .clipShape(RoundedRectangle(cornerRadius: OnboardingGuideStyle.windowCornerRadius, style: .continuous))
        .zIndex(10)
    }

    private var translationSetupDialog: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(AppLocalization.localizedString("Configure Translation Model")).font(.headline)
            Text(AppLocalization.localizedString("Choose a model for translation. Other features stay unchanged."))
                .font(.callout).foregroundStyle(.secondary)
            ScrollView {
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(AppLocalization.localizedString("Local")).font(.headline)
                        ForEach(displayedLocalLLMRepos, id: \.self) { localLLMModelRow(repo: $0) }
                    }.frame(maxWidth: .infinity)
                    VStack(alignment: .leading, spacing: 8) {
                        Text(AppLocalization.localizedString("Remote")).font(.headline)
                        ForEach(displayedRemoteLLMProviders) { provider in
                            remoteProviderRow(
                                title: provider.title,
                                isSelected: selectedTranslationModel == .remoteLLM(provider),
                                isConfigured: isTranslationModelReady(.remoteLLM(provider)),
                                onSelect: { modelDraft.translation = .remoteLLM(provider) },
                                onConfigure: { editingLLMProvider = provider }
                            )
                        }
                    }.frame(maxWidth: .infinity)
                }
            }.frame(height: 300)
            HStack {
                Button(AppLocalization.localizedString("More")) {
                    showsMoreLocalLLMModels = true
                    showsMoreRemoteLLMProviders = true
                }
                Spacer()
                Button(AppLocalization.localizedString("Apply")) {
                    commitModelDraft()
                    isTranslationSetupPresented = false
                    updateFocusedField()
                }
                .buttonStyle(OnboardingGuidePrimaryButtonStyle())
                .disabled(!isTranslationModelReady(selectedTranslationModel))
            }
        }
        .settingsDialogChrome(width: 760, cornerRadius: OnboardingGuideStyle.modalCornerRadius, onClose: {
            modelDraft = OnboardingModelDraft()
            isTranslationSetupPresented = false
            updateFocusedField()
        })
    }
}
