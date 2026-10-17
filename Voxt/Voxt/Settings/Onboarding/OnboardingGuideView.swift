// Onboarding view state stays in OnboardingGuideView; sibling files implement individual steps.
import SwiftUI
import AppKit

struct OnboardingGuideView: View {
    @Binding var currentStep: OnboardingGuideStep

    @ObservedObject var mlxModelManager: MLXModelManager
    @ObservedObject var customLLMManager: CustomLLMModelManager
    @ObservedObject var overlayState: OverlayState

    let onClose: () -> Void
    let onFinish: () -> Void

    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @AppStorage(AppPreferenceKey.userMainLanguageCodes) var userMainLanguageCodesRaw = UserMainLanguageOption.defaultStoredSelectionValue
    @AppStorage(AppPreferenceKey.modelStorageRootPath) private var modelStorageRootPath = ""
    @AppStorage(AppPreferenceKey.mlxModelRepo) var mlxModelRepo = MLXModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.customLLMModelRepo) var customLLMRepo = CustomLLMModelManager.defaultModelRepo
    @AppStorage(AppPreferenceKey.remoteASRSelectedProvider) var remoteASRSelectedProviderRaw = RemoteASRProvider.openAIWhisper.rawValue
    @AppStorage(AppPreferenceKey.remoteASRProviderConfigurations) var remoteASRProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.remoteLLMSelectedProvider) var remoteLLMSelectedProviderRaw = RemoteLLMProvider.openAI.rawValue
    @AppStorage(AppPreferenceKey.remoteLLMProviderConfigurations) var remoteLLMProviderConfigurationsRaw = ""
    @AppStorage(AppPreferenceKey.hotkeyPreset) var hotkeyPresetRaw = HotkeyPreference.defaultPreset.rawValue
    @AppStorage(AppPreferenceKey.hotkeyDistinguishModifierSides) var distinguishModifierSides = HotkeyPreference.defaultDistinguishModifierSides

    @State var inputDevices: [AudioInputDevice] = []
    @State var microphoneState = MicrophoneResolvedState.empty
    @State var permissionRefreshRevision = 0
    @State var permissionMonitoringKinds: Set<OnboardingContextualPermission> = []
    @State var permissionMonitorTasks: [OnboardingContextualPermission: Task<Void, Never>] = [:]
    @State var modelFocus: OnboardingGuideModelFocus = .local
    @State var modelDraft = OnboardingModelDraft()
    @State var practice = OnboardingPracticeState()
    @State var practiceAttempt = UUID()
    @State var isTranslationSetupPresented = false
    @State private var hasLoadedSettings = false
    @Environment(\.accessibilityReduceMotion) var reduceMotion
    @State var showsMoreLocalASRModels = false
    @State var showsMoreLocalLLMModels = false
    @State var showsMoreRemoteASRProviders = false
    @State var showsMoreRemoteLLMProviders = false
    @State var modelStorageDisplayPath = ""
    @State var modelStorageSelectionError: String?
    @State var featureSettings = FeatureSettings.placeholder
    @State var isMicrophoneDialogPresented = false
    @State var isUserMainLanguageDialogPresented = false
    @State var isModelStorageDialogPresented = false
    @State var editingASRProvider: RemoteASRProvider?
    @State var editingLLMProvider: RemoteLLMProvider?
    @State var editingShortcut: OnboardingGuideShortcutKind?
    @State var microphoneHasDetectedAudio = false
    @State var microphoneSignalFrameCount = 0
    @State var microphoneReceivedInitialBuffer = false
    @State var microphoneStartupRetryCount = 0
    @State var microphoneStartupWatchdogTask: Task<Void, Never>?
    @State var microphoneRefreshTask: Task<Void, Never>?
    @State var transcriptionInput = ""
    @State var translationInput = ""
    @State var selectionInput = Self.defaultTranslationSample
    @State var selectedTranslationRange = NSRange(location: 0, length: 0)
    @State var microphoneCapture: MeetingMicrophoneCapture?

    @FocusState var focusedField: OnboardingGuideFocusField?

    private static let defaultTranslationSampleKey = "Could we move our meeting to tomorrow afternoon?"
    static var defaultTranslationSample: String {
        AppLocalization.localizedString(Self.defaultTranslationSampleKey)
    }
    static let windowSize = CGSize(width: 880, height: 600)
    private static let outerPadding: CGFloat = 12
    private static let outerBottomPadding: CGFloat = 12
    private static let shellHeaderHeight: CGFloat = 58
    private static let shellSideCutoutWidth: CGFloat = 58
    private static let contentBottomCompensation: CGFloat = 0
    static let microphoneSignalThreshold: Float = 0.006
    static let microphoneRequiredSignalFrames = 2
    static let microphoneStartupWatchdogDelay: Duration = .milliseconds(1200)
    static let collapsedModelListLimit = 3
    static let preferredLocalASRRepos = [
        "mlx-community/SenseVoiceSmall",
        "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
        "mlx-community/Qwen3-ASR-1.7B-4bit",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "mlx-community/whisper-large-v3-turbo"
    ]
    static let preferredLocalLLMRepos = [
        "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/Qwen3.5-4B-OptiQ-4bit"
    ]
    static let preferredRemoteASRProviders: [RemoteASRProvider] = [
        .doubaoASR,
        .aliyunBailianASR,
        .stepFunASR,
        .xiaomiMiMoASR
    ]
    static let preferredRemoteLLMProviders: [RemoteLLMProvider] = [
        .volcengine,
        .aliyunBailian,
        .stepFun,
        .xiaomiMiMo
    ]

    var interfaceLanguage: AppInterfaceLanguage {
        AppInterfaceLanguage(rawValue: interfaceLanguageRaw) ?? .system
    }

    var selectedUserMainLanguageCodes: [String] {
        UserMainLanguageOption.storedSelection(from: userMainLanguageCodesRaw)
    }

    var userMainLanguageSummary: String {
        GeneralSettingsData.userMainLanguageSummary(
            selectedCodes: selectedUserMainLanguageCodes,
            locale: interfaceLanguage.locale
        )
    }

    var canContinue: Bool {
        switch currentStep {
        case .permissions: return areRequiredPermissionsGranted && !isPracticeSessionActive
        case .models: return modelStepReady && !isPracticeSessionActive
        case .transcriptionShortcut, .translationShortcut, .translationSelection:
            return practice.phase == .succeeded && !isPracticeSessionActive
        case .finish: return true
        }
    }

    var body: some View { guideWithNotifications }

    private var guideShell: some View {
        ZStack {
            GeometryReader { proxy in
                let shellHeight = max(
                    0,
                    proxy.size.height
                        - Self.outerPadding
                        - Self.outerBottomPadding
                        + Self.contentBottomCompensation
                )
                let contentHeight = max(0, shellHeight - Self.shellHeaderHeight)

                ZStack(alignment: .top) {
                    VStack(spacing: 0) {
                        headerNavigation
                            .frame(height: Self.shellHeaderHeight)

                        content
                            .frame(maxWidth: .infinity)
                            .frame(height: contentHeight)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: shellHeight)
                    .background(
                        OnboardingGuideShellShape(
                            headerHeight: Self.shellHeaderHeight,
                            sideCutoutWidth: Self.shellSideCutoutWidth,
                            cornerRadius: OnboardingGuideStyle.panelCornerRadius,
                            transitionRadius: OnboardingGuideStyle.headerTransitionRadius
                        )
                        .fill(OnboardingGuideStyle.panelFill)
                    )
                    .overlay(
                        OnboardingGuideShellShape(
                            headerHeight: Self.shellHeaderHeight,
                            sideCutoutWidth: Self.shellSideCutoutWidth,
                            cornerRadius: OnboardingGuideStyle.panelCornerRadius,
                            transitionRadius: OnboardingGuideStyle.headerTransitionRadius
                        )
                        .stroke(OnboardingGuideStyle.panelBorder, lineWidth: 1)
                    )
                    .clipShape(
                        OnboardingGuideShellShape(
                            headerHeight: Self.shellHeaderHeight,
                            sideCutoutWidth: Self.shellSideCutoutWidth,
                            cornerRadius: OnboardingGuideStyle.panelCornerRadius,
                            transitionRadius: OnboardingGuideStyle.headerTransitionRadius
                        )
                    )

                    topChrome
                        .frame(height: Self.shellHeaderHeight)
                }
                .padding(.top, Self.outerPadding)
                .padding(.horizontal, Self.outerPadding)
                .padding(.bottom, Self.outerBottomPadding)
                .frame(width: proxy.size.width, height: proxy.size.height, alignment: .top)
            }
            .frame(width: Self.windowSize.width, height: Self.windowSize.height)

            onboardingModalOverlay
        }
    }

    private var styledGuideShell: some View {
        guideShell
        .background(OnboardingGuideStyle.windowBackground)
        .clipShape(RoundedRectangle(cornerRadius: OnboardingGuideStyle.windowCornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: OnboardingGuideStyle.windowCornerRadius, style: .continuous)
                .strokeBorder(OnboardingGuideStyle.windowBorder, lineWidth: 1)
        )
        .frame(width: Self.windowSize.width, height: Self.windowSize.height)
        .environment(\.locale, interfaceLanguage.locale)
        .groupBoxStyle(SettingsPanelGroupBoxStyle())
    }

    private var guideWithStateLifecycle: some View {
        styledGuideShell
        .onAppear {
            refreshInputDevices()
            refreshModelStorageDisplayPath()
            refreshLocalizedGuideSamples()
            reloadFeatureSettings()
            if !hasLoadedSettings {
                if case .remote? = featureSettings.transcription.asrSelectionID.asrSelection {
                    modelFocus = .remote
                }
                hasLoadedSettings = true
            }
            updateFocusedField()
            updateMicrophoneCapture()
        }
        .onDisappear {
            cancelPracticeSessionIfNeeded()
            stopMicrophoneMeter()
            microphoneRefreshTask?.cancel()
            microphoneRefreshTask = nil
            for task in permissionMonitorTasks.values {
                task.cancel()
            }
        }
        .onChange(of: currentStep) { _, newStep in
            practice = OnboardingPracticeState()
            practiceAttempt = UUID()
            selectedTranslationRange = NSRange(location: 0, length: 0)
            OnboardingPreferenceManager.saveLastGuideStep(newStep)
            reloadFeatureSettings()
            refreshLocalizedGuideSamples()
            updateFocusedField()
            updateMicrophoneCapture()
        }
        .task(id: currentStep) {
            await Task.yield()
            guard !Task.isCancelled else { return }
            updateFocusedField()
        }
        .onChange(of: interfaceLanguageRaw) { _, _ in
            refreshLocalizedGuideSamples()
        }
        .onChange(of: modelStorageRootPath) { _, _ in
            refreshModelStorageDisplayPath()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtFeatureSettingsDidChange)) { _ in
            reloadFeatureSettings()
        }
        .onReceive(NotificationCenter.default.publisher(for: OnboardingSessionEvent.notification)) { notification in
            guard currentStep.isPractice, !isTranslationSetupPresented, editingShortcut == nil,
                  let event = notification.object as? OnboardingSessionEvent else { return }
            practice.receive(
                event,
                expectedKind: practiceKind,
                windowNumber: AppDelegate.shared?.onboardingWindowController?.window?.windowNumber
            )
        }
        .onChange(of: overlayState.isRecording) { _, isRecording in
            if isRecording, let sessionID = AppDelegate.shared?.activeRecordingSessionID {
                practice.startedListening(sessionID: sessionID)
            }
        }
    }

    private var guideWithNotifications: some View {
        guideWithStateLifecycle
        .onReceive(NotificationCenter.default.publisher(for: .voxtAudioInputDevicesDidChange)) { _ in
            scheduleMicrophoneRefresh()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtSelectedInputDeviceDidChange)) { _ in
            scheduleMicrophoneRefresh()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voxtRemoteProviderConfigurationsDidChange)) { _ in
            reloadFeatureSettings()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            permissionRefreshRevision += 1
            scheduleMicrophoneRefresh()
        }
    }

    private var topChrome: some View {
        HStack {
            Button {
                cancelPracticeSessionIfNeeded()
                onClose()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .bold))
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(OnboardingGuideIconButtonStyle())
            .help(AppLocalization.localizedString("Exit Guide"))

            Spacer(minLength: 0)

            OnboardingGuideProgressRing(
                current: currentStep.stepNumber,
                total: OnboardingGuideStep.allCases.count
            )
        }
        .padding(.horizontal, 12)
    }

    private var headerNavigation: some View {
        HStack(spacing: 8) {
            if let previous = currentStep.previous {
                OnboardingGuideHeaderStepButton(
                    title: previous.title,
                    alignment: .trailing,
                    isEnabled: !isPracticeSessionActive,
                    action: {
                        currentStep = previous
                    }
                )
            } else {
                Color.clear
                    .frame(maxWidth: .infinity)
            }

            Text(currentStep.title)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(OnboardingGuideStyle.primaryText)
                .lineLimit(1)
                .minimumScaleFactor(0.82)
                .frame(width: 200)

            if let next = currentStep.next {
                OnboardingGuideHeaderStepButton(
                    title: next.title,
                    alignment: .leading,
                    isEnabled: canContinue,
                    action: {
                        guard canContinue else { return }
                        advanceStep()
                    }
                )
                .help(canContinue ? "" : continueDisabledHelp)
            } else {
                Color.clear
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(.horizontal, Self.shellSideCutoutWidth + 18)
    }

    @ViewBuilder
    private var content: some View {
        if currentStep == .permissions {
            permissionsGuidePanel
        } else if currentStep == .models {
            modelGuidePanel
        } else if currentStep == .finish {
            discoveryPanel
        } else {
            practicePanel
        }
    }

    var footer: some View {
        HStack(spacing: 8) {
            if currentStep == .finish {
                Spacer(minLength: 0)
                Button {
                    OnboardingPreferenceManager.markCompleted()
                    onFinish()
                } label: {
                    Label(AppLocalization.localizedString("Start Voxt"), systemImage: "checkmark.circle")
                }
                .buttonStyle(OnboardingGuidePrimaryButtonStyle())
                Spacer(minLength: 0)
            } else if currentStep.next != nil {
                Spacer(minLength: 0)

                leadingFooterAction

                Button {
                    advanceStep()
                } label: {
                    Label(AppLocalization.localizedString("Continue"), systemImage: "chevron.right")
                        .labelStyle(OnboardingGuideNextLabelStyle())
                }
                .buttonStyle(OnboardingGuidePrimaryButtonStyle())
                .disabled(!canContinue)
                .help(canContinue ? "" : continueDisabledHelp)
            }
        }
        .padding(.top, 10)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(OnboardingGuideStyle.panelBorder)
                .frame(height: 1)
        }
    }

    @ViewBuilder
    private var leadingFooterAction: some View {
        if currentStep.isPractice {
            Button(AppLocalization.localizedString("Try Later")) { advanceStep() }
                .buttonStyle(OnboardingGuideSecondaryButtonStyle())
                .disabled(isPracticeSessionActive)
        }
    }

    var continueDisabledHelp: String {
        switch currentStep {
        case .permissions:
            return AppLocalization.localizedString("Grant all listed permissions to continue.")
        case .models:
            return AppLocalization.localizedString("Prepare the selected speech model to continue. Translation is optional.")
        default:
            return AppLocalization.localizedString("Complete this test to continue.")
        }
    }

    func reloadFeatureSettings() {
        featureSettings = FeatureSettingsStore.load(defaults: .standard)
    }

    func commitModelDraft() {
        guard modelDraft.hasChanges else { return }
        FeatureSettingsStore.update { current in
            current = modelDraft.applying(to: current)
        }
        modelDraft = OnboardingModelDraft()
        reloadFeatureSettings()
    }

    func advanceStep() {
        guard !isPracticeSessionActive, let next = currentStep.next else { return }
        if currentStep == .models { commitModelDraft() }
        currentStep = next
    }
}
