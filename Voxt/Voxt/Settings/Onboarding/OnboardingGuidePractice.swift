import SwiftUI
import AppKit

extension OnboardingGuideView {
    var practiceKind: OnboardingPracticeKind {
        switch currentStep {
        case .translationShortcut: return .voiceTranslation
        case .translationSelection: return .selectedTextTranslation
        default: return .transcription
        }
    }

    private var practiceShortcut: OnboardingGuideShortcutKind {
        currentStep == .transcriptionShortcut ? .transcription : .translation
    }

    var isPracticeSessionActive: Bool {
        practice.isBusy || AppDelegate.shared?.isSessionActive == true
    }

    private var practiceSample: String {
        AppLocalization.localizedString(currentStep == .transcriptionShortcut
            ? "We have a meeting tomorrow at three. Remember to bring the project notes."
            : "Hello, is there a coffee shop nearby?")
    }

    private var practiceStopInstruction: String {
        if let stopBinding = shortcutBindings(for: .transcription).first(where: {
            $0.behavior != .longPress && HotkeyModifierInterpreter.isModifierOnly($0.hotkey)
        }) {
            return AppLocalization.format(
                "Hold to speak: release to finish. Otherwise, tap %@ or choose Finish Speaking.",
                HotkeyPreference.displayString(for: stopBinding.hotkey, distinguishModifierSides: distinguishModifierSides)
            )
        }
        return AppLocalization.localizedString("Hold shortcuts: release to finish. Tap or double-tap shortcuts: use Finish Speaking below.")
    }

    private var practiceInstruction: String {
        switch practice.phase {
        case .starting: return AppLocalization.localizedString("Getting ready. Wait for the microphone before speaking.")
        case .listening: return AppLocalization.localizedString("Listening. Read the sample aloud, or say something of your own.")
        case .processing: return AppLocalization.localizedString("Working on your words…")
        case .succeeded:
            return AppLocalization.localizedString(currentStep == .translationSelection
                ? "Same shortcut: select text to translate it, or speak when nothing is selected."
                : "That's it! You can do the same in a chat, email, or document.")
        case .failed:
            return practice.message.isEmpty
                ? AppLocalization.localizedString("No result was delivered. Check the microphone and model, then try again.")
                : practice.message
        case .ready:
            if currentStep == .translationSelection && selectedTranslationRange.length == 0 {
                return AppLocalization.localizedString("First, drag to select the sample.")
            }
            return AppLocalization.localizedString("Use the shortcut below to begin.")
        }
    }

    var practicePanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(currentStep.subtitle).font(.callout).foregroundStyle(.secondary)
            HStack(alignment: .top, spacing: 24) {
                ScrollView { practiceInstructions.frame(maxWidth: .infinity, alignment: .topLeading) }
                    .frame(width: 300)
                practiceResult.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .frame(maxHeight: .infinity)
            footer
        }
        .padding(20)
    }

    private var practiceInstructions: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(practiceInstruction, systemImage: practice.phase == .succeeded ? "checkmark.circle.fill" : "hand.point.up.left")
                .font(.headline)
                .foregroundStyle(practice.phase == .succeeded ? Color.green : Color.primary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.updatesFrequently)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(shortcutBindings(for: practiceShortcut)) { binding in
                    HStack {
                        Text(binding.behavior.title).font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Text(HotkeyPreference.displayString(for: binding.hotkey, distinguishModifierSides: distinguishModifierSides))
                            .font(.system(.body, design: .rounded).weight(.semibold))
                    }
                }
                if currentStep != .translationSelection {
                    Text(practiceStopInstruction)
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .background(Color.accentColor.opacity(practice.phase == .ready ? 0.12 : 0.04), in: RoundedRectangle(cornerRadius: 12))
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: practice.phase)

            if practice.phase == .listening {
                Button(AppLocalization.localizedString("Finish Speaking")) {
                    updateFocusedField()
                    AppDelegate.shared?.endRecording()
                }
                .buttonStyle(OnboardingGuidePrimaryButtonStyle())
                ProgressView(value: Double(min(max(overlayState.audioLevel, 0), 1)))
                    .accessibilityLabel(AppLocalization.localizedString("Microphone Level"))
            } else if practice.phase == .starting || practice.phase == .processing {
                ProgressView().controlSize(.small)
            }

            if practice.isBusy {
                Button(AppLocalization.localizedString("Cancel")) {
                    cancelPracticeSessionIfNeeded()
                }
                .buttonStyle(OnboardingGuideSecondaryButtonStyle())
            }

            if currentStep != .transcriptionShortcut {
                translationPracticeControls
            }

            Button(AppLocalization.localizedString("Change Shortcut")) { editingShortcut = practiceShortcut }
                .buttonStyle(OnboardingGuideSecondaryButtonStyle())
                .disabled(isPracticeSessionActive)

            if practice.phase == .succeeded || practice.phase == .failed {
                Button(AppLocalization.localizedString("Try Again")) { resetPractice() }
                    .buttonStyle(OnboardingGuideSecondaryButtonStyle())
                    .disabled(isPracticeSessionActive)
            }
        }
    }

    @ViewBuilder
    private var practiceResult: some View {
        VStack(alignment: .leading, spacing: 12) {
            if currentStep == .translationSelection {
                Text(AppLocalization.localizedString("Select this sample")).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                SelectableGuideTextView(text: $selectionInput, selectedRange: $selectedTranslationRange)
                    .id(practiceAttempt)
                    .frame(height: 110)
                    .background(SettingsUIStyle.controlFillColor, in: RoundedRectangle(cornerRadius: 10))
                if !practice.result.isEmpty {
                    Text(AppLocalization.localizedString("Your Result")).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    ScrollView {
                        Text(practice.result).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            } else {
                Text(AppLocalization.localizedString("Read this aloud")).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(practiceSample)
                    .font(.title3.weight(.medium))
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.accentColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
                Text(AppLocalization.localizedString("Your words will appear here")).font(.caption).foregroundStyle(.secondary)
                if currentStep == .transcriptionShortcut {
                    TextEditor(text: $transcriptionInput)
                        .focused($focusedField, equals: .transcription)
                        .settingsPromptEditor(height: 150, contentPadding: 8)
                } else {
                    TextEditor(text: $translationInput)
                        .focused($focusedField, equals: .translation)
                        .settingsPromptEditor(height: 150, contentPadding: 8)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .background(OnboardingGuideStyle.cardFill, in: RoundedRectangle(cornerRadius: 14))
    }

    private var translationPracticeControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("Choose a target language different from the sample."))
                .font(.caption).foregroundStyle(.secondary)
            SettingsMenuPicker(
                selection: Binding(
                    get: { featureSettings.translation.targetLanguageRawValue },
                    set: { value in
                        FeatureSettingsStore.update { $0.translation.targetLanguageRawValue = value }
                        reloadFeatureSettings()
                        updateFocusedField()
                    }
                ),
                options: TranslationTargetLanguage.allCases.map {
                    SettingsMenuOption(value: $0.rawValue, title: $0.title)
                },
                selectedTitle: featureSettings.translation.targetLanguage.title,
                width: 240
            )
            if !featureSettings.availability.translationEnabled {
                Button(AppLocalization.localizedString("Enable Translation")) {
                    FeatureSettingsStore.update { $0.availability.translationEnabled = true }
                    reloadFeatureSettings()
                }
            }
            if currentStep == .translationShortcut && !isSpeechModelReady(featureSettings.translation.asrSelectionID) {
                Button(AppLocalization.localizedString("Use Voice Input's Speech Model")) {
                    FeatureSettingsStore.update { $0.translation.asrSelectionID = featureSettings.transcription.asrSelectionID }
                    reloadFeatureSettings()
                    updateFocusedField()
                }
            }
            if !isTranslationModelReady(featureSettings.translation.modelSelectionID) {
                Text(AppLocalization.localizedString("Translation needs a text model. Configure one, or try this later."))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Button(AppLocalization.localizedString("Configure Translation Model")) {
                modelDraft = OnboardingModelDraft()
                isTranslationSetupPresented = true
            }
            .buttonStyle(OnboardingGuideSecondaryButtonStyle())
        }
        .disabled(isPracticeSessionActive)
    }

    var discoveryPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(currentStep.subtitle).font(.callout).foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 14) {
                discoveryCard("Rewrite", icon: "pencil.line", description: "Select text and ask Voxt to make it clearer or shorter.", hint: shortcutDisplay(for: .rewrite))
                discoveryCard("Notes", icon: "note.text", description: "Capture a passing thought as a note without opening a document.", hint: noteShortcutSummary)
                discoveryCard("Meeting", icon: "person.2.wave.2", description: "Record a meeting, review the transcript, and generate a summary.", hint: shortcutDisplay(for: .meeting))
                discoveryCard("File Transcription", icon: "doc.waveform", description: "Import an audio or video file and turn it into searchable text.", hint: AppLocalization.localizedString("Open Files from the main window"))
            }
            Spacer(minLength: 0)
            footer
        }
        .padding(24)
    }

    private var noteShortcutSummary: String {
        let bindings = HotkeyPreference.loadNoteBindings()
        return bindings.isEmpty ? AppLocalization.localizedString("Open Notes from the main window") : bindings.map {
            HotkeyPreference.displayString(for: $0.hotkey, distinguishModifierSides: distinguishModifierSides)
        }.joined(separator: " / ")
    }

    private func discoveryCard(_ title: String, icon: String, description: String, hint: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(AppLocalization.localizedString(title), systemImage: icon).font(.headline)
            Text(AppLocalization.localizedString(description)).font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Text(hint).font(.system(.caption, design: .rounded).weight(.medium)).foregroundStyle(Color.accentColor)
        }
        .padding(18)
        .frame(maxWidth: .infinity, minHeight: 145, alignment: .topLeading)
        .background(OnboardingGuideStyle.cardFill, in: RoundedRectangle(cornerRadius: 14))
    }

    func cancelPracticeSessionIfNeeded() {
        guard let sessionID = practice.sessionID,
              AppDelegate.shared?.activeRecordingSessionID == sessionID,
              AppDelegate.shared?.isSessionActive == true else { return }
        AppDelegate.shared?.cancelActiveRecordingSession()
    }

    private func resetPractice() {
        guard !isPracticeSessionActive else { return }
        practice = OnboardingPracticeState()
        practiceAttempt = UUID()
        transcriptionInput = ""
        translationInput = ""
        selectionInput = Self.defaultTranslationSample
        selectedTranslationRange = NSRange(location: 0, length: 0)
        updateFocusedField()
    }

    func refreshLocalizedGuideSamples() {
        guard !isPracticeSessionActive else { return }
        selectionInput = Self.defaultTranslationSample
    }

    func updateFocusedField() {
        switch currentStep {
        case .transcriptionShortcut:
            focusedField = .transcription
        case .translationShortcut:
            focusedField = .translation
        default:
            focusedField = nil
        }
    }
}

enum OnboardingGuideFocusField: Hashable {
    case transcription
    case translation
}
