// CustomLLMGenerationSettingsSheet.swift
// Provides Custom LLMGeneration Settings Sheet for model settings.

import SwiftUI

private func customLLMLocalized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct CustomLLMGenerationSettingsSheet: View {
    let modelTitle: String
    @Binding var settings: LLMGenerationSettings
    let onDone: () -> Void

    @State private var maxOutputTokensText = ""
    @State private var temperatureText = ""
    @State private var topPText = ""
    @State private var topKText = ""
    @State private var minPText = ""
    @State private var repetitionPenaltyText = ""
    @State private var thinkingMode = LLMThinkingMode.providerDefault.rawValue
    @State private var advancedExpanded = false

    private var thinkingOptions: [SettingsMenuOption<String>] {
        [
            SettingsMenuOption(value: LLMThinkingMode.off.rawValue, title: customLLMLocalized("Off")),
            SettingsMenuOption(value: LLMThinkingMode.on.rawValue, title: customLLMLocalized("On"))
        ]
    }

    private var selectedThinkingTitle: String {
        thinkingOptions.first(where: { $0.value == thinkingMode })?.title ?? customLLMLocalized("Off")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(customLLMLocalized("Local LLM Configuration"))
                .font(.title3.weight(.semibold))

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text(modelTitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    VStack(alignment: .leading, spacing: 12) {
                        Text(customLLMLocalized("Configuration"))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        VStack(alignment: .leading, spacing: 8) {
                            Text(customLLMLocalized("Think"))
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            SettingsMenuPicker(
                                selection: $thinkingMode,
                                options: thinkingOptions,
                                selectedTitle: selectedThinkingTitle,
                                width: 240
                            )
                            Text(customLLMLocalized("Use Voxt's default thinking behavior for this model."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        numericField(
                            title: customLLMLocalized("Max Output Tokens (Optional)"),
                            placeholder: "1024",
                            text: $maxOutputTokensText
                        )

                        HStack(alignment: .top, spacing: 12) {
                            numericField(
                                title: customLLMLocalized("Temperature"),
                                placeholder: "0",
                                text: $temperatureText
                            )
                            numericField(
                                title: customLLMLocalized("Top P"),
                                placeholder: "1.0",
                                text: $topPText
                            )
                        }

                        VStack(alignment: .leading, spacing: 12) {
                            Button {
                                withAnimation(.easeInOut(duration: 0.12)) {
                                    advancedExpanded.toggle()
                                }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: advancedExpanded ? "chevron.down" : "chevron.right")
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.secondary)
                                        .frame(width: 12)
                                    Text(customLLMLocalized("Advanced"))
                                        .font(.subheadline)
                                }
                            }
                            .buttonStyle(.plain)

                            if advancedExpanded {
                                VStack(alignment: .leading, spacing: 12) {
                                    HStack(alignment: .top, spacing: 12) {
                                        numericField(
                                            title: customLLMLocalized("Top K"),
                                            placeholder: "0",
                                            text: $topKText
                                        )
                                        numericField(
                                            title: customLLMLocalized("Min P"),
                                            placeholder: "0",
                                            text: $minPText
                                        )
                                    }

                                    numericField(
                                        title: customLLMLocalized("Repetition Penalty"),
                                        placeholder: "1.05",
                                        text: $repetitionPenaltyText
                                    )
                                }
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .padding(.trailing, 4)
            }
            .frame(maxHeight: SettingsUIStyle.modelConfigurationScrollMaxHeight)

            SettingsDialogActionRow {
                Button(customLLMLocalized("Reset to Default")) {
                    applySettings(CustomLLMGenerationSettingsStore.defaultSettings)
                }
                .buttonStyle(SettingsPillButtonStyle())
            } trailing: {
                Button(customLLMLocalized("Cancel")) {
                    onDone()
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button(customLLMLocalized("Save")) {
                    save()
                    onDone()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(
            width: SettingsUIStyle.modelConfigurationDialogWidth,
            maxHeight: SettingsUIStyle.modelConfigurationDialogMaxHeight,
            onClose: onDone
        )
        .onAppear {
            applySettings(settings)
        }
    }

    private func numericField(
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

    private func applySettings(_ next: LLMGenerationSettings) {
        let sanitized = CustomLLMGenerationSettingsStore.sanitized(next)
        maxOutputTokensText = sanitized.maxOutputTokens.map(String.init) ?? ""
        temperatureText = sanitized.temperature.map { String(format: "%.3g", $0) } ?? ""
        topPText = sanitized.topP.map { String(format: "%.3g", $0) } ?? ""
        topKText = sanitized.topK.map(String.init) ?? ""
        minPText = sanitized.minP.map { String(format: "%.3g", $0) } ?? ""
        repetitionPenaltyText = sanitized.repetitionPenalty.map { String(format: "%.3g", $0) } ?? ""
        thinkingMode = sanitized.thinking.mode.rawValue
    }

    private func save() {
        settings = CustomLLMGenerationSettingsStore.sanitized(
            LLMGenerationSettings(
                maxOutputTokens: Int(maxOutputTokensText.trimmingCharacters(in: .whitespacesAndNewlines)),
                temperature: Double(temperatureText.trimmingCharacters(in: .whitespacesAndNewlines)),
                topP: Double(topPText.trimmingCharacters(in: .whitespacesAndNewlines)),
                topK: Int(topKText.trimmingCharacters(in: .whitespacesAndNewlines)),
                minP: Double(minPText.trimmingCharacters(in: .whitespacesAndNewlines)),
                repetitionPenalty: Double(repetitionPenaltyText.trimmingCharacters(in: .whitespacesAndNewlines)),
                thinking: LLMThinkingSettings(
                    mode: LLMThinkingMode(rawValue: thinkingMode) ?? .providerDefault,
                    effort: nil,
                    budgetTokens: nil,
                    exposeReasoning: false
                )
            )
        )
    }
}
