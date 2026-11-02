// VoiceEndCommandSettingsSection.swift
// Provides Voice End Command Settings Section for settings screens.

import SwiftUI

struct VoiceEndCommandSettingsSection: View {
    @AppStorage(AppPreferenceKey.voiceEndCommandEnabled) private var voiceEndCommandEnabled = false
    @AppStorage(AppPreferenceKey.voiceEndCommandPreset) private var voiceEndCommandPresetRaw = VoiceEndCommandPreset.over.rawValue
    @AppStorage(AppPreferenceKey.voiceEndCommandText) private var voiceEndCommandText = ""

    private var voiceEndCommandPreset: Binding<VoiceEndCommandPreset> {
        Binding(
            get: { VoiceEndCommandPreset(rawValue: voiceEndCommandPresetRaw) ?? .over },
            set: { voiceEndCommandPresetRaw = $0.rawValue }
        )
    }

    private var voiceEndCommandTextBinding: Binding<String> {
        Binding(
            get: { voiceEndCommandPreset.wrappedValue.resolvedCommand ?? voiceEndCommandText },
            set: { newValue in
                guard voiceEndCommandPreset.wrappedValue == .custom else { return }
                voiceEndCommandText = newValue
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 18) {
                Text("Voice End Command")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.primary.opacity(0.92))

                Spacer()

                Toggle("", isOn: $voiceEndCommandEnabled)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if voiceEndCommandEnabled {
                HStack(alignment: .center, spacing: 12) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Instruction")
                            .font(.body.weight(.semibold))
                            .foregroundStyle(.primary.opacity(0.92))
                        Text("Say this command at the end to end automatically.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    HStack(spacing: 8) {
                        SettingsMenuPicker(
                            selection: voiceEndCommandPreset,
                            options: VoiceEndCommandPreset.allCases.map { preset in
                                SettingsMenuOption(value: preset, title: preset.title)
                            },
                            selectedTitle: voiceEndCommandPreset.wrappedValue.title,
                            width: 132
                        )

                        if voiceEndCommandPreset.wrappedValue == .custom {
                            TextField("over", text: voiceEndCommandTextBinding)
                                .textFieldStyle(.plain)
                                .settingsFieldSurface(width: 172)
                                .transition(.opacity.combined(with: .move(edge: .trailing)))
                        }
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: voiceEndCommandEnabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
