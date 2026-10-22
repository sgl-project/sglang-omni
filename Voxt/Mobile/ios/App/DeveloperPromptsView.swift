import SwiftUI
import VoxtMobileKit

/// Settings > Developer > Prompts: one editor per mode, starting from the Mac's built-in prompt.
struct DeveloperPromptsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var drafts: [CompositionOptions.Mode: String] = [:]
    @State private var confirmResetAll = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text(VoxtL10n.shared.string("Each prompt replaces Voxt's built-in instruction for that mode. Custom prompts can lower quality. An empty prompt uses the built-in one."))
                        .font(VoxtTheme.Typography.caption)
                        .foregroundStyle(.secondary)
                    if let error = model.builtInPromptsError {
                        Text(VoxtL10n.shared.string(error))
                            .font(VoxtTheme.Typography.caption)
                            .foregroundStyle(.red)
                            .accessibilityIdentifier("builtInPromptsError")
                    }
                }
                ForEach(CompositionOptions.Mode.allCases, id: \.self) { mode in
                    Section {
                        TextEditor(text: Binding(
                            get: { drafts[mode] ?? model.promptText(for: mode) },
                            set: { text in
                                drafts[mode] = text
                                model.setCustomPrompt(text, for: mode)
                            }
                        ))
                        .font(.system(size: 13, design: .monospaced))
                        .frame(minHeight: 160)
                        .accessibilityIdentifier("prompt-\(mode.rawValue)")
                        Button(VoxtL10n.shared.string("Reset to default")) {
                            model.resetCustomPrompt(for: mode)
                            drafts[mode] = nil
                        }
                        .disabled(!model.hasCustomPrompt(for: mode))
                        .accessibilityIdentifier("resetPrompt-\(mode.rawValue)")
                    } header: {
                        HStack {
                            Text(mode.title)
                            if model.hasCustomPrompt(for: mode) {
                                Text(VoxtL10n.shared.string("Custom")).foregroundStyle(VoxtTheme.Colors.accent)
                                    .accessibilityIdentifier("customBadge-\(mode.rawValue)")
                            }
                        }
                    }
                }
                Section {
                    Button(VoxtL10n.shared.string("Reset all to default"), role: .destructive) { confirmResetAll = true }
                        .accessibilityIdentifier("resetAllPrompts")
                }
            }
            .navigationTitle(VoxtL10n.shared.string("Prompts"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button(VoxtL10n.shared.string("Done")) { dismiss() } }
            .task { await model.loadBuiltInPrompts() }
            .confirmationDialog(VoxtL10n.shared.string("Reset all prompts to the built-in ones?"), isPresented: $confirmResetAll) {
                Button(VoxtL10n.shared.string("Reset all to default"), role: .destructive) {
                    model.resetAllCustomPrompts()
                    drafts = [:]
                }
            }
        }
    }
}
