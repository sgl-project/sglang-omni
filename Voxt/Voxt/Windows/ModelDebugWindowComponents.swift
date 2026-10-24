// ModelDebugWindowComponents.swift
// Provides Model Debug Window Components for window and overlay UI.

import AppKit
import SwiftUI
import AVFoundation
import Combine

struct ModelDebugWindowBackground: View {
    var body: some View {
        RoundedRectangle(cornerRadius: DetailPanelUIStyle.windowCornerRadius, style: .continuous)
            .fill(DetailPanelUIStyle.windowFillColor)
            .overlay(
                RoundedRectangle(cornerRadius: DetailPanelUIStyle.windowCornerRadius, style: .continuous)
                    .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
            )
            .ignoresSafeArea()
    }
}

struct ModelDebugToolbarSelectorLabel: View {
    let title: String
    let value: String

    var body: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 8)
            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .frame(minWidth: ModelDebugWindowStyle.selectorMinWidth, idealWidth: ModelDebugWindowStyle.selectorIdealWidth, maxWidth: .infinity, minHeight: 44, maxHeight: 44)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(DetailPanelUIStyle.controlFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
        )
    }
}

struct ModelDebugHeaderBadge: View {
    var body: some View {
        Text(modelDebugLocalized("Debug"))
            .font(.system(size: 9, weight: .semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 7)
            .frame(height: 18)
            .background(
                Capsule(style: .continuous)
                    .fill(DetailPanelUIStyle.controlFillColor)
            )
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
            )
            .frame(width: 62, alignment: .leading)
            .offset(x: 1)
    }
}

struct ModelDebugToast: View {
    let message: String
    let onClose: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Text(message)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.primary)
                .lineLimit(2)
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .semibold))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .frame(height: 34)
        .background(
            Capsule(style: .continuous)
                .fill(DetailPanelUIStyle.controlFillColor)
        )
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.12), radius: 10, y: 4)
    }
}

struct ASRDebugAudioSelectorSheet: View {
    @Environment(\.dismiss) private var dismiss

    let clips: [ASRDebugClipItem]
    let selectedClipID: UUID?
    let onSelect: (UUID) -> Void
    let onDelete: (UUID) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text(modelDebugLocalized("Recorded Audio"))
                    .font(.title3.weight(.semibold))
            }

            if clips.isEmpty {
                debugEmptyState(
                    title: modelDebugLocalized("No audio clips yet"),
                    detail: modelDebugLocalized("Record audio here, then reuse it across models.")
                )
            } else {
                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(clips) { clip in
                            HStack(alignment: .center, spacing: 12) {
                                Button {
                                    onSelect(clip.id)
                                    dismiss()
                                } label: {
                                    HStack(alignment: .center, spacing: 12) {
                                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                                            Text(clip.displayTitle)
                                                .font(.system(size: 13, weight: .semibold))
                                                .foregroundStyle(.primary)
                                                .lineLimit(1)
                                            Text(clip.clip.summaryText)
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                                .lineLimit(1)
                                        }
                                        Spacer()
                                        if clip.id == selectedClipID {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundStyle(Color.accentColor)
                                        }
                                    }
                                    .padding(12)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(
                                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                                            .fill(DetailPanelUIStyle.controlFillColor)
                                    )
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                                            .strokeBorder(
                                                clip.id == selectedClipID ? Color.accentColor.opacity(0.35) : DetailPanelUIStyle.borderColor,
                                                lineWidth: 1
                                            )
                                    )
                                }
                                .buttonStyle(.plain)

                                Button {
                                    onDelete(clip.id)
                                } label: {
                                    Image(systemName: "trash")
                                        .font(.system(size: 13, weight: .semibold))
                                        .frame(width: 40, height: 40)
                                        .background(
                                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                                .fill(DetailPanelUIStyle.controlFillColor)
                                        )
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                                .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
                                        )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }
            }
        }
        .settingsDialogChrome(width: 580, height: 470, onClose: { dismiss() })
    }
}

struct LLMDebugPresetSelectorSheet: View {
    @Environment(\.dismiss) private var dismiss

    let presets: [LLMDebugPresetOption]
    let selectedPresetID: String
    let onSelect: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text(modelDebugLocalized("Choose Preset"))
                    .font(.title3.weight(.semibold))
            }

            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(presets) { preset in
                        Button {
                            onSelect(preset.id)
                            dismiss()
                        } label: {
                            HStack(alignment: .center, spacing: 12) {
                                HStack(alignment: .firstTextBaseline, spacing: 8) {
                                    Text(preset.title)
                                        .font(.system(size: 13, weight: .semibold))
                                        .foregroundStyle(.primary)
                                        .lineLimit(1)
                                    Text(preset.subtitle)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                                Spacer()
                                if preset.id == selectedPresetID {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(Color.accentColor)
                                }
                            }
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .fill(DetailPanelUIStyle.controlFillColor)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(
                                        preset.id == selectedPresetID ? Color.accentColor.opacity(0.35) : DetailPanelUIStyle.borderColor,
                                        lineWidth: 1
                                    )
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
        .settingsDialogChrome(width: 520, height: 470, onClose: { dismiss() })
    }
}

struct LLMDebugPromptSettingsSheet: View {
    @Environment(\.dismiss) private var dismiss

    let preset: LLMDebugPresetOption
    @Binding var variableValues: [String: String]
    let onApply: (String) -> Void
    let onSave: (String) -> Void

    @State private var draftPrompt = ""

    private var isCustomPreset: Bool {
        if case .custom = preset.kind {
            return true
        }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(modelDebugLocalized("Preset Settings"))
                        .font(.title3.weight(.semibold))
                    Text(preset.title)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                HStack(spacing: 8) {
                    if !isCustomPreset {
                        Button(modelDebugLocalized("Apply")) {
                            onApply(draftPrompt)
                        }
                        .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10))
                    }
                    Button(modelDebugLocalized("Save")) {
                        onSave(draftPrompt)
                        dismiss()
                    }
                    .buttonStyle(DetailPrimaryButtonStyle())
                }
                .padding(.trailing, 56)
            }

            ScrollView {
                Group {
                    if isCustomPreset {
                        HStack(alignment: .top, spacing: 16) {
                            VStack(alignment: .leading, spacing: 8) {
                                Text(modelDebugLocalized("Prompt"))
                                    .font(.subheadline.weight(.medium))
                                PromptEditorView(
                                    text: $draftPrompt,
                                    height: 310,
                                    variables: []
                                )
                            }
                            Spacer(minLength: 0)
                        }
                    } else {
                        HStack(alignment: .top, spacing: 16) {
                            VStack(alignment: .leading, spacing: 8) {
                                Text(modelDebugLocalized("Runtime Input"))
                                    .font(.subheadline.weight(.medium))
                                LLMDebugVariableEditor(
                                    descriptors: preset.runtimeInputDescriptors,
                                    values: $variableValues
                                )
                                Spacer(minLength: 0)
                            }
                            .frame(width: 210, alignment: .topLeading)

                            VStack(alignment: .leading, spacing: 8) {
                                Text(modelDebugLocalized("Prompt"))
                                    .font(.subheadline.weight(.medium))
                                PromptEditorView(
                                    text: $draftPrompt,
                                    height: 310,
                                    variables: preset.variables,
                                    variablesTitle: modelDebugLocalized("Prompt variables")
                                )
                            }
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .settingsDialogChrome(width: 720, height: 450, onClose: { dismiss() })
        .onAppear {
            draftPrompt = preset.promptTemplate
        }
    }
}

struct LLMDebugVariableEditor: View {
    let descriptors: [PromptTemplateVariableDescriptor]
    @Binding var values: [String: String]

    private static let multilineTokens: Set<String> = [
        "{{DICTATED_PROMPT}}",
        "{{SOURCE_TEXT}}",
        AppDelegate.rawTranscriptionTemplateVariable,
        TranscriptSummarySupport.transcriptRecordTemplateVariable
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(descriptors, id: \.id) { variable in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .center, spacing: 8) {
                            Text(variable.token)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                            Spacer(minLength: 8)
                            Button(modelDebugLocalized("Copy")) {
                                copy(variable.token)
                            }
                            .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 8, height: 26))
                        }

                        if isMultiline(variable) {
                            TextEditor(text: binding(for: variable))
                                .font(.system(size: 13))
                                .scrollContentBackground(.hidden)
                                .padding(8)
                                .frame(minHeight: 96, maxHeight: 120, alignment: .topLeading)
                                .background(
                                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                                        .fill(DetailPanelUIStyle.panelFillColor)
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                                        .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
                                )
                        } else {
                            TextField(
                                AppLocalization.localizedString(variable.tipKey),
                                text: binding(for: variable)
                            )
                            .textFieldStyle(.plain)
                            .font(.system(size: 13))
                            .padding(.horizontal, 10)
                            .frame(height: 32)
                            .background(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .fill(DetailPanelUIStyle.panelFillColor)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
                            )
                        }
                    }
                }
            }
            .padding(.vertical, 2)
        }
    }

    private func binding(for variable: PromptTemplateVariableDescriptor) -> Binding<String> {
        Binding(
            get: { values[variable.token, default: ""] },
            set: { values[variable.token] = $0 }
        )
    }

    private func isMultiline(_ variable: PromptTemplateVariableDescriptor) -> Bool {
        Self.multilineTokens.contains(variable.token)
    }

    private func copy(_ value: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(value, forType: .string)
    }
}

struct ASRDebugVADSnapshotView: View {
    let snapshot: VADDebugSnapshot

    private var gateText: String {
        switch snapshot.localGatePolicy {
        case .enabled:
            return modelDebugLocalized("Enabled")
        case .disabled(let reason):
            return AppLocalization.format("Disabled: %@", reason)
        }
    }

    private var frameVADText: String {
        snapshot.providerUsesServerVAD
            ? modelDebugLocalized("Server VAD")
            : snapshot.frameBackend.title
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "waveform")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 20)
                ASRDebugVADBadge(
                    title: modelDebugLocalized("VAD Mode"),
                    value: snapshot.mode.title
                )
                ASRDebugVADBadge(
                    title: modelDebugLocalized("Frame VAD"),
                    value: frameVADText
                )
                ASRDebugVADBadge(
                    title: modelDebugLocalized("Local Gate"),
                    value: gateText
                )
                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(DetailPanelUIStyle.controlFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
        )
    }
}

private struct ASRDebugVADBadge: View {
    let title: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }
}

struct ASRDebugResultCard: View {
    let result: ASRDebugResult
    let onClose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(result.modelTitle)
                        .font(.headline)
                        .lineLimit(1)
                    Text(result.clipTitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark")
                }
                .buttonStyle(SettingsDialogCloseButtonStyle())
            }

            ScrollView {
                Text(result.isError ? (result.errorText ?? "") : (result.outputText.isEmpty ? modelDebugLocalized("No output.") : result.outputText))
                    .font(.body)
                    .foregroundStyle(result.isError ? .red : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(height: ModelDebugWindowStyle.resultCardBodyHeight)

            HStack(spacing: 10) {
                Text(AppLocalization.format("%@ audio", result.audioDurationText))
                Text(AppLocalization.format("%@ run", result.runtimeText))
                Text(AppLocalization.format("%d chars", result.characterCount))
                Spacer()
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(DetailPanelUIStyle.panelFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(result.isError ? Color.red.opacity(0.22) : DetailPanelUIStyle.borderColor, lineWidth: 1)
        )
    }
}

struct LLMDebugResultCard: View {
    let result: LLMDebugResult
    let onClose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(result.modelTitle)
                        .font(.headline)
                        .lineLimit(1)
                    Text(result.presetTitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark")
                }
                .buttonStyle(SettingsDialogCloseButtonStyle())
            }

            if !result.inputSummary.isEmpty {
                Text(result.inputSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }

            if let metadata = result.requestMetadata {
                LLMDebugRequestMetadataView(metadata: metadata)
            }

            ScrollView {
                Text(result.isError ? (result.errorText ?? "") : (result.outputText.isEmpty ? modelDebugLocalized("No output.") : result.outputText))
                    .font(.body)
                    .foregroundStyle(result.isError ? .red : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(height: ModelDebugWindowStyle.resultCardBodyHeight)

            HStack(spacing: 10) {
                Text(result.durationText)
                Spacer()
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(DetailPanelUIStyle.panelFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(result.isError ? Color.red.opacity(0.22) : DetailPanelUIStyle.borderColor, lineWidth: 1)
        )
    }
}

private struct LLMDebugRequestMetadataView: View {
    let metadata: LLMDebugRequestMetadata

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !metadata.spokenInstruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                LLMDebugMetadataRow(
                    title: modelDebugLocalized("Spoken Instruction"),
                    value: metadata.spokenInstruction
                )
            }

            if !metadata.sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                LLMDebugMetadataRow(
                    title: modelDebugLocalized("Selected Source Text"),
                    value: metadata.sourceText
                )
            }

        }
    }
}

private struct LLMDebugMetadataRow: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption)
                .lineLimit(2)
                .foregroundStyle(.primary)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

func modelDebugConfigureWindowChrome(_ window: NSWindow) {
    window.isMovableByWindowBackground = false
    window.isOpaque = false
    window.backgroundColor = .clear
    window.hasShadow = true
}

func modelDebugPositionWindowTrafficLightButtons(_ window: NSWindow) {
    guard let closeButton = window.standardWindowButton(.closeButton),
          let miniaturizeButton = window.standardWindowButton(.miniaturizeButton),
          let zoomButton = window.standardWindowButton(.zoomButton),
          let container = closeButton.superview
    else {
        return
    }

    let leftInset: CGFloat = 15
    let topInset: CGFloat = 17
    let spacing: CGFloat = 6

    let buttonSize = closeButton.frame.size
    let y = container.bounds.height - topInset - buttonSize.height
    let closeX = leftInset
    let miniaturizeX = closeX + buttonSize.width + spacing
    let zoomX = miniaturizeX + buttonSize.width + spacing

    closeButton.translatesAutoresizingMaskIntoConstraints = true
    miniaturizeButton.translatesAutoresizingMaskIntoConstraints = true
    zoomButton.translatesAutoresizingMaskIntoConstraints = true

    closeButton.setFrameOrigin(CGPoint(x: closeX, y: y))
    miniaturizeButton.setFrameOrigin(CGPoint(x: miniaturizeX, y: y))
    zoomButton.setFrameOrigin(CGPoint(x: zoomX, y: y))
}

func modelDebugScheduleTrafficLightButtonPositionUpdate(for window: NSWindow) {
    DispatchQueue.main.async { [weak window] in
        guard let window else { return }
        modelDebugPositionWindowTrafficLightButtons(window)
    }
}

@ViewBuilder
func debugEmptyState(title: String, detail: String) -> some View {
    VStack(spacing: 8) {
        Text(title)
            .font(.headline)
        Text(detail)
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
    }
    .frame(maxWidth: .infinity)
    .padding(.vertical, 48)
}
