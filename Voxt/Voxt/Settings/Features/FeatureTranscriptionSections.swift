// FeatureTranscriptionSections.swift
// Provides Feature Transcription Sections for feature settings.

import SwiftUI

extension FeatureSettingsView {
    var transcriptionContent: some View {
        featurePage(
            title: featureSettingsLocalized("Transcription"),
            subtitle: featureSettingsLocalized("Choose a speech model, then add text enhancement if needed."),
            iconKind: .transcription,
            pills: transcriptionPills,
            showsHeroHeader: false
        ) {
            FeatureSettingsCard(title: "") {
                FeatureSettingSection(title: "", detail: "") {
                    FeatureSelectorRow(
                        title: featureSettingsLocalized("Speech Model"),
                        value: asrSelectionSummary(featureSettings.transcription.asrSelectionID),
                        action: { selectorSheet = .transcriptionASR }
                    )
                }

                FeatureToggleRow(
                    title: featureSettingsLocalized("Text Enhancement"),
                    detail: "",
                    isOn: transcriptionLLMEnabledBinding
                )

                if featureSettings.transcription.llmEnabled {
                    FeatureSettingSection(title: "", detail: "") {
                        FeatureSelectorRow(
                            title: featureSettingsLocalized("Enhancement Model"),
                            value: llmSelectionSummary(featureSettings.transcription.llmSelectionID),
                            action: { selectorSheet = .transcriptionLLM }
                        )
                        FeaturePromptSection(
                            title: featureSettingsLocalized("Enhancement Prompt"),
                            text: promptBinding(
                                get: { featureSettings.transcription.prompt },
                                set: { featureSettings.transcription.prompt = $0 },
                                kind: .enhancement
                            ),
                            defaultText: AppPromptDefaults.text(for: .enhancement),
                            kind: .enhancement,
                            selectedPresetID: Binding(
                                get: { featureSettings.transcription.promptPresetID },
                                set: { featureSettings.transcription.promptPresetID = $0 }
                            ),
                            variables: ModelSettingsPromptVariables.enhancement,
                            guidance: "",
                            persistChanges: { prompt in
                                FeatureSettingsStore.saveTranscriptionPrompt(
                                    prompt,
                                    presetID: featureSettings.transcription.promptPresetID
                                )
                            },
                            persistPresetSelection: { presetID, prompt in
                                featureSettings.transcription.promptPresetID = presetID
                                featureSettings.transcription.prompt = prompt
                                FeatureSettingsStore.saveTranscriptionPrompt(prompt, presetID: presetID)
                            }
                        )
                    }
                }

            }
        }
    }

    var noteContent: some View {
        featurePage(
            title: featureSettingsLocalized("Notes"),
            subtitle: featureSettingsLocalized("Capture key points during recording. Notes stay separate and get short AI titles."),
            iconKind: .note,
            pills: notePills,
            showsHeroHeader: false
        ) {
            FeatureSettingsCard(title: "") {
                if !noteStore.isAvailable || noteStore.lastRecoveryArchiveURL != nil {
                    VoxtNoteStorageRecoveryView(store: noteStore)
                }

                FeatureSettingSection(title: "", detail: "") {
                    FeatureSelectorRow(
                        title: featureSettingsLocalized("Title Model"),
                        value: llmSelectionSummary(featureSettings.transcription.notes.titleModelSelectionID),
                        action: { selectorSheet = .transcriptionNoteTitle }
                    )
                }

                FeatureSettingSection(title: "", detail: "") {
                    VStack(alignment: .leading, spacing: 16) {
                        VoxtNotePanelSectionHeader(
                            title: featureSettingsLocalized("Floating Panel"),
                            detail: featureSettingsLocalized("Reveal your notes by resting the pointer in a screen corner."),
                            cornerTitle: featureSettings.transcription.notes.panel.corner.title
                        )

                        VoxtNotePanelCornerPicker(
                            selection: binding(
                                get: { featureSettings.transcription.notes.panel.corner },
                                set: { featureSettings.transcription.notes.panel.corner = $0 }
                            )
                        )

                        FeatureToggleRow(
                            title: featureSettingsLocalized("Translucent Panel"),
                            detail: "",
                            isOn: binding(
                                get: { featureSettings.transcription.notes.panel.isTranslucent },
                                set: { featureSettings.transcription.notes.panel.isTranslucent = $0 }
                            )
                        )

                        VoxtNotePanelDelayRow(
                            title: featureSettingsLocalized("Reveal Delay"),
                            value: binding(
                                get: { featureSettings.transcription.notes.panel.revealDelay },
                                set: { featureSettings.transcription.notes.panel.revealDelay = $0 }
                            ),
                            range: 0.2...2.0
                        )

                        VoxtNotePanelDelayRow(
                            title: featureSettingsLocalized("Hide Delay"),
                            value: binding(
                                get: { featureSettings.transcription.notes.panel.hideDelay },
                                set: { featureSettings.transcription.notes.panel.hideDelay = $0 }
                            ),
                            range: 0.1...2.0
                        )
                    }
                }

                Divider()

                FeatureSettingSection(
                    title: "",
                    detail: ""
                ) {
                    noteObsidianSyncSection
                }

                FeatureSettingSection(
                    title: "",
                    detail: ""
                ) {
                    noteRemindersSyncSection
                }
            }
        }
    }
}

private struct VoxtNoteStorageRecoveryView: View {
    @ObservedObject var store: VoxtNoteStore
    @State private var confirmsArchiveAndRebuild = false

    var body: some View {
        FeatureSettingSection(
            title: featureSettingsLocalized("Note Storage"),
            detail: featureSettingsLocalized("Voxt keeps note data in its own local application storage.")
        ) {
            VStack(alignment: .leading, spacing: 12) {
                if store.isAvailable {
                    Label(
                        featureSettingsLocalized("Note storage is ready."),
                        systemImage: "checkmark.circle.fill"
                    )
                    .foregroundStyle(.green)

                    if let archiveURL = store.lastRecoveryArchiveURL {
                        Text(
                            String(
                                format: featureSettingsLocalized("The previous database was archived at %@"),
                                archiveURL.path
                            )
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    }
                } else {
                    Label(
                        featureSettingsLocalized("Note storage is unavailable."),
                        systemImage: "exclamationmark.triangle.fill"
                    )
                    .foregroundStyle(.orange)

                    if let message = store.availability.errorMessage, !message.isEmpty {
                        Text(message)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }

                    HStack(spacing: 10) {
                        Button(featureSettingsLocalized("Retry")) {
                            store.retryOpeningStorage()
                        }

                        Button(
                            featureSettingsLocalized("Archive and Rebuild"),
                            role: .destructive
                        ) {
                            confirmsArchiveAndRebuild = true
                        }
                    }
                }
            }
        }
        .confirmationDialog(
            featureSettingsLocalized("Archive the current note database and create a new one?"),
            isPresented: $confirmsArchiveAndRebuild,
            titleVisibility: .visible
        ) {
            Button(featureSettingsLocalized("Archive and Rebuild"), role: .destructive) {
                store.archiveAndRebuildStorage()
            }
            Button(featureSettingsLocalized("Cancel"), role: .cancel) {}
        } message: {
            Text(featureSettingsLocalized("The existing database will be preserved in a recovery folder. Notes in it will not appear in the new database."))
        }
    }
}

private struct VoxtNotePanelSectionHeader: View {
    let title: String
    let detail: String
    let cornerTitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.body.weight(.semibold))
                .foregroundStyle(.primary.opacity(0.92))

            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(detail)
                    .foregroundStyle(.secondary)

                Text(featureSettingsLocalized("Trigger Corner"))
                    .foregroundStyle(.secondary)

                Text(cornerTitle)
                    .fontWeight(.semibold)
                    .foregroundStyle(.primary.opacity(0.86))
            }
            .font(.caption)
        }
    }
}

private struct VoxtNotePanelCornerPicker: View {
    @Binding var selection: VoxtNotePanelCorner

    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                Image("OverlayPreviewBackground")
                    .resizable()
                    .scaledToFill()

                Color.black.opacity(0.10)

                Image(systemName: "display")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white.opacity(0.90))
                    .frame(width: 48, height: 36)
                    .background(.black.opacity(0.34), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

            }
            .frame(width: 360, height: 142)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(.white.opacity(0.20), lineWidth: 1)
            }
            .overlay(alignment: .topLeading) {
                cornerButton(.topLeft)
                    .padding(10)
            }
            .overlay(alignment: .topTrailing) {
                cornerButton(.topRight)
                    .padding(10)
            }
            .overlay(alignment: .bottomLeading) {
                cornerButton(.bottomLeft)
                    .padding(10)
            }
            .overlay(alignment: .bottomTrailing) {
                cornerButton(.bottomRight)
                    .padding(10)
            }
            .shadow(color: .black.opacity(0.10), radius: 10, y: 4)

            Text(featureSettingsLocalized("The same corner works on every connected display."))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .center)
    }

    private func cornerButton(_ corner: VoxtNotePanelCorner) -> some View {
        Button { selection = corner } label: {
            Circle()
                .fill(selection == corner ? Color.accentColor : Color.white.opacity(0.76))
                .frame(width: 19, height: 19)
                .overlay {
                    Circle()
                        .stroke(
                            selection == corner ? Color.white.opacity(0.92) : Color.black.opacity(0.10),
                            lineWidth: selection == corner ? 3 : 1
                        )
                }
                .shadow(color: .black.opacity(0.16), radius: 3, y: 1)
                .frame(width: 30, height: 30)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .help(corner.title)
        .accessibilityLabel(corner.title)
        .accessibilityAddTraits(selection == corner ? .isSelected : [])
    }
}

private struct VoxtNotePanelDelayRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>

    private var clampedValue: Binding<Double> {
        Binding(
            get: { normalized(value) },
            set: { value = normalized($0) }
        )
    }

    var body: some View {
        HStack(alignment: .center, spacing: 24) {
            Text(title)
                .font(.body.weight(.semibold))
                .foregroundStyle(.primary.opacity(0.92))
                .fixedSize(horizontal: true, vertical: false)

            Spacer(minLength: 24)

            HStack(spacing: 8) {
                VoxtNotePanelDelayScale(
                    title: title,
                    value: clampedValue,
                    range: range,
                    step: 0.1
                )
                .frame(maxWidth: .infinity)
                .frame(height: 32)

                TextField(
                    "",
                    value: clampedValue,
                    format: .number.precision(.fractionLength(1))
                )
                .font(.system(size: 12, weight: .semibold))
                .monospacedDigit()
                .multilineTextAlignment(.center)
                .textFieldStyle(.plain)
                .settingsFieldSurface(
                    width: 54,
                    minHeight: 32,
                    horizontalPadding: 8,
                    alignment: .center
                )
                .accessibilityLabel(title)

                Text(featureSettingsLocalized("sec"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 22, alignment: .leading)
            }
            .frame(width: 300, alignment: .trailing)
        }
        .onAppear { value = normalized(value) }
    }

    private func normalized(_ candidate: Double) -> Double {
        let clamped = min(max(candidate, range.lowerBound), range.upperBound)
        return (clamped * 10).rounded() / 10
    }
}

private struct VoxtNotePanelDelayScale: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double

    @State private var isHovering = false
    @State private var isDragging = false

    private let tickCount = 9

    var body: some View {
        GeometryReader { geometry in
            let horizontalInset: CGFloat = 14
            let availableWidth = max(geometry.size.width - horizontalInset * 2, 1)
            let thumbX = horizontalInset + availableWidth * progress

            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(SettingsUIStyle.subtleFillColor)

                Capsule(style: .continuous)
                    .fill(Color.accentColor.opacity(isActive ? 0.10 : 0.06))
                    .frame(width: max(0, thumbX - 5), height: 24)
                    .offset(x: 4)

                tickMarks(width: availableWidth)
                    .offset(x: horizontalInset)

                Capsule(style: .continuous)
                    .fill(isActive ? Color.accentColor : Color.primary.opacity(0.72))
                    .frame(width: 4, height: 21)
                    .shadow(color: Color.black.opacity(isActive ? 0.12 : 0.06), radius: 2, y: 1)
                    .offset(x: thumbX - 2)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(
                        isActive ? Color.accentColor.opacity(0.42) : SettingsUIStyle.subtleBorderColor,
                        lineWidth: 1
                    )
            }
            .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { gesture in
                        isDragging = true
                        updateValue(
                            at: gesture.location.x,
                            horizontalInset: horizontalInset,
                            availableWidth: availableWidth
                        )
                    }
                    .onEnded { _ in isDragging = false }
            )
            .onHover { isHovering = $0 }
        }
        .accessibilityElement()
        .accessibilityLabel(title)
        .accessibilityValue(value.formatted(.number.precision(.fractionLength(1))))
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment:
                value = normalized(value + step)
            case .decrement:
                value = normalized(value - step)
            @unknown default:
                break
            }
        }
    }

    private var isActive: Bool {
        isHovering || isDragging
    }

    private var progress: CGFloat {
        let span = max(range.upperBound - range.lowerBound, step)
        return CGFloat((normalized(value) - range.lowerBound) / span)
    }

    private func tickMarks(width: CGFloat) -> some View {
        GeometryReader { geometry in
            ForEach(0..<tickCount, id: \.self) { index in
                let tickProgress = CGFloat(index) / CGFloat(tickCount - 1)
                Capsule(style: .continuous)
                    .fill(Color.primary.opacity(index == 0 || index == tickCount - 1 ? 0.18 : 0.13))
                    .frame(width: 1, height: index == 0 || index == tickCount - 1 ? 11 : 8)
                    .position(
                        x: 0.5 + max(geometry.size.width - 1, 0) * tickProgress,
                        y: geometry.size.height / 2
                    )
            }
        }
        .frame(width: width, height: 12)
        .allowsHitTesting(false)
    }

    private func updateValue(
        at locationX: CGFloat,
        horizontalInset: CGFloat,
        availableWidth: CGFloat
    ) {
        let localX = min(max(locationX - horizontalInset, 0), availableWidth)
        let rawValue = range.lowerBound
            + Double(localX / availableWidth) * (range.upperBound - range.lowerBound)
        value = normalized(rawValue)
    }

    private func normalized(_ candidate: Double) -> Double {
        let clamped = min(max(candidate, range.lowerBound), range.upperBound)
        let steps = ((clamped - range.lowerBound) / step).rounded()
        return min(max(range.lowerBound + steps * step, range.lowerBound), range.upperBound)
    }
}
