// ModelSettingsDownloadComponents.swift
// Provides Model Settings Download Components for model settings.

import SwiftUI

private func modelSettingsDownloadLocalized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct ModelDownloadSettingsSheet: View {
    let modelStorageDisplayPath: String
    let modelStorageFallbackPath: String
    let modelStorageSelectionError: String?
    let onOpenModelStorageInFinder: () -> Void
    let onChooseModelStorageDirectory: () -> Void
    @Binding var localModelIdleUnloadDelaySeconds: Int
    @Binding var showIdleUnloadDelayInfo: Bool
    @Binding var isPresented: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(modelSettingsDownloadLocalized("Model Download Settings"))
                .font(.title3.weight(.semibold))

            GeneralSettingsCard(titleText: modelSettingsDownloadLocalized("Model Storage")) {
                SettingsPathSelectionRow(
                    title: modelSettingsDownloadLocalized("Storage Path"),
                    displayedPath: modelStorageDisplayPath,
                    fallbackPath: modelStorageFallbackPath,
                    openButtonHelp: modelSettingsDownloadLocalized("Open folder"),
                    chooseButtonTitle: modelSettingsDownloadLocalized("Choose"),
                    onOpen: onOpenModelStorageInFinder,
                    onChoose: onChooseModelStorageDirectory
                )

                if let modelStorageSelectionError, !modelStorageSelectionError.isEmpty {
                    Text(modelStorageSelectionError)
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                Text(modelSettingsDownloadLocalized("Model downloads automatically choose the fastest available source when installation starts."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            GeneralSettingsCard(titleText: modelSettingsDownloadLocalized("Memory")) {
                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 6) {
                        Text(modelSettingsDownloadLocalized("Unload Delay"))
                            .foregroundStyle(.secondary)

                        Button {
                            showIdleUnloadDelayInfo.toggle()
                        } label: {
                            Image(systemName: "info.circle")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .popover(isPresented: $showIdleUnloadDelayInfo, arrowEdge: .top) {
                            Text(modelSettingsDownloadLocalized("Idle unload delay for local ASR and local LLM models. Lower values reduce memory usage; higher values favor faster reuse."))
                                .font(.caption)
                                .foregroundStyle(.primary)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .frame(width: 280, alignment: .leading)
                        }

                        Spacer(minLength: 0)
                    }

                    LocalModelIdleUnloadDelayControl(
                        value: $localModelIdleUnloadDelaySeconds,
                        range: AppPreferenceKey.localModelIdleUnloadDelayMinimumSeconds...AppPreferenceKey.localModelIdleUnloadDelayMaximumSeconds
                    )
                }
            }

            SettingsDialogActionRow {
                Button(modelSettingsDownloadLocalized("Done")) {
                    isPresented = false
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: 560, onClose: {
            isPresented = false
        })
    }
}

private struct LocalModelIdleUnloadDelayControl: View {
    @Binding var value: Int
    let range: ClosedRange<Int>

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            LocalModelIdleUnloadDelaySlider(
                value: $value,
                range: range
            )
            .frame(maxWidth: .infinity)
            .frame(height: 38)

            HStack(alignment: .firstTextBaseline) {
                Text(formatIdleUnloadDelay(range.lowerBound))
                    .foregroundStyle(.secondary)

                Spacer()

                Text(formatIdleUnloadDelay(range.upperBound))
                    .foregroundStyle(.secondary)
            }
            .font(.caption2)
            .monospacedDigit()
        }
    }
}

private struct LocalModelIdleUnloadDelaySlider: View {
    @Binding var value: Int
    let range: ClosedRange<Int>

    @State private var isHovering = false
    @State private var isDragging = false

    var body: some View {
        GeometryReader { geometry in
            let progress = normalizedProgress
            let thumbX = max(0, min(geometry.size.width, geometry.size.width * progress))

            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(SettingsUIStyle.subtleFillColor)

                tickMarks
                    .padding(.horizontal, 18)

                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(Color.primary.opacity(isActive ? 0.10 : 0.08))
                    .frame(width: max(20, thumbX - 12), height: 30)
                    .offset(x: 4)

                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(activeTint.opacity(isActive ? 0.92 : 0.60))
                    .frame(width: 8, height: 26)
                    .overlay(
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .strokeBorder(activeTint.opacity(isActive ? 0.22 : 0.12), lineWidth: 1)
                    )
                    .offset(x: min(max(0, thumbX - 4), max(0, geometry.size.width - 8)))

                HStack {
                    Spacer()
                    Text(formatIdleUnloadDelay(value))
                        .font(.system(size: 12, weight: .semibold))
                        .monospacedDigit()
                        .foregroundStyle(isActive ? Color.accentColor : Color.secondary)
                        .padding(.trailing, 12)
                }
            }
            .frame(height: 38)
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(
                        isActive ? Color.accentColor.opacity(0.45) : SettingsUIStyle.subtleBorderColor,
                        lineWidth: 1
                    )
            )
            .shadow(color: isActive ? Color.accentColor.opacity(0.10) : .clear, radius: 6, y: 1)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { gesture in
                        isDragging = true
                        updateValue(at: gesture.location.x, width: geometry.size.width)
                    }
                    .onEnded { _ in
                        isDragging = false
                    }
            )
            .onHover { isHovering = $0 }
        }
    }

    private var isActive: Bool {
        isHovering || isDragging
    }

    private var activeTint: Color {
        isActive ? Color.accentColor : Color.primary
    }

    private var normalizedProgress: CGFloat {
        let clamped = min(max(value, range.lowerBound), range.upperBound)
        let span = max(range.upperBound - range.lowerBound, 1)
        return CGFloat(clamped - range.lowerBound) / CGFloat(span)
    }

    @ViewBuilder
    private var tickMarks: some View {
        GeometryReader { geometry in
            HStack(spacing: 0) {
                ForEach(0..<9, id: \.self) { index in
                    Circle()
                        .fill(Color.primary.opacity(index == 0 ? 0.18 : 0.14))
                        .frame(width: 4.5, height: 4.5)
                        .frame(maxWidth: .infinity, alignment: index == 0 ? .leading : index == 8 ? .trailing : .center)
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .allowsHitTesting(false)
    }

    private func updateValue(at locationX: CGFloat, width: CGFloat) {
        let resolvedWidth = max(width, 1)
        let clampedX = min(max(locationX, 0), resolvedWidth)
        let progress = clampedX / resolvedWidth
        let resolved = range.lowerBound + Int(round(progress * CGFloat(range.upperBound - range.lowerBound)))
        value = min(max(resolved, range.lowerBound), range.upperBound)
    }
}

private func formatIdleUnloadDelay(_ seconds: Int) -> String {
    guard seconds >= 60 else {
        return AppLocalization.format("%@s", String(seconds))
    }

    let minutes = Double(seconds) / 60.0
    let hasFraction = seconds % 60 != 0
    let formattedMinutes = minutes.formatted(
        .number.precision(.fractionLength(hasFraction && minutes < 10 ? 1 : 0))
    )
    return AppLocalization.format("%@min", formattedMinutes)
}
