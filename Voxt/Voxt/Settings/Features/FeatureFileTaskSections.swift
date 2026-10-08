// FeatureFileTaskSections.swift
// Provides the Custom > Files page and persistent meeting file task UI.

import AppKit
import SwiftUI
import UniformTypeIdentifiers

extension FeatureSettingsView {
    var filesContent: some View {
        VStack(alignment: .leading, spacing: 18) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if meetingFileTaskQueue.tasks.isEmpty {
                        MeetingFileTasksEmptyState()
                    } else {
                        ForEach(meetingFileTaskQueue.tasks) { task in
                            MeetingFileTaskRow(
                                task: task,
                                now: Date(),
                                onCancel: {
                                    meetingFileTaskQueue.cancel(taskID: task.id)
                                },
                                onPrioritize: {
                                    meetingFileTaskQueue.prioritize(taskID: task.id)
                                },
                                onRetry: {
                                    meetingFileTaskQueue.retry(taskID: task.id)
                                },
                                onOpenDetails: {
                                    AppDelegate.shared?.showMeetingFileTaskDetail(taskID: task.id)
                                },
                                onRemove: {
                                    meetingFileTaskQueue.removeFinishedTask(taskID: task.id)
                                },
                                onViewTranscript: {
                                    Task { @MainActor in
                                        await AppDelegate.shared?.showMeetingFileTranscript(taskID: task.id)
                                    }
                                }
                            )
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 1)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            MeetingFileUploadCard(
                onChooseFiles: chooseMeetingFiles,
                onImportDroppedFiles: importDroppedMeetingFiles
            )
        }
        // The settings shell owns the outer insets. Unlike a fully scrolling
        // feature page, this split layout must not inset the pinned upload card
        // again or reserve a scroll-indicator gutter for the entire page.
        .padding(.top, 2)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func chooseMeetingFiles() {
        let panel = NSOpenPanel()
        panel.title = featureSettingsLocalized("Analyze Meeting File")
        panel.prompt = featureSettingsLocalized("Choose File")
        panel.message = featureSettingsLocalized("Choose audio or video recordings to transcribe and analyze.")
        panel.allowedContentTypes = MeetingFileImportSupport.allowedContentTypes
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true

        guard panel.runModal() == .OK else { return }
        enqueueMeetingFiles(panel.urls)
    }

    private func importDroppedMeetingFiles(_ urls: [URL]) {
        enqueueMeetingFiles(urls)
    }

    private func enqueueMeetingFiles(_ urls: [URL]) {
        let validURLs = urls.filter { MeetingFileImportSupport.isSupportedImportFile(at: $0) }
        let invalidCount = urls.count - validURLs.count
        if invalidCount > 0 {
            showMeetingFileImportToast(
                featureSettingsLocalized("Unsupported file type. Drop an audio or video recording instead.")
            )
        }
        guard !validURLs.isEmpty else { return }
        SystemNotificationSupport.requestAuthorizationIfNeeded()
        meetingFileTaskQueue.enqueue(urls: validURLs)
    }

    private func showMeetingFileImportToast(_ message: String) {
        NotificationCenter.default.post(
            name: .voxtFeatureSettingsToastRequested,
            object: nil,
            userInfo: ["message": message]
        )
    }
}

private struct MeetingFileTasksEmptyState: View {
    var body: some View {
        VStack(spacing: 9) {
            Image(systemName: "tray")
                .font(.system(size: 22, weight: .medium))
                .foregroundStyle(Color.accentColor)
            Text(featureSettingsLocalized("No file tasks yet"))
                .font(.subheadline.weight(.semibold))
            Text(featureSettingsLocalized("Upload a recording below to create a meeting analysis task."))
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 34)
        .settingsPanelSurface()
    }
}

private struct MeetingFileTaskRow: View {
    private let fileNameColumnWidth: CGFloat = 360

    let task: MeetingFileTask
    let now: Date
    let onCancel: () -> Void
    let onPrioritize: () -> Void
    let onRetry: () -> Void
    let onOpenDetails: () -> Void
    let onRemove: () -> Void
    let onViewTranscript: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(task.fileName)
                            .font(.subheadline.weight(.semibold))
                            .frame(width: fileNameColumnWidth, alignment: .leading)
                            .lineLimit(1)
                            .truncationMode(.middle)

                        Text(statusText)
                            .font(.caption)
                            .foregroundStyle(statusColor)
                            .lineLimit(2)
                            .truncationMode(.tail)
                            .help(statusText)
                    }

                    timeMetadata
                        .padding(.top, 5)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                VStack(alignment: .trailing, spacing: 6) {
                    actionButtons
                    if task.progressStage == .identifyingSpeakers, task.status != .completed {
                        Button(featureSettingsLocalized("View Transcription"), action: onViewTranscript)
                            .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 27))
                    }
                }
            }

            if task.status == .preparing || task.status == .processing || task.status == .waitingForResources || task.status == .cancelling {
                ProgressView(value: task.progressFraction, total: 1)
                    .progressViewStyle(.linear)
                    .tint(statusColor)
                    .padding(.top, 5)
                    .accessibilityLabel(Text(featureSettingsLocalized("File conversion progress")))
                    .accessibilityValue(Text("\(Int((task.progressFraction * 100).rounded()))%"))
            }
        }
        .padding(14)
        .settingsPanelSurface(cornerRadius: SettingsUIStyle.compactCornerRadius)
    }

    @ViewBuilder
    private var actionButtons: some View {
        switch task.status {
        case .preparing, .processing, .waitingForResources, .cancelling:
            Button(featureSettingsLocalized(task.status == .cancelling ? "Cancelling…" : "Cancel"), action: onCancel)
                .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 27))
                .disabled(task.status == .cancelling)
        case .completed:
            HStack(spacing: 6) {
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "trash")
                        .font(.system(size: 12, weight: .medium))
                }
                .buttonStyle(SettingsPillButtonStyle(tone: .destructive, horizontalPadding: 8, height: 27))
                .help(featureSettingsLocalized("Delete"))
                .accessibilityLabel(Text(featureSettingsLocalized("Delete")))

                Button(featureSettingsLocalized("Details"), action: onOpenDetails)
                    .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 27))
            }
        case .failed, .cancelled:
            Button(featureSettingsLocalized("Retry"), action: onRetry)
                .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 27))
        case .queued:
            HStack(spacing: 6) {
                Button(featureSettingsLocalized("Prioritize"), action: onPrioritize)
                    .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 8, height: 27))
                Button(featureSettingsLocalized("Cancel"), action: onCancel)
                    .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 8, height: 27))
            }
        }
    }

    private func metadata(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.tertiary)
            Text(value)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .fixedSize(horizontal: true, vertical: false)
    }

    private var timeMetadata: some View {
        HStack(alignment: .firstTextBaseline, spacing: 14) {
            metadata(label: featureSettingsLocalized("Created"), value: createdTimeText)
            metadata(label: featureSettingsLocalized("Elapsed"), value: durationText(task.elapsedSeconds(now: now)))
            metadata(label: featureSettingsLocalized("Total Duration"), value: totalDurationText)
            if task.status != .completed {
                metadata(label: featureSettingsLocalized("Estimated Remaining"), value: estimatedRemainingText)
            }
        }
        .fixedSize(horizontal: true, vertical: false)
    }

    private var createdTimeText: String {
        let formatter = DateFormatter()
        formatter.locale = AppLocalization.locale
        formatter.timeStyle = .short
        formatter.dateStyle = .none
        return formatter.string(from: task.enqueuedAt)
    }

    private var totalDurationText: String {
        guard let duration = task.mediaDurationSeconds else { return "—" }
        return durationText(duration)
    }

    private var estimatedRemainingText: String {
        if let remaining = task.estimatedRemainingSeconds(now: now) {
            return durationText(remaining)
        }
        if task.status == .queued {
            return featureSettingsLocalized("Waiting")
        }
        return "—"
    }

    private func durationText(_ duration: TimeInterval) -> String {
        let totalSeconds = max(0, Int(duration.rounded()))
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        }
        return String(format: "%02d:%02d", minutes, seconds)
    }

    private var statusText: String {
        switch task.status {
        case .queued:
            return featureSettingsLocalized("Waiting")
        case .preparing:
            return featureSettingsLocalized("Preparing audio…")
        case .processing:
            return task.progressStage.displayTitle
        case .waitingForResources:
            return featureSettingsLocalized("Waiting for system resources…")
        case .cancelling:
            return featureSettingsLocalized("Cancelling…")
        case .completed:
            return featureSettingsLocalized("Completed")
        case .failed:
            let heading = featureSettingsLocalized("Failed") + " · " + task.progressStage.displayTitle
            guard let errorMessage = task.errorMessage,
                  !errorMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return heading
            }
            return heading + " · " + errorMessage
        case .cancelled:
            return featureSettingsLocalized("Cancelled")
        }
    }

    private var statusColor: Color {
        switch task.status {
        case .queued:
            return .secondary
        case .preparing, .processing, .waitingForResources, .cancelling:
            return .accentColor
        case .completed:
            return .green
        case .failed:
            return .orange
        case .cancelled:
            return .secondary
        }
    }
}

private struct MeetingFileUploadCard: View {
    let onChooseFiles: () -> Void
    let onImportDroppedFiles: ([URL]) -> Void
    @State private var isDropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(featureSettingsLocalized("Analyze a Meeting Recording"))
                        .font(.headline.weight(.semibold))
                    Text(featureSettingsLocalized("Upload a recording to create a transcript, identify speakers, and open the full meeting analysis."))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 12)
                Button(featureSettingsLocalized("Upload File"), action: onChooseFiles)
                    .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 13, height: 30))
            }

            Button(action: onChooseFiles) {
                VStack(spacing: 9) {
                    Image(systemName: isDropTargeted ? "arrow.down.doc" : "arrow.up.doc")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(Color.accentColor)
                    Text(featureSettingsLocalized(isDropTargeted ? "Drop to analyze this recording" : "Click to choose or drag in a meeting recording"))
                        .font(.subheadline.weight(.medium))
                    Text(featureSettingsLocalized("Supports MP3, MP4, M4A, WAV, AAC, MOV, and other common audio or video formats."))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity)
                .frame(minHeight: 96)
                .padding(.horizontal, 18)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                        .fill(isDropTargeted ? Color.accentColor.opacity(0.08) : SettingsUIStyle.groupedFillColor.opacity(0.55))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                        .stroke(
                            isDropTargeted ? Color.accentColor.opacity(0.72) : SettingsUIStyle.controlHoverBorderColor,
                            style: StrokeStyle(lineWidth: isDropTargeted ? 1.5 : 1, dash: [6, 5])
                        )
                )
                .contentShape(RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous))
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .settingsPanelSurface()
        .onDrop(of: [UTType.fileURL], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        let fileProviders = providers.filter {
            $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier)
        }
        guard !fileProviders.isEmpty else { return false }

        for provider in fileProviders {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                guard let fileURL = MeetingFileImportSupport.fileURL(fromDropItem: item) else {
                    return
                }
                Task { @MainActor in
                    onImportDroppedFiles([fileURL])
                }
            }
        }
        return true
    }
}
