// LogsViewerSheet.swift
// Provides Logs Viewer Sheet for settings screens.

import AppKit
import SwiftUI
import UniformTypeIdentifiers

private func logsViewerLocalized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

private struct LogsPreviewTextView: NSViewRepresentable {
    let text: String

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = false

        let textView = NSTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = false
        textView.usesFindBar = true
        textView.isHorizontallyResizable = true
        textView.isVerticallyResizable = true
        textView.autoresizingMask = [.width]
        textView.textContainerInset = NSSize(width: 12, height: 12)
        textView.font = .monospacedSystemFont(ofSize: 11.5, weight: .regular)
        textView.textColor = NSColor.labelColor.withAlphaComponent(0.86)
        textView.string = text
        textView.textContainer?.widthTracksTextView = false
        textView.textContainer?.containerSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.minSize = .zero
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)

        scrollView.documentView = textView
        context.coordinator.textView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = context.coordinator.textView else { return }
        guard textView.string != text else { return }
        textView.string = text
        textView.setSelectedRange(NSRange(location: 0, length: 0))
        scrollView.contentView.scroll(to: .zero)
        scrollView.reflectScrolledClipView(scrollView.contentView)
    }

    final class Coordinator {
        weak var textView: NSTextView?
    }
}

struct LogsViewerSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var latestLogsText = ""
    @State private var isLoadingLogs = false
    @State private var logsStatusMessage = ""
    @State private var toastMessage = ""
    @State private var toastDismissTask: Task<Void, Never>?
    @State private var logLoadTask: Task<Void, Never>?
    @State private var logLoadGeneration = 0
    @State private var isExportingLogs = false
    @State private var logExportDocument = LogExportDocument(text: "")
    @State private var logExportFilename = "voxt-log.txt"

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 10) {
                Text(logsViewerLocalized("Export Logs"))
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.primary)

                Text(logsViewerLocalized("Latest 1000"))
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        Capsule(style: .continuous)
                            .fill(Color.secondary.opacity(0.12))
                    )

                Spacer(minLength: 0)

                Button(logsViewerLocalized("Refresh")) {
                    refreshLogs()
                }
                .buttonStyle(SettingsCompactActionButtonStyle())

                Button(logsViewerLocalized("Copy")) {
                    copyLogs()
                }
                .disabled(isLoadingLogs || latestLogsText.isEmpty)
                .buttonStyle(SettingsCompactActionButtonStyle())

                Button(logsViewerLocalized("Export")) {
                    prepareLogExport()
                }
                .disabled(isLoadingLogs || latestLogsText.isEmpty)
                .buttonStyle(SettingsCompactActionButtonStyle())

                Button(action: dismiss.callAsFunction) {
                    Image(systemName: "xmark")
                }
                .buttonStyle(SettingsDialogCloseButtonStyle())
            }
            .padding(.horizontal, 20)
            .padding(.top, 18)
            .padding(.bottom, 4)

            if !logsStatusMessage.isEmpty {
                Text(logsStatusMessage)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 20)
            }

            ZStack {
                LogsPreviewTextView(text: latestLogsText)
                .opacity(latestLogsText.isEmpty && isLoadingLogs ? 0 : 1)

                if isLoadingLogs {
                    VStack(spacing: 10) {
                        ProgressView()
                            .controlSize(.regular)
                        Text(logsViewerLocalized("Loading logs..."))
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, minHeight: 420, idealHeight: 440, maxHeight: 440)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.black.opacity(0.045))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
            )
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
        .frame(minWidth: 680, idealWidth: 720, maxWidth: 760, minHeight: 500, idealHeight: 540)
        .fileExporter(
            isPresented: $isExportingLogs,
            document: logExportDocument,
            contentType: .plainText,
            defaultFilename: logExportFilename
        ) { result in
            switch result {
            case .success:
                logsStatusMessage = ""
                showToast(logsViewerLocalized("Exported latest 1000 log lines."))
            case .failure(let error):
                logsStatusMessage = AppLocalization.format(
                    "Log export failed: %@",
                    error.localizedDescription
                )
            }
        }
        .overlay(alignment: .top) {
            if !toastMessage.isEmpty {
                ModelDebugToast(message: toastMessage) {
                    dismissToast()
                }
                .padding(.top, 12)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: toastMessage)
        .onAppear {
            refreshLogs()
        }
        .onDisappear {
            logLoadTask?.cancel()
            logLoadTask = nil
            toastDismissTask?.cancel()
        }
    }

    private func refreshLogs() {
        logLoadTask?.cancel()
        logLoadGeneration += 1
        let generation = logLoadGeneration

        isLoadingLogs = true
        logsStatusMessage = ""

        logLoadTask = Task {
            let loadedText = await Task.detached(priority: .userInitiated) {
                VoxtLog.latestLogDisplayText(limit: 1000)
            }.value

            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard generation == logLoadGeneration else { return }
                latestLogsText = loadedText
                isLoadingLogs = false
                logLoadTask = nil
            }
        }
    }

    private func copyLogs() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(latestLogsText, forType: .string)
        logsStatusMessage = ""
        showToast(logsViewerLocalized("Copied to clipboard"))
    }

    private func prepareLogExport() {
        let payload = VoxtLog.latestLogExportPayload(limit: 1000)
        logExportDocument = LogExportDocument(text: payload.content)
        logExportFilename = payload.filename
        isExportingLogs = true
    }

    private func showToast(_ message: String, duration: TimeInterval = 2.2) {
        toastDismissTask?.cancel()
        toastMessage = message
        toastDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(duration))
            guard !Task.isCancelled else { return }
            toastMessage = ""
        }
    }

    private func dismissToast() {
        toastDismissTask?.cancel()
        toastMessage = ""
    }
}
