// ModelSettingsComponents.swift
// Provides Model Settings Components for model settings.

import SwiftUI
import AppKit

struct PromptEditorView: View {
    @Binding var text: String
    var height: CGFloat = 100
    var contentPadding: CGFloat = 6
    var variables: [PromptTemplateVariableDescriptor] = []
    var variablesLayout: PromptTemplateVariablesLayout = .adaptive
    var variablesTitle: String? = nil
    var onTextChange: ((String) -> Void)?
    var onFocusChange: ((Bool) -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            PromptTextEditor(
                text: $text,
                onTextChange: onTextChange,
                onFocusChange: onFocusChange
            )
                .settingsPromptEditor(height: height, contentPadding: contentPadding)

            if !variables.isEmpty {
                PromptTemplateVariablesView(
                    title: variablesTitle ?? AppLocalization.localizedString("Supported variables"),
                    variables: variables,
                    layout: variablesLayout
                )
            }
        }
    }
}

enum PromptTextEditorUpdatePolicy {
    static func shouldPublishChange(hasMarkedText: Bool) -> Bool {
        // Marked text is an in-progress input method candidate, not a committed edit.
        !hasMarkedText
    }

    static func shouldApplyExternalText(
        currentText: String,
        externalText: String,
        hasMarkedText: Bool
    ) -> Bool {
        // Reassigning even identical text resets NSTextView's selection. Never
        // replace its storage while an input method owns a marked range either.
        !hasMarkedText && currentText != externalText
    }
}

private struct PromptTextEditor: NSViewRepresentable {
    @Binding var text: String
    let onTextChange: ((String) -> Void)?
    let onFocusChange: ((Bool) -> Void)?

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder

        let textView = NSTextView()
        textView.delegate = context.coordinator
        textView.string = text
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.drawsBackground = false
        textView.font = .monospacedSystemFont(ofSize: 13, weight: .medium)
        textView.textColor = .labelColor
        applyParagraphStyle(to: textView)
        textView.textContainerInset = NSSize(width: 2, height: 4)
        textView.minSize = .zero
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true

        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = scrollView.documentView as? NSTextView else { return }
        guard PromptTextEditorUpdatePolicy.shouldApplyExternalText(
            currentText: textView.string,
            externalText: text,
            hasMarkedText: textView.hasMarkedText()
        ) else {
            return
        }

        let selectedRange = textView.selectedRange()
        textView.string = text
        applyParagraphStyle(to: textView)
        textView.setSelectedRange(clamped(selectedRange, for: text))
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    private func applyParagraphStyle(to textView: NSTextView) {
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineSpacing = 4
        textView.defaultParagraphStyle = paragraphStyle
        textView.typingAttributes[.paragraphStyle] = paragraphStyle
        textView.textStorage?.addAttribute(
            .paragraphStyle,
            value: paragraphStyle,
            range: NSRange(location: 0, length: textView.textStorage?.length ?? 0)
        )
    }

    private func clamped(_ range: NSRange, for text: String) -> NSRange {
        let textLength = (text as NSString).length
        let location = min(range.location, textLength)
        return NSRange(
            location: location,
            length: min(range.length, textLength - location)
        )
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PromptTextEditor

        init(parent: PromptTextEditor) {
            self.parent = parent
        }

        func textDidBeginEditing(_ notification: Notification) {
            parent.onFocusChange?(true)
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            publishCommittedText(from: textView)
        }

        func textDidEndEditing(_ notification: Notification) {
            if let textView = notification.object as? NSTextView {
                publishCommittedText(from: textView)
            }
            parent.onFocusChange?(false)
        }

        private func publishCommittedText(from textView: NSTextView) {
            guard PromptTextEditorUpdatePolicy.shouldPublishChange(
                hasMarkedText: textView.hasMarkedText()
            ) else {
                return
            }

            let newValue = textView.string
            guard parent.text != newValue else { return }
            parent.text = newValue
            parent.onTextChange?(newValue)
        }
    }
}

struct PromptTemplateVariableDescriptor: Identifiable, Hashable {
    let token: String
    let tipKey: String

    var id: String { token }
}

enum PromptTemplateVariablesLayout {
    case adaptive
    case twoColumns
}

struct PromptTemplateVariablesView: View {
    let title: String
    let variables: [PromptTemplateVariableDescriptor]
    let layout: PromptTemplateVariablesLayout

    @State private var copiedToken: String?
    @State private var resetTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)

            LazyVGrid(
                columns: gridColumns,
                alignment: .leading,
                spacing: 8
            ) {
                ForEach(variables) { variable in
                    PromptTemplateVariableChip(
                        variable: variable,
                        isCopied: copiedToken == variable.token,
                        onCopy: { copy(variable.token) }
                    )
                }
            }
        }
    }

    private var gridColumns: [GridItem] {
        switch layout {
        case .adaptive:
            return [GridItem(.adaptive(minimum: 220), alignment: .leading)]
        case .twoColumns:
            return [
                GridItem(.flexible(minimum: 180), alignment: .leading),
                GridItem(.flexible(minimum: 180), alignment: .leading)
            ]
        }
    }

    private func copy(_ token: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(token, forType: .string)
        copiedToken = token
        resetTask?.cancel()
        resetTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled else { return }
            if copiedToken == token {
                copiedToken = nil
            }
            resetTask = nil
        }
    }
}

private struct PromptTemplateVariableChip: View {
    let variable: PromptTemplateVariableDescriptor
    let isCopied: Bool
    let onCopy: () -> Void

    @State private var isShowingTip = false

    var body: some View {
        HStack(spacing: 0) {
            Text(variable.token)
                .font(.caption.monospaced())
                .foregroundStyle(.primary)
                .padding(.horizontal, 10)
                .frame(height: 30)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .onHover { hovering in
                    isShowingTip = hovering
                }
                .popover(isPresented: $isShowingTip, arrowEdge: .bottom) {
                    Text(AppLocalization.localizedString(variable.tipKey))
                        .font(.caption)
                        .foregroundStyle(.primary)
                        .padding(10)
                        .frame(width: 260, alignment: .leading)
                }

            Divider()
                .frame(height: 18)

            Button(isCopied ? AppLocalization.localizedString("Copied") : AppLocalization.localizedString("Copy")) {
                onCopy()
            }
            .buttonStyle(.plain)
            .font(.caption.weight(.semibold))
            .foregroundStyle(isCopied ? Color.green : Color.accentColor)
            .padding(.horizontal, 10)
            .frame(height: 30)
        }
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(SettingsUIStyle.controlFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .stroke(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
    }
}

struct ModelTableAction {
    let title: String
    var role: ButtonRole? = nil
    var isEnabled: Bool = true
    var showsProgress: Bool = false
    let handler: () -> Void
}

struct ModelTableRow: Identifiable {
    let id: String
    let title: String
    let isActive: Bool
    let status: String
    var badgeText: String? = nil
    var isTitleUnderlined: Bool = false
    var onTapTitle: (() -> Void)? = nil
    let actions: [ModelTableAction]
}

struct ModelTableView: View {
    let title: LocalizedStringKey
    let rows: [ModelTableRow]
    var viewportHeight: CGFloat = 280
    @State private var contentHeight: CGFloat = 0

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text(AppLocalization.localizedString("Actions"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .overlay(alignment: .bottom) {
                Divider()
            }

            ScrollView(.vertical) {
                VStack(spacing: 0) {
                    tableRows
                }
                .background {
                    GeometryReader { proxy in
                        Color.clear
                            .preference(key: ModelTableContentHeightPreferenceKey.self, value: proxy.size.height)
                    }
                }
            }
            .frame(height: resolvedViewportHeight)
        }
        .onPreferenceChange(ModelTableContentHeightPreferenceKey.self) { newHeight in
            contentHeight = newHeight
        }
        .tableContainerStyle
    }

    private var resolvedViewportHeight: CGFloat {
        guard contentHeight > 0 else { return viewportHeight }
        return min(contentHeight, viewportHeight)
    }

    @ViewBuilder
    private var tableRows: some View {
        ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
            VStack(spacing: 0) {
                HStack(alignment: .center, spacing: 10) {
                    VStack(alignment: .leading, spacing: 2) {
                        if let onTapTitle = row.onTapTitle {
                            Button(action: onTapTitle) {
                                rowTitle(row)
                            }
                            .buttonStyle(.plain)
                        } else {
                            rowTitle(row)
                        }
                        Text(row.status)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer(minLength: 8)

                    HStack(spacing: 6) {
                        ForEach(Array(row.actions.enumerated()), id: \.offset) { _, action in
                            Button(role: action.role) {
                                action.handler()
                            } label: {
                                ModelActionLabel(
                                    title: action.title,
                                    showsProgress: action.showsProgress
                                )
                            }
                            .buttonStyle(
                                SettingsCompactActionButtonStyle(
                                    tone: action.role == .destructive ? .destructive : .neutral
                                )
                            )
                            .disabled(!action.isEnabled)
                        }
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)

                if index < rows.count - 1 {
                    Divider()
                }
            }
        }
    }

    @ViewBuilder
    private func rowTitle(_ row: ModelTableRow) -> some View {
        HStack(spacing: 6) {
            Text(row.title)
                .font(.subheadline.weight(row.isActive ? .semibold : .regular))
                .underline(row.isTitleUnderlined)
                .foregroundStyle(row.isTitleUnderlined ? Color.accentColor : Color.primary)
            if let badgeText = row.badgeText {
                Text(badgeText)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.orange)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        Capsule()
                            .fill(Color.orange.opacity(0.14))
                    )
                    .overlay(
                        Capsule()
                            .stroke(Color.orange.opacity(0.35), lineWidth: 1)
                    )
            }
        }
    }
}

struct ModelActionLabel: View {
    let title: String
    var showsProgress: Bool = false

    var body: some View {
        HStack(spacing: 6) {
            if showsProgress {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.72)
            }
            Text(title)
        }
    }
}

private struct ModelTableContentHeightPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

private extension View {
    var tableContainerStyle: some View {
        background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(SettingsUIStyle.groupedFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(SettingsUIStyle.panelBorderColor, lineWidth: 1)
        )
    }
}
