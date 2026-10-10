// WaveformAnswerCardComponents.swift
// Provides Waveform Answer Card Components for window and overlay UI.

import SwiftUI
import AppKit

struct RewriteConversationBottomVisibilityPreferenceKey: PreferenceKey {
    static var defaultValue = true

    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = nextValue()
    }
}

enum OverlayTranslationMenuStyle {
    case answer
    case compact

    var height: CGFloat {
        switch self {
        case .answer:
            return 24
        case .compact:
            return 22
        }
    }

    var horizontalPadding: CGFloat {
        switch self {
        case .answer:
            return 10
        case .compact:
            return 6
        }
    }

    var textFontSize: CGFloat {
        switch self {
        case .answer:
            return 11
        case .compact:
            return 10
        }
    }

    var indicatorFontSize: CGFloat {
        switch self {
        case .answer:
            return 9
        case .compact:
            return 9
        }
    }

    var titleMaxWidth: CGFloat {
        switch self {
        case .answer:
            return 92
        case .compact:
            return 56
        }
    }

    var minWidth: CGFloat {
        switch self {
        case .answer:
            return 0
        case .compact:
            return WaveformView.defaultSessionLanguagePickerWidth
        }
    }

    var fixedWidth: CGFloat? {
        switch self {
        case .answer:
            return nil
        case .compact:
            return WaveformView.defaultSessionLanguagePickerWidth
        }
    }

    var backgroundColor: NSColor {
        NSColor.white.withAlphaComponent(self == .answer ? 0.08 : 0.09)
    }

    var borderColor: NSColor {
        NSColor.white.withAlphaComponent(self == .answer ? 0.12 : 0.14)
    }

    var hoverBorderColor: NSColor {
        NSColor.controlAccentColor.withAlphaComponent(0.28)
    }

    var textColor: NSColor {
        NSColor.white.withAlphaComponent(0.92)
    }

    var indicatorColor: NSColor {
        NSColor.white.withAlphaComponent(0.72)
    }
}

struct AnswerSessionTranslationMenuPicker: View {
    let selectedLanguage: TranslationTargetLanguage?
    let isPresented: Bool
    let onTogglePresentation: () -> Void
    let onDismissPresentation: () -> Void
    let onSelectLanguage: (TranslationTargetLanguage) -> Void
    var style: OverlayTranslationMenuStyle = .answer
    var fixedWidth: CGFloat?

    var body: some View {
        OverlayTranslationMenuPickerRepresentable(
            selectedLanguage: selectedLanguage,
            isPresented: isPresented,
            style: style,
            onTogglePresentation: onTogglePresentation,
            onDismissPresentation: onDismissPresentation,
            onSelectLanguage: onSelectLanguage
        )
        .frame(width: fixedWidth ?? style.fixedWidth, height: style.height)
        .fixedSize(horizontal: true, vertical: false)
        .accessibilityLabel(Text(AppLocalization.localizedString("Target Language")))
    }
}

private struct OverlayTranslationMenuPickerRepresentable: NSViewRepresentable {
    let selectedLanguage: TranslationTargetLanguage?
    let isPresented: Bool
    let style: OverlayTranslationMenuStyle
    let onTogglePresentation: () -> Void
    let onDismissPresentation: () -> Void
    let onSelectLanguage: (TranslationTargetLanguage) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(
            selectedLanguage: selectedLanguage,
            onTogglePresentation: onTogglePresentation,
            onDismissPresentation: onDismissPresentation,
            onSelectLanguage: onSelectLanguage
        )
    }

    func makeNSView(context: Context) -> OverlayTranslationMenuHostView {
        let view = OverlayTranslationMenuHostView()
        view.onSelectLanguage = { [weak coordinator = context.coordinator] language in
            coordinator?.select(language)
        }
        view.onTogglePresentation = { [weak coordinator = context.coordinator] in
            coordinator?.togglePresentation()
        }
        view.onDismissPresentation = { [weak coordinator = context.coordinator] in
            coordinator?.dismissPresentation()
        }
        return view
    }

    func updateNSView(_ nsView: OverlayTranslationMenuHostView, context: Context) {
        context.coordinator.selectedLanguage = selectedLanguage
        context.coordinator.isPresented = isPresented
        nsView.onSelectLanguage = { [weak coordinator = context.coordinator] language in
            coordinator?.select(language)
        }
        nsView.onTogglePresentation = { [weak coordinator = context.coordinator] in
            coordinator?.togglePresentation()
        }
        nsView.onDismissPresentation = { [weak coordinator = context.coordinator] in
            coordinator?.dismissPresentation()
        }
        nsView.update(
            selectedLanguage: selectedLanguage,
            isPresented: isPresented,
            style: style
        )
    }

    final class Coordinator {
        var selectedLanguage: TranslationTargetLanguage?
        var isPresented: Bool
        let onTogglePresentation: () -> Void
        let onDismissPresentation: () -> Void
        let onSelectLanguage: (TranslationTargetLanguage) -> Void

        init(
            selectedLanguage: TranslationTargetLanguage?,
            onTogglePresentation: @escaping () -> Void,
            onDismissPresentation: @escaping () -> Void,
            onSelectLanguage: @escaping (TranslationTargetLanguage) -> Void
        ) {
            self.selectedLanguage = selectedLanguage
            self.isPresented = false
            self.onTogglePresentation = onTogglePresentation
            self.onDismissPresentation = onDismissPresentation
            self.onSelectLanguage = onSelectLanguage
        }

        func togglePresentation() {
            onTogglePresentation()
        }

        func dismissPresentation() {
            isPresented = false
            onDismissPresentation()
        }

        func select(_ language: TranslationTargetLanguage) {
            guard selectedLanguage != language else { return }
            selectedLanguage = language
            onSelectLanguage(language)
        }
    }
}

private final class OverlayTranslationMenuHostView: NSView, NSMenuDelegate {
    private let titleField = NSTextField(labelWithString: "")
    private let indicatorView = NSImageView()
    private let popupMenu = NSMenu()
    private var trackingArea: NSTrackingArea?
    private var currentStyle: OverlayTranslationMenuStyle = .answer
    private var selectedLanguage: TranslationTargetLanguage?
    private var isPresented = false
    private var isHovered = false
    private var isMenuOpen = false
    private var isMenuPresentationScheduled = false
    private var suppressPresentationUntilStateReset = false
    private var didRequestDismissForCurrentPresentation = false
    private var localMouseDownMonitor: Any?
    private var titleLeadingConstraint: NSLayoutConstraint?
    private var titleMaxWidthConstraint: NSLayoutConstraint?
    private var indicatorTrailingConstraint: NSLayoutConstraint?
    private var indicatorWidthConstraint: NSLayoutConstraint?
    var onSelectLanguage: ((TranslationTargetLanguage) -> Void)?
    var onTogglePresentation: (() -> Void)?
    var onDismissPresentation: (() -> Void)?

    override var isFlipped: Bool { true }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        popupMenu.delegate = self
        popupMenu.autoenablesItems = false
        popupMenu.showsStateColumn = true

        titleField.translatesAutoresizingMaskIntoConstraints = false
        titleField.lineBreakMode = .byTruncatingTail
        titleField.maximumNumberOfLines = 1
        titleField.alignment = .center
        titleField.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        indicatorView.translatesAutoresizingMaskIntoConstraints = false
        indicatorView.imageScaling = .scaleProportionallyDown
        indicatorView.setContentCompressionResistancePriority(.required, for: .horizontal)
        indicatorView.setContentHuggingPriority(.required, for: .horizontal)

        addSubview(titleField)
        addSubview(indicatorView)

        let titleLeadingConstraint = titleField.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8)
        let titleMaxWidthConstraint = titleField.widthAnchor.constraint(lessThanOrEqualToConstant: 92)
        let indicatorTrailingConstraint = indicatorView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8)
        let indicatorWidthConstraint = indicatorView.widthAnchor.constraint(equalToConstant: 13)
        self.titleLeadingConstraint = titleLeadingConstraint
        self.titleMaxWidthConstraint = titleMaxWidthConstraint
        self.indicatorTrailingConstraint = indicatorTrailingConstraint
        self.indicatorWidthConstraint = indicatorWidthConstraint

        NSLayoutConstraint.activate([
            titleLeadingConstraint,
            titleField.centerYAnchor.constraint(equalTo: centerYAnchor),
            titleField.trailingAnchor.constraint(equalTo: indicatorView.leadingAnchor, constant: -6),
            titleMaxWidthConstraint,
            indicatorTrailingConstraint,
            indicatorWidthConstraint,
            indicatorView.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var intrinsicContentSize: NSSize {
        let title = titleField.stringValue as NSString
        let titleWidth = min(
            ceil(title.size(withAttributes: [.font: titleField.font as Any]).width),
            currentStyle.titleMaxWidth
        )
        let indicatorWidth = currentStyle.indicatorFontSize + 4
        let width = currentStyle.horizontalPadding +
            titleWidth +
            6 +
            indicatorWidth +
            currentStyle.horizontalPadding
        return NSSize(width: max(width, currentStyle.minWidth), height: currentStyle.height)
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let trackingArea {
            removeTrackingArea(trackingArea)
        }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        trackingArea = area
        addTrackingArea(area)
    }

    override func mouseEntered(with event: NSEvent) {
        isHovered = true
        updateAppearance()
    }

    override func mouseExited(with event: NSEvent) {
        isHovered = false
        updateAppearance()
    }

    override func mouseDown(with event: NSEvent) {
        guard !popupMenu.items.isEmpty else { return }
        onTogglePresentation?()
    }

    deinit {
        removeLocalMouseDownMonitor()
    }

    func update(
        selectedLanguage: TranslationTargetLanguage?,
        isPresented: Bool,
        style: OverlayTranslationMenuStyle
    ) {
        self.selectedLanguage = selectedLanguage
        self.isPresented = isPresented
        if !isPresented {
            suppressPresentationUntilStateReset = false
            didRequestDismissForCurrentPresentation = false
        }
        currentStyle = style
        rebuildMenu()
        titleField.stringValue = selectedLanguage?.title ?? ""
        titleField.font = .systemFont(ofSize: style.textFontSize, weight: .semibold)
        titleField.textColor = style.textColor
        titleLeadingConstraint?.constant = style.horizontalPadding
        titleMaxWidthConstraint?.constant = style.titleMaxWidth
        indicatorTrailingConstraint?.constant = -style.horizontalPadding
        indicatorWidthConstraint?.constant = style.indicatorFontSize + 4
        indicatorView.image = NSImage(
            systemSymbolName: "chevron.down",
            accessibilityDescription: nil
        )?.withSymbolConfiguration(.init(pointSize: style.indicatorFontSize, weight: .bold))
        indicatorView.contentTintColor = style.indicatorColor
        DispatchQueue.main.async { [weak self] in
            self?.invalidateIntrinsicContentSize()
            self?.needsLayout = true
        }
        updateAppearance()

        if isPresented {
            presentMenuIfNeeded()
        } else if isMenuOpen || isMenuPresentationScheduled {
            isMenuPresentationScheduled = false
            popupMenu.cancelTracking()
        }
    }

    override func layout() {
        super.layout()
        layer?.cornerRadius = bounds.height / 2
    }

    private func updateAppearance() {
        guard let layer else { return }
        layer.backgroundColor = currentStyle.backgroundColor.cgColor
        layer.borderColor = (isHovered ? currentStyle.hoverBorderColor : currentStyle.borderColor).cgColor
        layer.borderWidth = 1
    }

    private func presentMenuIfNeeded() {
        guard isPresented,
              !suppressPresentationUntilStateReset,
              !isMenuOpen,
              !isMenuPresentationScheduled,
              !popupMenu.items.isEmpty
        else {
            return
        }
        isMenuPresentationScheduled = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.isMenuPresentationScheduled = false
            guard self.isPresented,
                  !self.suppressPresentationUntilStateReset,
                  !self.isMenuOpen,
                  self.window != nil
            else {
                return
            }
            let selectedItem = self.selectedLanguage.flatMap { language in
                self.popupMenu.items.first(where: { $0.representedObject as? String == language.rawValue })
            }
            _ = self.popupMenu.popUp(
                positioning: selectedItem,
                at: NSPoint(x: 0, y: self.bounds.height + 8),
                in: self
            )
        }
    }

    private func rebuildMenu() {
        popupMenu.removeAllItems()
        for language in TranslationTargetLanguage.allCases {
            let item = NSMenuItem(title: language.title, action: #selector(selectMenuItem(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = language.rawValue
            item.state = language == selectedLanguage ? .on : .off
            popupMenu.addItem(item)
        }
    }

    @objc
    private func selectMenuItem(_ sender: NSMenuItem) {
        guard
            let rawValue = sender.representedObject as? String,
            let language = TranslationTargetLanguage(rawValue: rawValue)
        else {
            return
        }
        onSelectLanguage?(language)
    }

    func menuWillOpen(_ menu: NSMenu) {
        guard menu === popupMenu else { return }
        isMenuOpen = true
        didRequestDismissForCurrentPresentation = false
        installLocalMouseDownMonitor()
    }

    func menuDidClose(_ menu: NSMenu) {
        guard menu === popupMenu else { return }
        isMenuOpen = false
        removeLocalMouseDownMonitor()
        requestDismissPresentation(cancelMenu: false)
    }

    private func installLocalMouseDownMonitor() {
        guard localMouseDownMonitor == nil else { return }
        localMouseDownMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]
        ) { [weak self] event in
            self?.dismissMenuForOverlayClickIfNeeded(event)
            return event
        }
    }

    private func removeLocalMouseDownMonitor() {
        if let localMouseDownMonitor {
            NSEvent.removeMonitor(localMouseDownMonitor)
            self.localMouseDownMonitor = nil
        }
    }

    private func dismissMenuForOverlayClickIfNeeded(_ event: NSEvent) {
        guard isMenuOpen, event.window === window else { return }
        let location = convert(event.locationInWindow, from: nil)
        guard !bounds.contains(location) else { return }
        requestDismissPresentation(cancelMenu: true)
    }

    private func requestDismissPresentation(cancelMenu: Bool) {
        isPresented = false
        isMenuPresentationScheduled = false
        suppressPresentationUntilStateReset = true
        if !didRequestDismissForCurrentPresentation {
            didRequestDismissForCurrentPresentation = true
            onDismissPresentation?()
        }
        if cancelMenu, isMenuOpen {
            popupMenu.cancelTracking()
        }
    }
}

struct AnswerConversationBodyView: View {
    private let conversationBottomAnchorID = "rewrite-conversation-bottom-anchor"

    let conversationTurns: [RewriteConversationTurn]
    let streamingUserPromptText: String?
    let streamingDraftPayload: RewriteAnswerPayload?
    let isProcessing: Bool

    @State private var isScrolledToConversationBottom = true
    @State private var wasScrolledToConversationBottom = true
    @State private var hasUnreadConversationMessages = false
    @State private var pendingScrollRequestToken = UUID()

    var body: some View {
        GeometryReader { outerProxy in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottomTrailing) {
                    ScrollView(.vertical, showsIndicators: true) {
                        LazyVStack(alignment: .leading, spacing: 14) {
                            ForEach(conversationTurns) { turn in
                                if !turn.userPromptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                    HStack {
                                        Spacer(minLength: 48)
                                        conversationBubble(
                                            title: AppLocalization.localizedString("You"),
                                            content: turn.userPromptText,
                                            alignment: .trailing,
                                            isUser: true
                                        )
                                    }
                                }

                                HStack {
                                    conversationBubble(
                                        title: assistantPayload(for: turn).title,
                                        content: assistantPayload(for: turn).content,
                                        alignment: .leading,
                                        isUser: false
                                    )
                                    Spacer(minLength: 48)
                                }
                                .id(turn.id)
                            }

                            if let streamingUserPromptText,
                               !streamingUserPromptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                HStack {
                                    Spacer(minLength: 48)
                                    conversationBubble(
                                        title: AppLocalization.localizedString("You"),
                                        content: streamingUserPromptText,
                                        alignment: .trailing,
                                        isUser: true,
                                        isStreaming: true
                                    )
                                }
                                .id("streaming-user-draft")
                            }

                            if let streamingDraftPayload,
                               !streamingDraftPayload.trimmedTitle.isEmpty || !streamingDraftPayload.trimmedContent.isEmpty || isProcessing {
                                HStack {
                                    conversationBubble(
                                        title: streamingDraftPayload.title,
                                        content: streamingDraftPayload.trimmedContent.isEmpty ? "…" : streamingDraftPayload.content,
                                        alignment: .leading,
                                        isUser: false,
                                        isStreaming: true
                                    )
                                    Spacer(minLength: 48)
                                }
                                .id("streaming-draft")
                            } else if isProcessing {
                                HStack {
                                    conversationBubble(
                                        title: "",
                                        content: "…",
                                        alignment: .leading,
                                        isUser: false,
                                        isStreaming: true
                                    )
                                    Spacer(minLength: 48)
                                }
                                .id("streaming-placeholder")
                            }

                            GeometryReader { geo in
                                Color.clear
                                    .preference(
                                        key: RewriteConversationBottomVisibilityPreferenceKey.self,
                                        value: abs(geo.frame(in: .named("RewriteConversationScroll")).maxY - outerProxy.size.height) < 36
                                    )
                            }
                            .frame(height: 1)
                            .id(conversationBottomAnchorID)
                        }
                        .padding(.trailing, 10)
                    }
                    .coordinateSpace(name: "RewriteConversationScroll")
                    .frame(maxWidth: .infinity, maxHeight: 220, alignment: .topLeading)
                    .onPreferenceChange(RewriteConversationBottomVisibilityPreferenceKey.self) { isVisible in
                        wasScrolledToConversationBottom = isScrolledToConversationBottom
                        isScrolledToConversationBottom = isVisible
                        if isVisible {
                            hasUnreadConversationMessages = false
                        }
                    }
                    .onAppear {
                        scrollConversationToBottom(using: proxy)
                    }
                    .onChange(of: conversationTurns.count) { oldValue, newValue in
                        guard newValue > oldValue else { return }
                        handleConversationMessagesUpdate(using: proxy, animated: true)
                    }
                    .onChange(of: isProcessing) { oldValue, newValue in
                        guard newValue, newValue != oldValue else { return }
                        handleConversationMessagesUpdate(using: proxy, forceScroll: true, animated: true)
                    }
                    .onChange(of: streamingDraftPayload?.content ?? "") { _, _ in
                        handleConversationMessagesUpdate(using: proxy, animated: false)
                    }
                    .onChange(of: streamingDraftPayload?.trimmedTitle ?? "") { oldValue, newValue in
                        guard oldValue != newValue else { return }
                        handleConversationMessagesUpdate(
                            using: proxy,
                            forceScroll: !newValue.isEmpty && oldValue.isEmpty,
                            animated: oldValue.isEmpty && !newValue.isEmpty
                        )
                    }
                    .onChange(of: streamingUserPromptText ?? "") { oldValue, newValue in
                        guard oldValue != newValue else { return }
                        handleConversationMessagesUpdate(
                            using: proxy,
                            forceScroll: !newValue.isEmpty && oldValue.isEmpty,
                            animated: oldValue.isEmpty && !newValue.isEmpty
                        )
                    }

                    if hasUnreadConversationMessages {
                        Button {
                            hasUnreadConversationMessages = false
                            scrollConversationToBottom(using: proxy)
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.down")
                                    .font(.system(size: 10, weight: .semibold))
                                Text(AppLocalization.localizedString("Latest"))
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .foregroundStyle(.white.opacity(0.92))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(.black.opacity(0.78))
                            )
                            .overlay(
                                Capsule(style: .continuous)
                                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                        .padding(.trailing, 8)
                        .padding(.bottom, 4)
                    }
                }
            }
        }
    }

    private func handleConversationMessagesUpdate(
        using proxy: ScrollViewProxy,
        forceScroll: Bool = false,
        animated: Bool = false
    ) {
        if forceScroll || isScrolledToConversationBottom || wasScrolledToConversationBottom {
            hasUnreadConversationMessages = false
            scrollConversationToBottom(using: proxy, animated: animated)
        } else {
            hasUnreadConversationMessages = true
        }
    }

    private func scrollConversationToBottom(using proxy: ScrollViewProxy, animated: Bool = true) {
        let token = UUID()
        pendingScrollRequestToken = token
        DispatchQueue.main.async {
            guard token == pendingScrollRequestToken else { return }
            if animated {
                withAnimation(.easeOut(duration: 0.18)) {
                    proxy.scrollTo(conversationBottomAnchorID, anchor: .bottom)
                }
            } else {
                proxy.scrollTo(conversationBottomAnchorID, anchor: .bottom)
            }
        }
    }

    private func assistantPayload(for turn: RewriteConversationTurn) -> RewriteAnswerPayload {
        let rawPayload = RewriteAnswerPayload(
            title: turn.resultTitle,
            content: turn.resultContent
        )
        if rawPayload.trimmedTitle.isEmpty { return rawPayload }
        return RewriteAnswerPayloadParser.normalize(rawPayload)
    }

    private func conversationBubble(
        title: String,
        content: String,
        alignment: Alignment,
        isUser: Bool,
        isStreaming: Bool = false
    ) -> some View {
        RewriteConversationBubble(
            title: title,
            content: content,
            alignment: alignment,
            isUser: isUser,
            isStreaming: isStreaming
        )
    }
}

struct AnswerHeaderActionButton<Label: View>: View {
    let accessibilityLabel: String
    let action: () -> Void
    let isEnabled: Bool
    @ViewBuilder let label: () -> Label

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            label()
                .frame(width: 24, height: 24)
                .opacity(isEnabled ? 1 : 0.42)
                .background(
                    Circle()
                        .fill(isEnabled ? (isHovered ? .white.opacity(0.16) : .white.opacity(0.08)) : .white.opacity(0.04))
                )
                .overlay(
                    Circle()
                        .strokeBorder(.white.opacity(isEnabled && isHovered ? 0.18 : 0), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .accessibilityLabel(Text(accessibilityLabel))
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }
}

struct RewriteConversationBubble: View {
    let title: String
    let content: String
    let alignment: Alignment
    let isUser: Bool
    let isStreaming: Bool

    @State private var isHovered = false
    @State private var didCopy = false
    @State private var copyFeedbackToken = UUID()

    private var resolvedTitle: String {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? AppLocalization.localizedString("AI Answer") : trimmed
    }

    private var copyText: String {
        let trimmedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedContent.isEmpty else { return "" }
        guard !isUser, !trimmedTitle.isEmpty else { return trimmedContent }
        return "\(trimmedTitle)\n\n\(trimmedContent)"
    }

    var body: some View {
        VStack(alignment: isUser ? .trailing : .leading, spacing: 7) {
            if isUser || !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(resolvedTitle)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white.opacity(isUser ? 0.7 : (isStreaming ? 0.54 : 0.64)))
            }

            Group {
                if isStreaming {
                    Text(content)
                } else {
                    Text(content)
                        .textSelection(.enabled)
                }
            }
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(.white.opacity(isStreaming ? 0.84 : 0.92))
            .frame(maxWidth: .infinity, alignment: alignment)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: 340, alignment: alignment)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(isUser ? .white.opacity(0.1) : .white.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isUser ? .white.opacity(0.12) : .white.opacity(0.08), lineWidth: 1)
        )
        .overlay(alignment: .topTrailing) {
            if !isUser && isHovered {
                Button(action: copyToPasteboard) {
                    HStack(spacing: 4) {
                        Group {
                            if didCopy {
                                CopySuccessIconView()
                            } else {
                                CopyIconView()
                            }
                        }
                        .frame(width: 12, height: 12)

                        Text(didCopy ? AppLocalization.localizedString("Copied") : AppLocalization.localizedString("Copy"))
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .foregroundStyle(.white.opacity(0.94))
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(
                        Capsule(style: .continuous)
                            .fill(.black.opacity(0.74))
                    )
                    .overlay(
                        Capsule(style: .continuous)
                            .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
                .padding(.top, 8)
                .padding(.trailing, 8)
                .transition(.opacity.combined(with: .scale(scale: 0.94, anchor: .topTrailing)))
            }
        }
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }

    private func copyToPasteboard() {
        guard !copyText.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(copyText, forType: .string)
        copyFeedbackToken = UUID()
        let token = copyFeedbackToken
        didCopy = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            guard token == copyFeedbackToken else { return }
            didCopy = false
        }
    }
}

struct AnswerContinueButton: View {
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Text(AppLocalization.localizedString("Continue"))
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.94))
                .padding(.horizontal, 10)
                .frame(height: 24)
                .background(
                    Capsule()
                        .fill(isHovered ? .white.opacity(0.16) : .white.opacity(0.08))
                )
                .overlay(
                    Capsule()
                        .strokeBorder(.white.opacity(isHovered ? 0.18 : 0.1), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text(AppLocalization.localizedString("Continue")))
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }
}

struct AnswerConversationWaveView: View {
    let isRecording: Bool
    let isProcessing: Bool
    let audioLevel: Float
    let shouldAnimate: Bool

    @StateObject private var waveformState = RecentAudioWaveformState(
        barCount: 16,
        historyDuration: 0.9,
        framesPerSecond: 20,
        silenceFloor: 0.01,
        peakHoldFrames: 1,
        peakDecayFactor: 0.74,
        riseSmoothing: 0.82,
        fallSmoothing: 0.24
    )

    var body: some View {
        SessionMiniWaveform(
            waveformState: waveformState,
            isSubdued: !isRecording || isProcessing
        )
        .scaleEffect(x: 0.84, y: 0.82, anchor: .leading)
        .onAppear {
            waveformState.setActive(shouldAnimate && isRecording && !isProcessing)
        }
        .onChange(of: shouldAnimate) {
            waveformState.setActive(shouldAnimate && isRecording && !isProcessing)
        }
        .onChange(of: isRecording) {
            waveformState.setActive(shouldAnimate && isRecording && !isProcessing)
        }
        .onChange(of: isProcessing) {
            waveformState.setActive(shouldAnimate && isRecording && !isProcessing)
        }
        .onChange(of: audioLevel) {
            waveformState.ingest(level: emphasizedWaveformInputLevel(audioLevel))
        }
        .onDisappear {
            waveformState.setActive(false)
        }
    }

    private func emphasizedWaveformInputLevel(_ level: Float) -> Float {
        let clamped = max(0, min(level, 1))
        let expanded = min(1.0, pow(Double(clamped), 0.72) * 1.24)
        return Float(expanded)
    }
}

private struct SessionMiniWaveform: View {
    @ObservedObject var waveformState: RecentAudioWaveformState
    var isSubdued = false

    var body: some View {
        HStack(alignment: .center, spacing: 2.5) {
            ForEach(0..<waveformState.barCount, id: \.self) { index in
                RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                    .fill(WaveformBarVisuals.barGradient)
                    .frame(width: 4, height: barHeight(for: index))
                    .shadow(color: .white.opacity(glowOpacity(for: index)), radius: 2.5, x: 0, y: 0)
            }
        }
        .frame(height: 28, alignment: .center)
    }

    private func barHeight(for index: Int) -> CGFloat {
        if isSubdued {
            let quietPattern: [CGFloat] = [3.2, 3.9, 4.6, 5.1, 4.2, 3.5, 4.4, 4.9]
            return quietPattern[index % quietPattern.count]
        }
        let baseLevel = waveformState.barLevels.indices.contains(index) ? waveformState.barLevels[index] : 0
        return WaveformBarVisuals.barHeight(
            level: baseLevel,
            minHeight: 2.5,
            maxHeight: 22
        )
    }

    private func glowOpacity(for index: Int) -> Double {
        if isSubdued {
            return 0.03
        }
        let baseLevel = waveformState.barLevels.indices.contains(index) ? waveformState.barLevels[index] : 0
        return WaveformBarVisuals.glowOpacity(level: baseLevel, base: 0.03, gain: 0.18, cap: 0.22)
    }
}
