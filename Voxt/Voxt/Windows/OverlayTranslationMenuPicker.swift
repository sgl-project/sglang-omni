// OverlayTranslationMenuPicker.swift
// Provides the translation menu picker for window and overlay UI.

import SwiftUI
import AppKit

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
