// DictionaryHeaderActions.swift
// Provides Dictionary Header Actions for dictionary settings.

import SwiftUI
import AppKit

struct DictionaryHeaderMenuAction {
    let title: String
    let handler: () -> Void

    init(title: String, handler: @escaping () -> Void) {
        self.title = title
        self.handler = handler
    }
}

struct DictionaryHeaderActionMenuButton: View {
    let actions: [DictionaryHeaderMenuAction]
    var accessibilityLabel = AppLocalization.localizedString("More")

    var body: some View {
        DictionaryHeaderActionMenuRepresentable(
            actions: actions,
            accessibilityLabel: accessibilityLabel
        )
        .frame(width: 30, height: 30)
        .accessibilityLabel(accessibilityLabel)
    }
}

private struct DictionaryHeaderActionMenuRepresentable: NSViewRepresentable {
    let actions: [DictionaryHeaderMenuAction]
    let accessibilityLabel: String

    func makeCoordinator() -> Coordinator {
        Coordinator(actions: actions)
    }

    func makeNSView(context: Context) -> DictionaryHeaderActionMenuHostView {
        let hostView = DictionaryHeaderActionMenuHostView()
        hostView.toolTip = accessibilityLabel
        hostView.setAccessibilityLabel(accessibilityLabel)
        hostView.update(actions: actions, target: context.coordinator)
        return hostView
    }

    func updateNSView(_ nsView: DictionaryHeaderActionMenuHostView, context: Context) {
        context.coordinator.actions = actions
        nsView.toolTip = accessibilityLabel
        nsView.setAccessibilityLabel(accessibilityLabel)
        nsView.update(actions: actions, target: context.coordinator)
    }

    final class Coordinator: NSObject {
        var actions: [DictionaryHeaderMenuAction]

        init(actions: [DictionaryHeaderMenuAction]) {
            self.actions = actions
        }

        @objc
        func performAction(_ sender: NSMenuItem) {
            guard actions.indices.contains(sender.tag) else { return }
            actions[sender.tag].handler()
        }
    }
}

private final class DictionaryHeaderActionMenuHostView: NSView {
    private let popupMenu = NSMenu()
    private let iconView = NSHostingView(rootView: SettingsMoreMenuIconView(color: Color(nsColor: .secondaryLabelColor), size: 15))
    private var trackingAreaRef: NSTrackingArea?
    private var isHovered = false {
        didSet { updateAppearance() }
    }
    private var isPressed = false {
        didSet { updateAppearance() }
    }

    override var isFlipped: Bool {
        true
    }

    override var intrinsicContentSize: NSSize {
        NSSize(width: 30, height: 30)
    }

    override var acceptsFirstResponder: Bool {
        true
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        popupMenu.autoenablesItems = false

        iconView.translatesAutoresizingMaskIntoConstraints = false
        iconView.setContentHuggingPriority(.required, for: .horizontal)
        iconView.setContentHuggingPriority(.required, for: .vertical)
        addSubview(iconView)

        NSLayoutConstraint.activate([
            iconView.centerXAnchor.constraint(equalTo: centerXAnchor),
            iconView.centerYAnchor.constraint(equalTo: centerYAnchor),
            iconView.widthAnchor.constraint(equalToConstant: 15),
            iconView.heightAnchor.constraint(equalToConstant: 15)
        ])

        setAccessibilityRole(.button)
        updateAppearance()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        layer?.cornerRadius = 8
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let trackingAreaRef {
            removeTrackingArea(trackingAreaRef)
        }

        let trackingArea = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeInActiveApp, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(trackingArea)
        trackingAreaRef = trackingArea
    }

    override func mouseEntered(with event: NSEvent) {
        isHovered = true
    }

    override func mouseExited(with event: NSEvent) {
        isHovered = false
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        showMenu()
    }

    override func keyDown(with event: NSEvent) {
        switch event.keyCode {
        case 36, 49, 76:
            showMenu()
        default:
            super.keyDown(with: event)
        }
    }

    override func accessibilityPerformPress() -> Bool {
        showMenu()
        return true
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateAppearance()
    }

    func update(actions: [DictionaryHeaderMenuAction], target: AnyObject) {
        popupMenu.removeAllItems()
        for (index, action) in actions.enumerated() {
            let item = NSMenuItem(
                title: action.title,
                action: #selector(DictionaryHeaderActionMenuRepresentable.Coordinator.performAction(_:)),
                keyEquivalent: ""
            )
            item.target = target
            item.tag = index
            popupMenu.addItem(item)
        }
    }

    private func showMenu() {
        guard !popupMenu.items.isEmpty else { return }
        isPressed = true
        _ = popupMenu.popUp(positioning: nil, at: NSPoint(x: 0, y: bounds.height + 6), in: self)
        isPressed = false
    }

    private func updateAppearance() {
        let fillColor: NSColor
        if isPressed {
            fillColor = SettingsUIStyle.subtleFillNSColor.blended(withFraction: 0.18, of: .labelColor) ?? SettingsUIStyle.subtleFillNSColor
        } else if isHovered {
            fillColor = SettingsUIStyle.subtleFillNSColor.blended(withFraction: 0.08, of: .labelColor) ?? SettingsUIStyle.subtleFillNSColor
        } else {
            fillColor = SettingsUIStyle.subtleFillNSColor
        }

        layer?.backgroundColor = fillColor.cgColor
        layer?.borderColor = SettingsUIStyle.subtleBorderNSColor.cgColor
        layer?.borderWidth = 1
        iconView.rootView = SettingsMoreMenuIconView(
            color: Color(nsColor: isPressed ? .labelColor : .secondaryLabelColor),
            size: 15
        )
    }
}
