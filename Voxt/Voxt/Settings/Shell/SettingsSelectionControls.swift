// SettingsSelectionControls.swift
// Provides Settings Selection Controls for settings shell.

import AppKit
import SwiftUI

struct SettingsMenuOption<Value: Hashable>: Identifiable {
    let value: Value
    let title: String

    var id: AnyHashable { AnyHashable(value) }
}

struct SettingsMenuPicker<Value: Hashable>: View {
    @Binding var selection: Value
    let options: [SettingsMenuOption<Value>]
    let selectedTitle: String
    let width: CGFloat
    var leadingAccessory: AnyView?
    var selectedStatusSystemImageName: String?
    var selectedStatusTintColor: NSColor = .systemGreen
    var allowsCompactWidth = false
    var isCompact = false
    var usesCompactInsets = false

    private var resolvedWidth: CGFloat {
        allowsCompactWidth ? max(ceil(width), 1) : SettingsUIStyle.resolvedSelectWidth(width)
    }

    private var resolvedHeight: CGFloat {
        isCompact ? 24 : 34
    }

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            if let leadingAccessory {
                leadingAccessory
            }

            SettingsNativeMenuPicker(
                selection: $selection,
                options: options,
                selectedTitle: selectedTitle,
                selectedStatusSystemImageName: selectedStatusSystemImageName,
                selectedStatusTintColor: selectedStatusTintColor,
                preferredWidth: resolvedWidth,
                isCompact: isCompact,
                usesCompactInsets: usesCompactInsets
            )
            .frame(width: resolvedWidth, height: resolvedHeight)
        }
        .alignmentGuide(.firstTextBaseline) { dimensions in
            dimensions[VerticalAlignment.center]
        }
        .alignmentGuide(.lastTextBaseline) { dimensions in
            dimensions[VerticalAlignment.center]
        }
    }
}

struct SettingsFixedSelectionBlock: View {
    let title: String
    let width: CGFloat
    var usesCompactInsets = false

    var body: some View {
        Text(title)
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(.primary)
            .lineLimit(1)
            .padding(.horizontal, usesCompactInsets ? 8 : 12)
            .frame(
                width: max(ceil(width), 1),
                height: 34,
                alignment: .leading
            )
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .fill(SettingsUIStyle.controlFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
            )
    }
}

struct SettingsSelectionButton<Label: View>: View {
    let width: CGFloat
    var height: CGFloat = 34
    var allowsCompactWidth = false
    let action: () -> Void
    @ViewBuilder let label: () -> Label

    private var resolvedWidth: CGFloat {
        allowsCompactWidth ? max(ceil(width), 1) : SettingsUIStyle.resolvedSelectWidth(width)
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                label()
                    .frame(maxWidth: .infinity, alignment: .leading)

                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(SettingsSelectLikeButtonStyle(height: height))
        .frame(width: resolvedWidth)
    }
}

struct SettingsPathSelectionRow: View {
    let title: String
    let displayedPath: String
    let fallbackPath: String
    var pathWidth: CGFloat = 260
    let openButtonHelp: String
    let chooseButtonTitle: String
    let onOpen: () -> Void
    let onChoose: () -> Void

    private var resolvedPath: String {
        displayedPath.isEmpty ? fallbackPath : displayedPath
    }

    var body: some View {
        GeneralFieldRow(titleText: title) {
            Button(action: onOpen) {
                HStack(spacing: 6) {
                    Image(systemName: "folder")
                        .font(.caption)
                    Text(resolvedPath)
                        .underline()
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Image(systemName: "arrow.up.forward.square")
                        .font(.caption)
                }
                .frame(width: pathWidth, alignment: .leading)
            }
            .buttonStyle(SettingsInlineSelectorButtonStyle())
            .help(openButtonHelp)

            Button(chooseButtonTitle, action: onChoose)
                .buttonStyle(SettingsPillButtonStyle())
        }
    }
}

struct SettingsSelectLikeButtonStyle: ButtonStyle {
    var height: CGFloat = 34

    func makeBody(configuration: Configuration) -> some View {
        SettingsSelectLikeButtonBody(configuration: configuration, height: height)
    }
}

private struct SettingsSelectLikeButtonBody: View {
    let configuration: SettingsSelectLikeButtonStyle.Configuration
    let height: CGFloat
    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(.primary)
            .padding(.horizontal, 12)
            .frame(height: height)
            .background(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .fill(SettingsUIStyle.controlFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    .strokeBorder(isHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous))
            .opacity(configuration.isPressed ? 0.92 : 1)
            .onHover { isHovered = $0 }
    }
}

struct SettingsDialogActionRow<Leading: View, Trailing: View>: View {
    @ViewBuilder let leading: Leading
    @ViewBuilder let trailing: Trailing

    init(
        @ViewBuilder leading: () -> Leading = { EmptyView() },
        @ViewBuilder trailing: () -> Trailing
    ) {
        self.leading = leading()
        self.trailing = trailing()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            leading
            Spacer(minLength: 12)
            trailing
        }
        .padding(.top, 4)
    }
}

enum SettingsMenuInteraction {
    @discardableResult
    static func performSelection(for menuItem: NSMenuItem?) -> Bool {
        guard
            let menuItem,
            let menu = menuItem.menu
        else {
            return false
        }

        let index = menu.index(of: menuItem)
        guard index >= 0 else {
            return false
        }

        menu.performActionForItem(at: index)
        menu.cancelTracking()
        return true
    }
}

private struct SettingsNativeMenuPicker<Value: Hashable>: NSViewRepresentable {
    @Binding var selection: Value
    let options: [SettingsMenuOption<Value>]
    let selectedTitle: String
    let selectedStatusSystemImageName: String?
    let selectedStatusTintColor: NSColor
    let preferredWidth: CGFloat
    let isCompact: Bool
    let usesCompactInsets: Bool

    private var state: SettingsNativeMenuPickerState {
        let selectionBinding = $selection
        return SettingsNativeMenuPickerState(
            options: options.map { option in
                SettingsNativeMenuPickerOption(value: AnyHashable(option.value), title: option.title)
            },
            selectedValue: AnyHashable(selection),
            selectedTitle: selectedTitle,
            selectedStatusSystemImageName: selectedStatusSystemImageName,
            selectedStatusTintColor: selectedStatusTintColor,
            preferredWidth: preferredWidth,
            isCompact: isCompact,
            usesCompactInsets: usesCompactInsets,
            onSelectValue: { selectedValue in
                guard let value = selectedValue.base as? Value else { return }
                if selectionBinding.wrappedValue != value {
                    selectionBinding.wrappedValue = value
                }
            }
        )
    }

    func makeCoordinator() -> SettingsNativeMenuPickerCoordinator {
        SettingsNativeMenuPickerCoordinator(state: state)
    }

    func makeNSView(context: Context) -> SettingsMenuHostView {
        let hostView = SettingsMenuHostView()
        hostView.onSelectIndex = { [weak coordinator = context.coordinator] index in
            coordinator?.selectionDidChange(index: index)
        }
        return hostView
    }

    func updateNSView(_ nsView: SettingsMenuHostView, context: Context) {
        context.coordinator.state = state
        nsView.onSelectIndex = { [weak coordinator = context.coordinator] index in
            coordinator?.selectionDidChange(index: index)
        }
        context.coordinator.update(nsView)
    }
}

private final class SettingsNativeMenuPickerCoordinator: NSObject {
    var state: SettingsNativeMenuPickerState

    init(state: SettingsNativeMenuPickerState) {
        self.state = state
    }

    func update(_ hostView: SettingsMenuHostView) {
        hostView.setCompactMode(state.isCompact, usesCompactInsets: state.usesCompactInsets)
        let titles = state.options.map(\.title)
        if let selectedIndex = state.options.firstIndex(where: { $0.value == state.selectedValue }) {
            hostView.toolTip = state.options[selectedIndex].title
            hostView.updateMenu(
                titles: titles,
                selectedIndex: selectedIndex,
                fallbackTitle: state.options[selectedIndex].title,
                statusSystemImageName: state.selectedStatusSystemImageName,
                statusTintColor: state.selectedStatusTintColor,
                preferredWidth: state.preferredWidth
            )
        } else if let firstOption = state.options.first {
            hostView.toolTip = firstOption.title
            hostView.updateMenu(
                titles: titles,
                selectedIndex: 0,
                fallbackTitle: firstOption.title,
                statusSystemImageName: state.selectedStatusSystemImageName,
                statusTintColor: state.selectedStatusTintColor,
                preferredWidth: state.preferredWidth
            )
            if state.selectedValue != firstOption.value {
                DispatchQueue.main.async { [weak self] in
                    self?.state.onSelectValue(firstOption.value)
                }
            }
        } else {
            hostView.toolTip = state.selectedTitle
            hostView.updateMenu(
                titles: [],
                selectedIndex: nil,
                fallbackTitle: state.selectedTitle,
                statusSystemImageName: state.selectedStatusSystemImageName,
                statusTintColor: state.selectedStatusTintColor,
                preferredWidth: state.preferredWidth
            )
        }
    }

    func selectionDidChange(index: Int) {
        guard state.options.indices.contains(index) else { return }
        state.onSelectValue(state.options[index].value)
    }
}

private struct SettingsNativeMenuPickerState {
    let options: [SettingsNativeMenuPickerOption]
    let selectedValue: AnyHashable
    let selectedTitle: String
    let selectedStatusSystemImageName: String?
    let selectedStatusTintColor: NSColor
    let preferredWidth: CGFloat
    let isCompact: Bool
    let usesCompactInsets: Bool
    let onSelectValue: (AnyHashable) -> Void
}

private struct SettingsNativeMenuPickerOption {
    let value: AnyHashable
    let title: String
}

private final class SettingsMenuHostView: NSView {
    private let titleField = NSTextField(labelWithString: "")
    private let statusImageView = NSImageView()
    private let indicatorView = NSImageView()
    private let popupMenu = NSMenu()
    private var titleLeadingConstraint: NSLayoutConstraint?
    private var titleStatusConstraint: NSLayoutConstraint?
    private var statusIndicatorConstraint: NSLayoutConstraint?
    private var statusWidthConstraint: NSLayoutConstraint?
    private var statusHeightConstraint: NSLayoutConstraint?
    private var indicatorTrailingConstraint: NSLayoutConstraint?
    private var indicatorWidthConstraint: NSLayoutConstraint?
    private var indicatorHeightConstraint: NSLayoutConstraint?
    private var selectedIndex: Int?
    private var currentMenuWidth: CGFloat = 0
    private var isCompact = false
    private var usesCompactInsets = false
    private var isHovered = false
    private var trackingArea: NSTrackingArea?
    var onSelectIndex: ((Int) -> Void)?

    override var isFlipped: Bool {
        true
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        popupMenu.autoenablesItems = false
        popupMenu.showsStateColumn = true

        titleField.translatesAutoresizingMaskIntoConstraints = false
        titleField.font = .systemFont(ofSize: 13, weight: .medium)
        titleField.textColor = .labelColor
        titleField.lineBreakMode = .byTruncatingTail
        titleField.maximumNumberOfLines = 1
        titleField.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        statusImageView.translatesAutoresizingMaskIntoConstraints = false
        statusImageView.isHidden = true
        statusImageView.contentTintColor = .systemGreen

        indicatorView.translatesAutoresizingMaskIntoConstraints = false
        indicatorView.image = NSImage(
            systemSymbolName: "chevron.up.chevron.down",
            accessibilityDescription: nil
        )?.withSymbolConfiguration(.init(pointSize: 11, weight: .semibold))
        indicatorView.contentTintColor = .secondaryLabelColor

        addSubview(titleField)
        addSubview(statusImageView)
        addSubview(indicatorView)

        titleLeadingConstraint = titleField.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12)
        titleStatusConstraint = titleField.trailingAnchor.constraint(lessThanOrEqualTo: statusImageView.leadingAnchor, constant: -5)
        statusIndicatorConstraint = statusImageView.trailingAnchor.constraint(equalTo: indicatorView.leadingAnchor, constant: -7)
        statusWidthConstraint = statusImageView.widthAnchor.constraint(equalToConstant: 0)
        statusHeightConstraint = statusImageView.heightAnchor.constraint(equalToConstant: 0)
        indicatorTrailingConstraint = indicatorView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10)
        indicatorWidthConstraint = indicatorView.widthAnchor.constraint(equalToConstant: 14)
        indicatorHeightConstraint = indicatorView.heightAnchor.constraint(equalToConstant: 14)

        NSLayoutConstraint.activate([
            titleLeadingConstraint!,
            titleStatusConstraint!,
            titleField.centerYAnchor.constraint(equalTo: centerYAnchor),
            statusIndicatorConstraint!,
            statusImageView.centerYAnchor.constraint(equalTo: centerYAnchor),
            statusWidthConstraint!,
            statusHeightConstraint!,
            indicatorTrailingConstraint!,
            indicatorView.centerYAnchor.constraint(equalTo: centerYAnchor),
            indicatorWidthConstraint!,
            indicatorHeightConstraint!
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        layer?.cornerRadius = SettingsUIStyle.controlCornerRadius
        layer?.backgroundColor = SettingsUIStyle.controlFillNSColor.cgColor
        updateBorder()
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
        updateBorder()
    }

    override func mouseExited(with event: NSEvent) {
        isHovered = false
        updateBorder()
    }

    private func updateBorder() {
        layer?.borderColor = (isHovered ? SettingsUIStyle.controlHoverBorderNSColor : SettingsUIStyle.subtleBorderNSColor).cgColor
        layer?.borderWidth = 1
    }

    func setCompactMode(_ compact: Bool, usesCompactInsets compactInsets: Bool) {
        guard isCompact != compact || usesCompactInsets != compactInsets else { return }
        isCompact = compact
        usesCompactInsets = compactInsets
        let shouldUseCompactInsets = compact || compactInsets
        titleField.font = .systemFont(ofSize: compact ? 11 : 13, weight: .medium)
        indicatorView.image = NSImage(
            systemSymbolName: "chevron.up.chevron.down",
            accessibilityDescription: nil
        )?.withSymbolConfiguration(.init(pointSize: compact ? 9 : 11, weight: .semibold))
        titleLeadingConstraint?.constant = shouldUseCompactInsets ? 8 : 12
        titleStatusConstraint?.constant = shouldUseCompactInsets ? -3 : -5
        statusIndicatorConstraint?.constant = shouldUseCompactInsets ? -4 : -7
        indicatorTrailingConstraint?.constant = shouldUseCompactInsets ? -7 : -10
        indicatorWidthConstraint?.constant = compact ? 10 : 14
        indicatorHeightConstraint?.constant = compact ? 10 : 14
        updateStatusImageSize()
        titleField.alignment = compactInsets ? .right : .left
        DispatchQueue.main.async { [weak self] in
            self?.needsLayout = true
        }
    }

    func updateMenu(
        titles: [String],
        selectedIndex: Int?,
        fallbackTitle: String,
        statusSystemImageName: String?,
        statusTintColor: NSColor,
        preferredWidth: CGFloat
    ) {
        let menuWidth = max(ceil(preferredWidth), 1)
        let needsRebuild = popupMenu.items.map(\.title) != titles || abs(currentMenuWidth - menuWidth) > 0.5

        if needsRebuild {
            popupMenu.removeAllItems()
            for (index, title) in titles.enumerated() {
                let item = NSMenuItem(title: title, action: #selector(selectMenuItem(_:)), keyEquivalent: "")
                item.target = self
                item.tag = index
                item.state = index == selectedIndex ? .on : .off
                applyMenuItemAppearance(item)
                popupMenu.addItem(item)
            }
            currentMenuWidth = menuWidth
        }

        self.selectedIndex = selectedIndex
        for item in popupMenu.items {
            item.state = item.tag == selectedIndex ? .on : .off
            applyMenuItemAppearance(item)
        }

        popupMenu.minimumWidth = menuWidth
        titleField.stringValue = fallbackTitle
        updateStatusImage(systemImageName: statusSystemImageName, tintColor: statusTintColor)
    }

    private func updateStatusImage(systemImageName: String?, tintColor: NSColor) {
        guard let systemImageName,
              let image = NSImage(
                systemSymbolName: systemImageName,
                accessibilityDescription: nil
              )?.withSymbolConfiguration(.init(pointSize: isCompact ? 9 : 11, weight: .semibold))
        else {
            statusImageView.image = nil
            statusImageView.isHidden = true
            statusWidthConstraint?.constant = 0
            statusHeightConstraint?.constant = 0
            return
        }
        statusImageView.image = image
        statusImageView.contentTintColor = tintColor
        statusImageView.isHidden = false
        updateStatusImageSize()
    }

    private func updateStatusImageSize() {
        guard !statusImageView.isHidden else { return }
        statusWidthConstraint?.constant = isCompact ? 10 : 12
        statusHeightConstraint?.constant = isCompact ? 10 : 12
    }

    private func applyMenuItemAppearance(_ item: NSMenuItem) {
        guard isCompact else { return }
        item.attributedTitle = NSAttributedString(
            string: item.title,
            attributes: [
                .font: NSFont.systemFont(ofSize: 11, weight: .medium)
            ]
        )
    }

    override func mouseDown(with event: NSEvent) {
        guard !popupMenu.items.isEmpty else { return }
        _ = popupMenu.popUp(positioning: nil, at: NSPoint(x: 0, y: bounds.height + 8), in: self)
    }

    @objc
    private func selectMenuItem(_ sender: NSMenuItem) {
        onSelectIndex?(sender.tag)
    }
}
