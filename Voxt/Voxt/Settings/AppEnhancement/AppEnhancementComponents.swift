// AppEnhancementComponents.swift
// Provides App Enhancement Components for app enhancement settings.

import SwiftUI

enum AppEnhancementToolbarMetrics {
    static let controlHeight: CGFloat = 26
}

struct SourceTabPicker: View {
    @Binding var selectedTab: SourceTab

    var body: some View {
        HStack(spacing: 2) {
            ForEach(SourceTab.allCases) { tab in
                Button {
                    selectedTab = tab
                } label: {
                    Text(tab.title)
                        .font(.system(size: 11.5, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .frame(height: 22)
                }
                .buttonStyle(.plain)
                .foregroundStyle(selectedTab == tab ? Color.accentColor : Color.secondary)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(selectedTab == tab ? Color.accentColor.opacity(0.14) : .clear)
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(selectedTab == tab ? Color.accentColor.opacity(0.45) : .clear, lineWidth: 1)
                }
            }
        }
        .padding(2)
        .frame(width: 154, height: AppEnhancementToolbarMetrics.controlHeight)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(SettingsUIStyle.controlFillColor)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        }
    }
}

struct URLPatternRowView: View {
    let pattern: String
    let groupName: String?
    var onRemoveFromGroup: (() -> Void)? = nil
    let onEdit: () -> Void
    let onDelete: () -> Void
    @State private var isHovering = false

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "globe")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 18)

            Text(pattern)
                .font(.system(size: 12, weight: .medium))
                .lineLimit(1)
                .truncationMode(.middle)

            Spacer()

            if let groupName {
                HStack(spacing: 4) {
                    Text(groupName)
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    if let onRemoveFromGroup {
                        Button {
                            onRemoveFromGroup()
                        } label: {
                            Image(systemName: "trash")
                                .font(.system(size: 8, weight: .bold))
                                .foregroundStyle(.white)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Capsule().fill(Color.accentColor))
                .frame(maxWidth: 56, alignment: .trailing)
            }

            Button(AppLocalization.localizedString("Edit")) {
                onEdit()
            }
            .buttonStyle(SettingsCompactActionButtonStyle(height: 26, horizontalPadding: 8))

            Button(AppLocalization.localizedString("Delete")) {
                onDelete()
            }
            .buttonStyle(SettingsCompactActionButtonStyle(tone: .destructive, height: 26, horizontalPadding: 8))
        }
        .padding(.horizontal, 10)
        .frame(height: 36)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(isHovering ? SettingsUIStyle.sidebarItemFillColor : SettingsUIStyle.controlFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(isHovering ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
        .contentShape(Rectangle())
        .animation(.easeInOut(duration: 0.14), value: isHovering)
        .onHover { hovering in
            isHovering = hovering
        }
    }
}
