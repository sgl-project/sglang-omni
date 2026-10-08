// FeatureSettingsCards.swift
// Provides Feature Settings Cards for feature settings.

import SwiftUI

struct FeatureSummaryPill: Identifiable {
    let title: String
    let value: String

    var id: String { "\(title)-\(value)" }
}

struct FeatureHeroCard: View {
    let title: String
    var titleBadge: String? = nil
    let subtitle: String
    let iconKind: SettingsSidebarIconKind?
    let systemImageName: String?
    let pills: [FeatureSummaryPill]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            let hasTitle = !title.isEmpty
            let hasSubtitle = !subtitle.isEmpty
            let hasBadge = titleBadge?.isEmpty == false

            if hasTitle || hasSubtitle || hasBadge {
                HStack(alignment: .top, spacing: 14) {
                    heroIcon

                    VStack(alignment: .leading, spacing: 6) {
                        if hasTitle || hasBadge {
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                if hasTitle {
                                    Text(title)
                                        .font(.title2.weight(.semibold))
                                }

                                if let titleBadge, !titleBadge.isEmpty {
                                    Text(titleBadge)
                                        .font(.caption2.weight(.semibold))
                                        .padding(.horizontal, 7)
                                        .padding(.vertical, 3)
                                        .background(
                                            Capsule()
                                                .fill(Color.accentColor.opacity(0.14))
                                        )
                                        .foregroundStyle(Color.accentColor)
                                }
                            }
                        }
                        if !subtitle.isEmpty {
                            Text(subtitle)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    Spacer(minLength: 0)
                }
            }

            if !pills.isEmpty {
                HStack(spacing: 10) {
                    ForEach(pills) { pill in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(pill.title.uppercased())
                                .font(.system(size: 10, weight: .bold))
                                .foregroundStyle(.secondary)
                            Text(pill.value)
                                .font(.system(size: 12, weight: .semibold))
                                .lineLimit(1)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                                .fill(SettingsUIStyle.controlFillColor)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                                .stroke(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
                        )
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var heroIcon: some View {
        if let iconKind {
            SettingsSidebarIconView(kind: iconKind)
                .foregroundStyle(Color.accentColor)
                .frame(width: 20, height: 20)
                .frame(width: 36, height: 36)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.accentColor.opacity(0.12))
                )
        } else if let systemImageName {
            Image(systemName: systemImageName)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Color.accentColor)
                .frame(width: 36, height: 36)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.accentColor.opacity(0.12))
                )
        }
    }
}

struct FeatureSettingsCard<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !title.isEmpty {
                HStack {
                    Text(title)
                        .font(.headline.weight(.semibold))
                    Spacer(minLength: 0)
                }
            }
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct FeatureSettingSection<Content: View>: View {
    let title: String
    let detail: String
    @ViewBuilder let content: Content

    init(title: String, detail: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.detail = detail
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !title.isEmpty {
                Text(title)
                    .font(.subheadline.weight(.semibold))
            }
            if !detail.isEmpty {
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
