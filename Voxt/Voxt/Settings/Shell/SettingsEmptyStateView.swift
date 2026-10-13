// SettingsEmptyStateView.swift
// Provides Settings Empty State View for settings shell.

import SwiftUI

struct SettingsEmptyStateView: View {
    enum Illustration {
        case history
        case dictionary
        case generic
    }

    let illustration: Illustration
    let title: String
    let message: String

    init(
        illustration: Illustration = .generic,
        title: String,
        message: String
    ) {
        self.illustration = illustration
        self.title = title
        self.message = message
    }

    var body: some View {
        VStack(spacing: 24) {
            SettingsEmptyStateIllustration(kind: illustration)
                .frame(width: 150, height: 106)

            VStack(spacing: 12) {
                Text(title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Color.primary.opacity(0.86))
                    .multilineTextAlignment(.center)

                Text(message)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 320)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
    }
}

private struct SettingsEmptyStateIllustration: View {
    @Environment(\.colorScheme) private var colorScheme
    let kind: SettingsEmptyStateView.Illustration

    var body: some View {
        Image("EmptyStateIllustration")
            .resizable()
            .renderingMode(.template)
            .scaledToFit()
            .foregroundStyle(illustrationColor)
            .accessibilityHidden(true)
    }

    private var illustrationColor: Color {
        switch colorScheme {
        case .dark:
            return Color.white.opacity(0.32)
        default:
            return Color.black.opacity(0.18)
        }
    }
}
