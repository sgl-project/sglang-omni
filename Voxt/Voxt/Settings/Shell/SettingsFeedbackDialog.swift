import SwiftUI
import AppKit

extension SettingsView {
    struct FeedbackDialogView: View {
        let feedbackURL: URL
        let qrCodeURL: URL
        let onClose: () -> Void
        let onOpenFeedback: () -> Void

        var body: some View {
            VStack(alignment: .leading, spacing: 16) {
                Text(AppLocalization.localizedString("Feedback"))
                    .font(.title3.weight(.semibold))

                Text(AppLocalization.localizedString("Feedback Dialog Message"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                VStack(alignment: .leading, spacing: 10) {
                    Text(AppLocalization.localizedString("Author WeChat QR Code"))
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)

                    AsyncImage(url: qrCodeURL) { phase in
                        switch phase {
                        case .success(let image):
                            image
                                .resizable()
                                .interpolation(.high)
                                .scaledToFit()
                        case .failure:
                            VStack(spacing: 8) {
                                Image(systemName: "qrcode")
                                    .font(.system(size: 28, weight: .medium))
                                Text(AppLocalization.localizedString("Unable to load QR code"))
                                    .font(.system(size: 11, weight: .medium))
                            }
                            .foregroundStyle(.secondary)
                        case .empty:
                            ProgressView()
                                .controlSize(.small)
                        @unknown default:
                            ProgressView()
                                .controlSize(.small)
                        }
                    }
                    .frame(width: 180, height: 180)
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                            .fill(SettingsUIStyle.groupedFillColor)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                            .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
                    )
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text(AppLocalization.localizedString("GitHub Issues"))
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)

                    Text(feedbackURL.absoluteString)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .textSelection(.enabled)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                                .fill(SettingsUIStyle.controlFillColor)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                                .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
                        )
                }

                SettingsDialogActionRow {
                    Button(AppLocalization.localizedString("Close"), action: onClose)
                        .buttonStyle(SettingsPillButtonStyle())
                        .keyboardShortcut(.cancelAction)

                    Button(AppLocalization.localizedString("Open Feedback Page"), action: onOpenFeedback)
                        .buttonStyle(SettingsPrimaryButtonStyle())
                        .keyboardShortcut(.defaultAction)
                }
            }
            .settingsDialogChrome(width: 460, onClose: onClose)
        }
    }
}
