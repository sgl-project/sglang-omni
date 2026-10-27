import SwiftUI
import AppKit

struct OnboardingGuideHeaderStepButton: View {
    let title: String
    let alignment: Alignment
    let isEnabled: Bool
    let action: () -> Void

    @State private var isHovered = false

    private var opacity: Double {
        guard isEnabled else { return 0.24 }
        return isHovered ? 0.82 : 0.38
    }

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(OnboardingGuideStyle.primaryText)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: alignment)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(opacity)
        .onHover { isHovered = $0 }
        .animation(.easeOut(duration: 0.12), value: isHovered)
        .frame(maxWidth: .infinity)
    }
}

struct OnboardingGuideProgressRing: View {
    let current: Int
    let total: Int

    private let ringSize: CGFloat = 32
    private let totalBadgeSize: CGFloat = 14

    private var progress: CGFloat {
        guard total > 0 else { return 0 }
        return CGFloat(current) / CGFloat(total)
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            ZStack {
                Circle()
                    .stroke(OnboardingGuideStyle.progressTrack, lineWidth: 2.5)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(
                        LinearGradient(
                            colors: [
                                Color(red: 0.30, green: 0.74, blue: 1.0),
                                Color(red: 0.08, green: 0.48, blue: 1.0)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        style: StrokeStyle(lineWidth: 2.5, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))

                Text("\(current)")
                    .font(.system(size: 12, weight: .bold, design: .rounded))
                    .foregroundStyle(OnboardingGuideStyle.primaryText)
                    .monospacedDigit()
            }
            .frame(width: ringSize, height: ringSize)

            Text("\(total)")
                .font(.system(size: 7, weight: .bold, design: .rounded))
                .foregroundStyle(OnboardingGuideStyle.primaryText)
                .monospacedDigit()
                .frame(width: totalBadgeSize, height: totalBadgeSize)
                .background(
                    Circle()
                        .fill(OnboardingGuideStyle.panelFill)
                )
                .overlay(
                    Circle()
                        .strokeBorder(OnboardingGuideStyle.subtleBorder, lineWidth: 1)
                )
        }
        .frame(width: 32, height: 32)
        .accessibilityLabel(Text(AppLocalization.format("%d/%d", current, total)))
    }
}

struct OnboardingMicrophoneSelectLabel: View {
    let deviceName: String
    let hasDetectedAudio: Bool
    let showsMenuIndicator: Bool

    var body: some View {
        HStack(spacing: 8) {
            TranscriptionModeIconView(
                color: hasDetectedAudio ? Color.green : OnboardingGuideStyle.primaryText
            )
            .frame(width: 15, height: 15)
            .accessibilityHidden(true)

            Text(deviceName)
                .font(.caption.weight(.medium))
                .foregroundStyle(OnboardingGuideStyle.primaryText)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)

            if showsMenuIndicator {
                Image(systemName: "chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(OnboardingGuideStyle.secondaryText)
            }
        }
        .padding(.horizontal, 9)
        .frame(width: 190, height: 30)
        .background(
            Capsule(style: .continuous)
                .fill(OnboardingGuideStyle.panelFill)
        )
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(OnboardingGuideStyle.subtleBorder, lineWidth: 1)
        )
        .contentShape(Capsule(style: .continuous))
    }
}

struct OnboardingLanguageSelectLabel: View {
    let summary: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "globe")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(OnboardingGuideStyle.primaryText)
                .frame(width: 15, height: 15)
                .accessibilityHidden(true)

            Text(summary)
                .font(.caption.weight(.medium))
                .foregroundStyle(OnboardingGuideStyle.primaryText)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)

            Image(systemName: "chevron.down")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(OnboardingGuideStyle.secondaryText)
        }
        .padding(.horizontal, 9)
        .frame(width: 190, height: 30)
        .background(
            Capsule(style: .continuous)
                .fill(OnboardingGuideStyle.panelFill)
        )
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(OnboardingGuideStyle.subtleBorder, lineWidth: 1)
        )
        .contentShape(Capsule(style: .continuous))
    }
}
