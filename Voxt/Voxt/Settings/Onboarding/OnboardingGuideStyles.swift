import SwiftUI
import AppKit

enum OnboardingGuideStyle {
    static let windowCornerRadius: CGFloat = 22
    static let panelCornerRadius: CGFloat = 20
    static let modalCornerRadius: CGFloat = 18
    static let innerCornerRadius: CGFloat = 16
    static let headerTransitionRadius: CGFloat = 24

    static let windowBackground = dynamicColor(
        light: NSColor(calibratedRed: 0.965, green: 0.968, blue: 0.972, alpha: 1),
        dark: NSColor(calibratedRed: 0.080, green: 0.085, blue: 0.090, alpha: 1)
    )
    static let windowBorder = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.08),
        dark: NSColor.white.withAlphaComponent(0.12)
    )
    static let modalScrim = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.22),
        dark: NSColor.black.withAlphaComponent(0.46)
    )
    static let panelFill = dynamicColor(
        light: NSColor(calibratedWhite: 1.0, alpha: 1),
        dark: NSColor(calibratedRed: 0.105, green: 0.110, blue: 0.120, alpha: 1)
    )
    static let panelBorder = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.065),
        dark: NSColor.white.withAlphaComponent(0.10)
    )
    static let subtleBorder = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.075),
        dark: NSColor.white.withAlphaComponent(0.12)
    )
    static let visualTopFill = dynamicColor(
        light: NSColor(calibratedRed: 0.925, green: 0.940, blue: 0.960, alpha: 1),
        dark: NSColor(calibratedRed: 0.160, green: 0.180, blue: 0.200, alpha: 1)
    )
    static let visualBottomFill = dynamicColor(
        light: NSColor(calibratedRed: 0.982, green: 0.984, blue: 0.988, alpha: 1),
        dark: NSColor(calibratedRed: 0.085, green: 0.090, blue: 0.100, alpha: 1)
    )
    static let primaryText = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.90),
        dark: NSColor.white.withAlphaComponent(0.96)
    )
    static let secondaryText = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.62),
        dark: NSColor.white.withAlphaComponent(0.68)
    )
    static let mutedText = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.42),
        dark: NSColor.white.withAlphaComponent(0.46)
    )
    static let controlFill = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.055),
        dark: NSColor.white.withAlphaComponent(0.11)
    )
    static let cardFill = dynamicColor(
        light: NSColor(calibratedWhite: 0.955, alpha: 1),
        dark: NSColor(calibratedWhite: 0.14, alpha: 1)
    )
    static let controlPressedFill = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.095),
        dark: NSColor.white.withAlphaComponent(0.17)
    )
    static let progressTrack = dynamicColor(
        light: NSColor.black.withAlphaComponent(0.12),
        dark: NSColor.white.withAlphaComponent(0.16)
    )
    static let primaryButtonBorder = dynamicColor(
        light: NSColor.white.withAlphaComponent(0.30),
        dark: NSColor.white.withAlphaComponent(0.20)
    )

    private static func dynamicColor(light: NSColor, dark: NSColor) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            switch appearance.bestMatch(from: [.darkAqua, .aqua]) {
            case .darkAqua:
                return dark
            default:
                return light
            }
        })
    }
}

struct OnboardingGuideShellShape: Shape {
    let headerHeight: CGFloat
    let sideCutoutWidth: CGFloat
    let cornerRadius: CGFloat
    let transitionRadius: CGFloat

    func path(in rect: CGRect) -> Path {
        let width = rect.width
        let height = rect.height
        let topInset = min(max(sideCutoutWidth, 0), width / 3)
        let headerHeight = min(max(headerHeight, 0), height)
        let cornerRadius = min(max(cornerRadius, 0), width / 2, height / 2)
        let transitionRadius = min(max(transitionRadius, 0), topInset, headerHeight)

        var path = Path()
        path.move(to: CGPoint(x: topInset + cornerRadius, y: 0))
        path.addLine(to: CGPoint(x: width - topInset - cornerRadius, y: 0))
        path.addQuadCurve(
            to: CGPoint(x: width - topInset, y: cornerRadius),
            control: CGPoint(x: width - topInset, y: 0)
        )
        path.addLine(to: CGPoint(x: width - topInset, y: headerHeight - transitionRadius))
        path.addQuadCurve(
            to: CGPoint(x: width - topInset + transitionRadius, y: headerHeight),
            control: CGPoint(x: width - topInset, y: headerHeight)
        )
        path.addLine(to: CGPoint(x: width - cornerRadius, y: headerHeight))
        path.addQuadCurve(
            to: CGPoint(x: width, y: headerHeight + cornerRadius),
            control: CGPoint(x: width, y: headerHeight)
        )
        path.addLine(to: CGPoint(x: width, y: height - cornerRadius))
        path.addQuadCurve(
            to: CGPoint(x: width - cornerRadius, y: height),
            control: CGPoint(x: width, y: height)
        )
        path.addLine(to: CGPoint(x: cornerRadius, y: height))
        path.addQuadCurve(
            to: CGPoint(x: 0, y: height - cornerRadius),
            control: CGPoint(x: 0, y: height)
        )
        path.addLine(to: CGPoint(x: 0, y: headerHeight + cornerRadius))
        path.addQuadCurve(
            to: CGPoint(x: cornerRadius, y: headerHeight),
            control: CGPoint(x: 0, y: headerHeight)
        )
        path.addLine(to: CGPoint(x: topInset - transitionRadius, y: headerHeight))
        path.addQuadCurve(
            to: CGPoint(x: topInset, y: headerHeight - transitionRadius),
            control: CGPoint(x: topInset, y: headerHeight)
        )
        path.addLine(to: CGPoint(x: topInset, y: cornerRadius))
        path.addQuadCurve(
            to: CGPoint(x: topInset + cornerRadius, y: 0),
            control: CGPoint(x: topInset, y: 0)
        )
        path.closeSubpath()
        return path
    }
}

struct OnboardingGuideIconButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(OnboardingGuideStyle.primaryText)
            .background(
                Circle()
                    .fill(configuration.isPressed ? OnboardingGuideStyle.controlPressedFill : OnboardingGuideStyle.controlFill)
            )
            .overlay(
                Circle()
                    .strokeBorder(OnboardingGuideStyle.subtleBorder, lineWidth: 1)
            )
            .opacity(isEnabled ? 1 : 0.45)
            .contentShape(Circle())
    }
}

struct OnboardingGuidePrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .frame(minWidth: 86, minHeight: 32)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.accentColor.opacity(configuration.isPressed ? 0.82 : 0.94))
            )
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(OnboardingGuideStyle.primaryButtonBorder, lineWidth: 1)
            )
            .opacity(isEnabled ? 1 : 0.45)
            .contentShape(Capsule(style: .continuous))
    }
}

struct OnboardingGuideSecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(OnboardingGuideStyle.primaryText)
            .padding(.horizontal, 12)
            .frame(minHeight: 32)
            .background(
                Capsule(style: .continuous)
                    .fill(configuration.isPressed ? OnboardingGuideStyle.controlPressedFill : OnboardingGuideStyle.controlFill)
            )
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(OnboardingGuideStyle.subtleBorder, lineWidth: 1)
            )
            .opacity(isEnabled ? 1 : 0.45)
            .contentShape(Capsule(style: .continuous))
    }
}

struct OnboardingGuideNextLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 6) {
            configuration.title
            configuration.icon
        }
    }
}
