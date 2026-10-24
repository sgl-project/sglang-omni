import SwiftUI
import AppKit

struct SettingsSidebarHeader: View {
    let sidebarMode: SettingsSidebarMode
    let updateBadgeState: UpdateBadgeState
    let onTapUpdateBadge: () -> Void
    let onReturnToRoot: () -> Void

    private var appVersionText: String {
        let bundle = Bundle.main
        let shortVersion = (bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if let shortVersion, !shortVersion.isEmpty {
            return "v\(shortVersion)"
        }
        return ""
    }

    private var showsNewVersionTag: Bool {
        if case .newVersion = updateBadgeState {
            return true
        }
        return false
    }

    private var headerBadgeHeight: CGFloat {
        19
    }

    var body: some View {
        HStack {
            switch sidebarMode {
            case .root:
                Spacer(minLength: 0)

                if !appVersionText.isEmpty {
                    if showsNewVersionTag {
                        Button(action: onTapUpdateBadge) {
                            badgeContent
                        }
                        .buttonStyle(.plain)
                    } else {
                        badgeContent
                    }
                }

            case .feature, .history, .settings:
                Spacer(minLength: 0)

                Button(action: onReturnToRoot) {
                    SettingsSidebarBackIcon()
                        .frame(width: 16, height: 16)
                        .padding(.horizontal, 15)
                        .frame(height: headerBadgeHeight)
                        .contentShape(Capsule(style: .continuous))
                }
                .buttonStyle(SettingsSidebarHeaderBackButtonStyle())
                .accessibilityLabel(AppLocalization.localizedString("Back"))
            }
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
        .padding(.top, 5)
        .padding(.bottom, 14)
    }

    private var badgeContent: some View {
        HStack(spacing: 4) {
            if showsNewVersionTag {
                Text(AppLocalization.localizedString("New"))
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(Color.green)
                    .lineLimit(1)
                    .padding(.horizontal, 7)
                    .frame(height: headerBadgeHeight)
                    .background(
                        Capsule(style: .continuous)
                            .fill(Color.green.opacity(0.12))
                    )
            }

            Text(appVersionText)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .padding(.horizontal, 7)
                .frame(height: headerBadgeHeight)
                .background(
                    Capsule(style: .continuous)
                        .fill(Color.primary.opacity(0.055))
                )
        }
        .contentShape(Capsule(style: .continuous))
    }
}

private struct SettingsSidebarBackIcon: View {
    var body: some View {
        SettingsSidebarBackIconShape()
            .stroke(
                style: StrokeStyle(
                    lineWidth: 1.35,
                    lineCap: .round,
                    lineJoin: .round,
                    miterLimit: 10
                )
            )
    }
}

private struct SettingsSidebarBackIconShape: Shape {
    func path(in rect: CGRect) -> Path {
        let scale = min(rect.width / 24, rect.height / 24)
        let xOffset = rect.midX - 12 * scale
        let yOffset = rect.midY - 12 * scale

        func point(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
            CGPoint(x: xOffset + x * scale, y: yOffset + y * scale)
        }

        var path = Path()
        path.move(to: point(7.12988, 18.3096))
        path.addLine(to: point(15.1299, 18.3096))
        path.addCurve(
            to: point(20.1299, 13.3096),
            control1: point(17.8899, 18.3096),
            control2: point(20.1299, 16.0696)
        )
        path.addCurve(
            to: point(15.1299, 8.30957),
            control1: point(20.1299, 10.5496),
            control2: point(17.8899, 8.30957)
        )
        path.addLine(to: point(4.12988, 8.30957))

        path.move(to: point(6.43012, 10.8104))
        path.addLine(to: point(3.87012, 8.25043))
        path.addLine(to: point(6.43012, 5.69043))

        return path
    }
}

private struct SettingsSidebarHeaderBackButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        SettingsSidebarHeaderBackButtonBody(configuration: configuration)
    }
}

private struct SettingsSidebarHeaderBackButtonBody: View {
    let configuration: SettingsSidebarHeaderBackButtonStyle.Configuration
    @State private var isHovered = false

    var body: some View {
        configuration.label
            .foregroundStyle(Color.secondary.opacity(configuration.isPressed ? 0.72 : 1))
            .background(
                Capsule(style: .continuous)
                    .fill(Color.primary.opacity(backgroundOpacity))
            )
            .contentShape(Capsule(style: .continuous))
            .onHover { isHovered = $0 }
    }

    private var backgroundOpacity: Double {
        if configuration.isPressed {
            return 0.11
        }
        if isHovered {
            return 0.075
        }
        return 0.055
    }
}

enum UpdateBadgeState: Equatable {
    case none
    case checkFailed(String)
    case newVersion(String?)
    case openingWindow(String?)

    var iconName: String {
        switch self {
        case .none:
            return "arrow.down.circle.fill"
        case .checkFailed:
            return "exclamationmark.triangle.fill"
        case .newVersion:
            return "arrow.down.circle.fill"
        case .openingWindow:
            return "arrow.down.circle.fill"
        }
    }

    var tintColor: Color {
        switch self {
        case .none:
            return .clear
        case .checkFailed:
            return .orange
        case .newVersion:
            return .green
        case .openingWindow:
            return .green
        }
    }

    var showsSpinner: Bool {
        switch self {
        case .openingWindow:
            return true
        case .none, .checkFailed, .newVersion:
            return false
        }
    }

    var isTriggerDisabled: Bool {
        switch self {
        case .openingWindow:
            return true
        case .none, .checkFailed, .newVersion:
            return false
        }
    }

    var title: String {
        switch self {
        case .none:
            return AppLocalization.localizedString("New Update")
        case .checkFailed:
            return AppLocalization.localizedString("Update Check Failed")
        case .newVersion:
            return AppLocalization.localizedString("New Update")
        case .openingWindow:
            return AppLocalization.localizedString("Opening…")
        }
    }
}
