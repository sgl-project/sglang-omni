import SwiftUI
import VoxtMobileKit

/// Whether the keyboard's handoff screen covers Voxt. It is for a keyboard that just
/// brought Voxt forward; the keyboard used inside Voxt itself must not be covered.
enum KeyboardHandoffCover {
    /// Voxt is "already in front" only after it has been active this long.
    static let settle: TimeInterval = 2

    /// `activeSince` is nil while Voxt is not active (a URL can arrive before the scene
    /// reports active, so that also counts as just brought forward).
    static func shows(activeSince: Date?, now: Date) -> Bool {
        guard let activeSince else { return true }
        return now.timeIntervalSince(activeSince) < settle
    }
}

/// Shown when the keyboard had to open Voxt to start the microphone. iOS gives no way
/// to return automatically, so it shows the one thing to do and how to skip this next time.
struct KeyboardHandoffView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            DictationMicButton(phase: model.phase, sessionActive: true, level: model.audioLevel, size: .app, startLabel: VoxtL10n.shared.string("Start recording"), stopLabel: VoxtL10n.shared.string("Stop recording")) {}
                .allowsHitTesting(false)
                .accessibilityIdentifier("keyboardHandoffMic")
            Text(VoxtL10n.shared.string("Voxt is listening"))
                .font(.title2.weight(.semibold))
                .multilineTextAlignment(.center)
            Spacer()
            if let window = model.floatingWindow { KeepReadyOffer(window: window) }
            Button(VoxtL10n.shared.string("Cancel")) { model.cancelComposition() }
                .buttonStyle(.plain)
                .foregroundStyle(VoxtTheme.Colors.ink)
                .padding(.horizontal, 40).padding(.vertical, 14)
                .background(VoxtTheme.Colors.keyGray, in: Capsule())
                .accessibilityIdentifier("keyboardHandoffCancel")
            SwipeBackHint()
        }
        .padding(.horizontal, 32)
        .padding(.top, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(VoxtTheme.Colors.screenBackground)
        .interactiveDismissDisabled()
    }
}

/// "Turn on Keep Voxt ready to skip this", with a button that turns it on here.
private struct KeepReadyOffer: View {
    @ObservedObject var window: FloatingDictationWindow

    var body: some View {
        if !window.isActive {
            VStack(spacing: 8) {
                Text(VoxtL10n.shared.string("Turn on Keep Voxt ready to skip this"))
                    .font(VoxtTheme.Typography.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Button(VoxtL10n.shared.string("Keep Voxt ready")) { window.start() }
                    .font(VoxtTheme.Typography.caption.weight(.semibold))
                    .accessibilityIdentifier("keyboardHandoffKeepReady")
            }
        }
    }
}

/// A dot sliding right along the bottom edge, where the system back swipe starts.
private struct SwipeBackHint: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var moving = false

    var body: some View {
        VStack(spacing: 10) {
            Text(VoxtL10n.shared.string("Swipe right along the bottom to go back"))
                .font(VoxtTheme.Typography.caption)
                .foregroundStyle(VoxtTheme.Colors.ink)
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(VoxtTheme.Colors.keyGray).frame(height: 4)
                    Image(systemName: "arrow.right.circle.fill")
                        .font(.system(size: 22))
                        .foregroundStyle(VoxtTheme.Colors.ink)
                        .offset(x: moving ? geometry.size.width - 22 : 0)
                }
                .frame(maxHeight: .infinity)
            }
            .frame(height: 26)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(VoxtL10n.shared.string("Swipe right along the bottom to go back"))
        .accessibilityIdentifier("keyboardHandoffSwipeHint")
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 1.4).repeatForever(autoreverses: false)) { moving = true }
        }
    }
}
