import SwiftUI
import UIKit
import VoxtMobileKit

/// First open, one step per screen. The model decides which step shows.
struct OnboardingView: View {
    @Environment(AppModel.self) private var model
    let step: OnboardingProgress.Step

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        VStack(alignment: .leading, spacing: 0) {
            if step != .welcome {
                OnboardingProgressBar(filled: step.rawValue + 1)
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
            }
            switch step {
            case .welcome: WelcomeStep()
            case .connect: ConnectMacView()
            case .addKeyboard: AddKeyboardStep()
            case .fullAccess: FullAccessStep()
            case .tryIt: TryItStep()
            }
        }
        .background(VoxtTheme.Colors.screenBackground)
        .foregroundStyle(VoxtTheme.Colors.ink)
        .animation(.default, value: step)
    }
}

private struct OnboardingProgressBar: View {
    let filled: Int
    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<5, id: \.self) { index in
                Capsule()
                    .fill(index < filled ? VoxtTheme.Colors.ink : VoxtTheme.Colors.keyGray.opacity(0.6))
                    .frame(height: 4)
            }
        }
        .accessibilityElement()
        .accessibilityLabel(VoxtL10n.shared.format("Step %d of 5", filled))
    }
}

/// Title, body and one black pill at the bottom: the frame every step shares.
struct OnboardingPage<Content: View>: View {
    let title: String
    var subtitle: String?
    var button: String?
    var buttonEnabled = true
    var buttonID = "onboardingNext"
    var action: () -> Void = {}
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text(VoxtL10n.shared.string(title))
                        .font(.system(size: 34, weight: .bold))
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityAddTraits(.isHeader)
                        .accessibilityIdentifier("onboardingTitle")
                    if let subtitle {
                        Text(VoxtL10n.shared.string(subtitle))
                            .font(.system(size: 17))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    content
                }
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .padding(.bottom, 24)
            }
            if let button {
                Button(VoxtL10n.shared.string(button), action: action)
                    .buttonStyle(BlackPillButtonStyle())
                    .disabled(!buttonEnabled)
                    .accessibilityIdentifier(buttonID)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
            }
        }
    }
}

// MARK: Welcome

private struct WelcomeStep: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        OnboardingPage(title: "Say anything. Only your Mac hears it.", button: "Get started", action: model.advanceOnboarding) {
            VStack(alignment: .leading, spacing: 14) {
                Text(VoxtL10n.shared.string("You say"))
                    .font(.system(size: 15, weight: .semibold)).foregroundStyle(.secondary)
                Text(VoxtL10n.shared.string("“hi karen um the heater in 4B stopped working on tuesday, uh, can you send someone this week? thanks”"))
                    .foregroundStyle(.secondary)
                HStack(spacing: 12) {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(VoxtTheme.Colors.onInk)
                        .frame(width: 44, height: 28)
                        .background(VoxtTheme.Colors.ink, in: Capsule())
                    Rectangle().fill(VoxtTheme.Colors.keyGray).frame(height: 1)
                }
                .accessibilityHidden(true)
                Text(VoxtL10n.shared.string("Voxt types"))
                    .font(.system(size: 15, weight: .semibold)).foregroundStyle(.secondary)
                Text(VoxtL10n.shared.string("Hi Karen,\n\nThe heater in Unit 4B stopped working on Tuesday. Could you send someone this week?\n\nThanks"))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .font(.system(size: 17))
            .frame(maxWidth: .infinity, alignment: .leading)
            .voxtCard()
        }
    }
}

// MARK: Add the keyboard

private struct AddKeyboardStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var tapStage = 0

    var body: some View {
        OnboardingPage(title: "Add the Voxt keyboard", button: "Continue", action: model.advanceOnboarding) {
            Text(VoxtL10n.shared.string("Settings opens on Voxt. Tap Keyboards, then turn on Voxt."))
                .font(.system(size: 17))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("addKeyboardPath")
            HStack(alignment: .top, spacing: 10) {
                FakeSettingsPage(back: "Apps", title: "Voxt", header: "ALLOW VOXT TO ACCESS") {
                    FakeSettingsRow(icon: .orange, title: "Microphone")
                    FakeSettingsRow(icon: .gray, title: "Camera")
                    FakeSettingsRow(icon: .indigo, title: "Siri", chevron: true)
                    FakeSettingsRow(icon: .gray, title: "Keyboards", chevron: true, highlighted: tapStage == 0, ring: tapStage == 0)
                }
                FakeSettingsPage(back: "Voxt", title: "Keyboards") {
                    FakeSettingsRow(title: "Voxt", toggle: tapStage >= 1, highlighted: tapStage == 1, ring: tapStage == 1)
                    FakeSettingsRow(title: "Allow Full Access", toggle: false)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(VoxtL10n.shared.string("In Settings, tap Keyboards, then turn on Voxt."))
            Text(VoxtL10n.shared.string("The picture plays these taps in a loop."))
                .font(VoxtTheme.Typography.caption)
                .foregroundStyle(.secondary)
        }
        .task {
            guard !reduceMotion else { tapStage = 1; return }
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1.4))
                withAnimation(.easeInOut(duration: 0.3)) { tapStage = (tapStage + 1) % 3 }
            }
        }
    }
}

// MARK: Full Access

private struct FullAccessStep: View {
    @Environment(AppModel.self) private var model
    @State private var explained = false

    var body: some View {
        OnboardingPage(title: "Then allow Full Access", button: "Open Settings", action: openSettings) {
            Text(VoxtL10n.shared.string("The keyboard needs it to talk to the Voxt app on this phone. Voxt doesn't keep what you type. When you ask it to edit text, that text goes only to your Mac."))
                .font(.system(size: 17))
                .fixedSize(horizontal: false, vertical: true)
            if explained {
                Text(VoxtL10n.shared.string("iOS asks this of every keyboard that works with its own app. Voxt's keyboard uses it only to pass your recording to the Voxt app and to get the text back."))
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .voxtCard()
                    .accessibilityIdentifier("fullAccessExplained")
            } else {
                Button(VoxtL10n.shared.string("What's Full Access?")) { withAnimation { explained = true } }
                    .foregroundStyle(VoxtTheme.Colors.accent)
                    .accessibilityIdentifier("whatsFullAccess")
            }
            ZStack {
                FakeSettingsPage(back: "Voxt", title: "Keyboards") {
                    FakeSettingsRow(title: "Voxt", toggle: true)
                    FakeSettingsRow(title: "Allow Full Access", toggle: true, highlighted: true, ring: true)
                    Color.clear.frame(height: 150)
                }
                .overlay(Color.black.opacity(0.25).clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous)))
                VStack(spacing: 10) {
                    Text(VoxtL10n.shared.string("Allow Full Access for “Voxt” Keyboards?"))
                        .font(.system(size: 17, weight: .semibold))
                        .multilineTextAlignment(.center)
                    Text(VoxtL10n.shared.string("iOS shows this warning for every keyboard with Full Access."))
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                    HStack(spacing: 10) {
                        Text(VoxtL10n.shared.string("Don't Allow"))
                            .frame(maxWidth: .infinity).padding(.vertical, 10)
                            .background(VoxtTheme.Colors.keyGray, in: Capsule())
                        Text(VoxtL10n.shared.string("Allow"))
                            .frame(maxWidth: .infinity).padding(.vertical, 10)
                            .background(VoxtTheme.Colors.keyGray, in: Capsule())
                            .overlay(Capsule().strokeBorder(VoxtTheme.Colors.accent, lineWidth: 3))
                    }
                    .font(.system(size: 16, weight: .semibold))
                }
                .padding(16)
                .frame(maxWidth: 270)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .offset(y: 55)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(VoxtL10n.shared.string("In Keyboards, turn on Allow Full Access, then tap Allow."))
        }
    }

    private func openSettings() {
        model.advanceOnboarding()
        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
    }
}

// MARK: Try it

private struct TryItStep: View {
    @Environment(AppModel.self) private var model
    @State private var text = ""
    @State private var progress = TryItProgress(startHistoryCount: 0)
    @State private var keyboardEnabled = false
    @FocusState private var focused: Bool
    private let defaults = UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { context in
            let card = progress.card(keyboardEnabled: keyboardEnabled, status: keyboardStatus(),
                                     historyCount: model.history.count, now: context.date)
            OnboardingPage(title: "Try it") {
                TextField(VoxtL10n.shared.string("Say anything"), text: $text, axis: .vertical)
                    .lineLimit(2...5)
                    .focused($focused)
                    .voxtCard()
                    .accessibilityIdentifier("tryItField")
                cardView(card)
                if card != .done {
                    Button(VoxtL10n.shared.string("Skip for now")) { model.finishOnboarding() }
                        .font(VoxtTheme.Typography.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)
                        .accessibilityIdentifier("onboardingSkip")
                }
            }
            .safeAreaInset(edge: .bottom) {
                if card == .ready {
                    Text(VoxtL10n.shared.string("Tap and say anything"))
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(VoxtTheme.Colors.onInk)
                        .padding(.horizontal, 18).padding(.vertical, 10)
                        .background(VoxtTheme.Colors.ink, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .frame(maxWidth: .infinity)
                        .padding(.bottom, 6)
                }
            }
        }
        .onAppear(perform: start)
        .onReceive(NotificationCenter.default.publisher(for: UITextInputMode.currentInputModeDidChangeNotification)) { _ in
            progress.keyboardChanged(at: Date())
        }
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillShowNotification)) { _ in
            progress.keyboardChanged(at: Date())
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            keyboardEnabled = Self.voxtKeyboardEnabled
        }
    }

    @ViewBuilder private func cardView(_ card: TryItProgress.Card) -> some View {
        switch card {
        case .switchKeyboard:
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: "globe").font(.system(size: 20)).accessibilityHidden(true)
                Text(VoxtL10n.shared.string("Hold the globe key and choose Voxt."))
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("tryItHint")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .voxtCard()
        case .fullAccessOff:
            VStack(alignment: .leading, spacing: 10) {
                Text(VoxtL10n.shared.string("Allow Full Access is off")).font(.system(size: 19, weight: .semibold))
                Text(VoxtL10n.shared.string("Voxt is added, but the keyboard can't reach the Voxt app without Full Access."))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button(VoxtL10n.shared.string("Open Settings")) {
                    if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                }
                .buttonStyle(BlackPillButtonStyle())
                .accessibilityIdentifier("tryItOpenSettings")
            }
            .voxtCard()
        case .ready:
            EmptyView()
        case .done:
            VStack(alignment: .leading, spacing: 14) {
                Text(VoxtL10n.shared.string("That's it. Use it in any app.")).font(.system(size: 19, weight: .semibold))
                Button(VoxtL10n.shared.string("Done")) { model.finishOnboarding() }
                    .buttonStyle(BlackPillButtonStyle())
                    .accessibilityIdentifier("onboardingNext")
            }
            .voxtCard()
        }
    }

    private func start() {
        progress = TryItProgress(startHistoryCount: model.history.count)
        keyboardEnabled = Self.voxtKeyboardEnabled
        focused = true
        #if DEBUG
        // UI tests can't switch the simulator's keyboard; they name the keyboard that is "up".
        switch ProcessInfo.processInfo.environment["VOXT_TEST_TRY_KEYBOARD"] {
        case "noFullAccess":
            keyboardEnabled = true
            progress.keyboardChanged(at: Date())
        case "voxt":
            keyboardEnabled = true
            progress.keyboardChanged(at: Date())
            KeyboardStatus(hasFullAccess: true, updatedAt: Date()).save(to: defaults)
        default: break
        }
        #endif
    }

    private func keyboardStatus() -> KeyboardStatus? {
        #if DEBUG
        // A test that names the keyboard state ignores the simulator's real keyboard, which may
        // have Full Access from an earlier end-to-end run.
        if ProcessInfo.processInfo.environment["VOXT_TEST_TRY_KEYBOARD"] == "noFullAccess" { return nil }
        #endif
        return KeyboardStatus.load(from: defaults)
    }

    /// AppleKeyboards lists enabled keyboards by their extension bundle ID.
    private static var voxtKeyboardEnabled: Bool {
        #if DEBUG
        if ProcessInfo.processInfo.environment["VOXT_TEST_TRY_KEYBOARD"] != nil { return true }
        #endif
        guard let app = Bundle.main.bundleIdentifier else { return false }
        return UserDefaults.standard.stringArray(forKey: "AppleKeyboards")?.contains(app + ".keyboard") == true
    }
}

// MARK: Pictures of the Settings app

private struct FakeSettingsPage<Rows: View>: View {
    let back: String
    let title: String
    var header: String?
    @ViewBuilder var rows: Rows

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Text("‹ " + VoxtL10n.shared.string(back)).foregroundStyle(Color.blue)
                Text(VoxtL10n.shared.string(title)).fontWeight(.semibold)
            }
            .font(.system(size: 14))
            .lineLimit(1)
            if let header {
                Text(VoxtL10n.shared.string(header)).font(.system(size: 11)).foregroundStyle(.secondary)
                    .lineLimit(2).minimumScaleFactor(0.7)
            }
            VStack(spacing: 0) { rows }
                .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Color(.systemGroupedBackground), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
    }
}

private struct FakeSettingsRow: View {
    var icon: Color?
    let title: String
    var chevron = false
    var toggle: Bool?
    var highlighted = false
    var ring = false

    var body: some View {
        HStack(spacing: 8) {
            if let icon { RoundedRectangle(cornerRadius: 5).fill(icon).frame(width: 20, height: 20) }
            // Narrow on an iPhone SE, and longer in Spanish or Japanese: shrink before wrapping
            // mid-word or cutting off.
            Text(VoxtL10n.shared.string(title)).font(.system(size: 14))
                .lineLimit(toggle == nil ? 1 : 3)
                .minimumScaleFactor(0.6)
                .allowsTightening(true)
            Spacer(minLength: 4)
            if let toggle {
                Capsule().fill(toggle ? Color.green : Color(.systemGray4))
                    .frame(width: 38, height: 23)
                    .overlay(alignment: toggle ? .trailing : .leading) {
                        Circle().fill(.white).padding(2)
                            .overlay { if ring { Circle().strokeBorder(VoxtTheme.Colors.accent, lineWidth: 2.5).padding(-3) } }
                    }
            }
            if chevron {
                ZStack {
                    Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold)).foregroundStyle(.tertiary)
                    if ring {
                        Circle().fill(VoxtTheme.Colors.accent.opacity(0.25)).frame(width: 30, height: 30)
                            .overlay(Circle().strokeBorder(VoxtTheme.Colors.accent, lineWidth: 2.5))
                            .offset(x: -18)
                    }
                }
            }
        }
        .padding(.horizontal, 10)
        .frame(minHeight: 40)
        .background(highlighted ? VoxtTheme.Colors.accent.opacity(0.12) : Color.clear)
    }
}
