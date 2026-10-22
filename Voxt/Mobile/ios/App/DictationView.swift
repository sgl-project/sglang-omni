import SwiftUI
import UIKit
import AVFoundation
import VoxtMobileKit

/// Home: the Mac, "Keep Voxt ready", recent dictations. No numbers until
/// there is something to count.
struct DictationView: View {
    @Environment(AppModel.self) private var model
    var onSeeAll: () -> Void = {}
    @State private var showsSettings = false
    @State private var practiceText = ""
    @FocusState private var practiceFocused: Bool

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    #if DEBUG
                    if ProcessInfo.processInfo.environment["VOXT_TEST_PRACTICE_FIELD"] == "1" {
                        // UI tests type into this with the Voxt keyboard; never shown otherwise.
                        TextField("Practice", text: $practiceText, axis: .vertical)
                            .lineLimit(3...6)
                            .accessibilityIdentifier("tryField")
                            .focused($practiceFocused)
                            .voxtCard()
                    }
                    #endif
                    ConnectionCard(onOpen: { showsSettings = true })
                    if let window = model.floatingWindow { KeepVoxtReadyCard(window: window) }
                    recent
                }
                .padding(16)
                .padding(.bottom, 120)
            }
            .dictationFeedback(phase: model.phase)
            .background(VoxtTheme.Colors.screenBackground)
            .navigationTitle(VoxtL10n.shared.string("Voxt"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(VoxtTheme.Colors.screenBackground, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(VoxtL10n.shared.string("Settings"), systemImage: "gearshape") { showsSettings = true }
                        .accessibilityIdentifier("openSettings")
                }
            }
            // Only the UI-test practice field can be focused on Home; keyboard tests close
            // the keyboard with this bar (DictationFlowUITests).
            .safeAreaInset(edge: .bottom) {
                if practiceFocused {
                    HStack {
                        Spacer()
                        Button(VoxtL10n.shared.string("Hide keyboard")) { practiceFocused = false }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.regularMaterial)
                }
            }
            .sheet(isPresented: $showsSettings) {
                SettingsView()
            }
            // voxtmobile://settings, from the Voxt name at the keyboard's top left.
            .onChange(of: model.showsSettings, initial: true) { _, show in
                guard show else { return }
                model.showsSettings = false
                showsSettings = true
            }
        }
    }

    @ViewBuilder private var recent: some View {
        let items = Array(model.history.prefix(3))
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(VoxtL10n.shared.string("Recent")).font(.title3.weight(.semibold))
                Spacer()
                if !items.isEmpty {
                    Button(VoxtL10n.shared.string("See all"), action: onSeeAll)
                        .foregroundStyle(VoxtTheme.Colors.accent)
                        .accessibilityIdentifier("seeAllHistory")
                }
            }
            if let week = WeekSummary(records: model.history) {
                Text(VoxtL10n.shared.format("This week: %@ on %@",
                                            VoxtL10n.shared.plural("%d words", count: week.words),
                                            VoxtL10n.shared.plural("%d days", count: week.days)))
                    .font(VoxtTheme.Typography.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("weekSummary")
            }
        }
        .padding(.top, 8)
        .padding(.horizontal, 4)
        if items.isEmpty {
            Text(VoxtL10n.shared.string("Your dictations show up here"))
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .voxtCard()
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(items.enumerated()), id: \.element.id) { index, item in
                    if index > 0 { Divider().padding(.vertical, 14) }
                    HStack(alignment: .top, spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(item.text).lineLimit(2).foregroundStyle(VoxtTheme.Colors.ink)
                            Text(Self.when(item.date))
                                .font(VoxtTheme.Typography.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer(minLength: 0)
                        Button { UIPasteboard.general.string = item.text } label: {
                            Image(systemName: "doc.on.doc").foregroundStyle(.secondary)
                        }
                        .accessibilityLabel(VoxtL10n.shared.string("Copy"))
                        .accessibilityIdentifier("copyRecent-\(index)")
                    }
                }
            }
            .voxtCard()
        }
    }

    /// "2 min ago" within the hour, the time today, "Yesterday", then the date.
    static func when(_ date: Date, now: Date = Date(), calendar: Calendar = .current) -> String {
        let locale = VoxtL10n.shared.locale
        if now.timeIntervalSince(date) < 3_600 {
            let formatter = RelativeDateTimeFormatter()
            formatter.locale = locale
            formatter.unitsStyle = .short
            // Under a minute (or a hair in the future): "now", never "in 0 sec.".
            if now.timeIntervalSince(date) < 60 {
                formatter.dateTimeStyle = .named
                return formatter.localizedString(fromTimeInterval: 0)
            }
            return formatter.localizedString(for: max(date, now.addingTimeInterval(-3_599)), relativeTo: now)
        }
        if calendar.isDate(date, inSameDayAs: now) { return date.formatted(.dateTime.hour().minute().locale(locale)) }
        if calendar.isDateInYesterday(date) { return VoxtL10n.shared.string("Yesterday") }
        return date.formatted(.dateTime.month().day().locale(locale))
    }
}

/// Home's "Keep Voxt ready": the Picture in Picture window that keeps the app awake, so the
/// keyboard can start the mic without switching apps.
struct KeepVoxtReadyCard: View {
    @ObservedObject var window: FloatingDictationWindow

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        VStack(alignment: .leading, spacing: 10) {
            Toggle(isOn: Binding(get: { window.isActive }, set: { $0 ? window.start() : window.stop() })) {
                Label(VoxtL10n.shared.string("Keep Voxt ready"), systemImage: "pip")
                    .font(VoxtTheme.Typography.cardTitle)
            }
            .tint(.green)
            .disabled(!window.isAvailable)
            .accessibilityIdentifier("keepVoxtReady")
            Text(VoxtL10n.shared.string("The keyboard won't switch apps to start the mic. Voxt waits in a small floating window; the mic turns on only when you tap it."))
                .font(VoxtTheme.Typography.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if !window.isAvailable {
                Text(VoxtL10n.shared.string("Picture in Picture is unavailable on this device."))
                    .font(VoxtTheme.Typography.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("floatingWindowUnavailable")
            } else if let error = window.lastError {
                Text(VoxtL10n.shared.string(error)).font(VoxtTheme.Typography.caption).foregroundStyle(.red)
            }
        }
        .foregroundStyle(VoxtTheme.Colors.ink)
        .voxtCard()
    }
}

/// Keeps the floating window's PiP source layer in the app's window for as long as the app
/// runs. Invisible (UIPiPView does the same with alpha 0) and outside any sheet.
struct PiPSourceHost: UIViewRepresentable {
    let layer: AVSampleBufferDisplayLayer
    func makeUIView(context: Context) -> FloatingWindowPreview.PreviewView {
        let view = FloatingWindowPreview.PreviewView(displayLayer: layer)
        view.alpha = 0
        view.isUserInteractionEnabled = false
        view.accessibilityElementsHidden = true
        return view
    }
    func updateUIView(_ uiView: FloatingWindowPreview.PreviewView, context: Context) {}
}

struct FloatingWindowPreview: UIViewRepresentable {
    let layer: AVSampleBufferDisplayLayer
    func makeUIView(context: Context) -> PreviewView { PreviewView(displayLayer: layer) }
    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        let displayLayer: AVSampleBufferDisplayLayer
        init(displayLayer: AVSampleBufferDisplayLayer) {
            self.displayLayer = displayLayer
            super.init(frame: .zero)
            layer.addSublayer(displayLayer)
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
        override func layoutSubviews() {
            super.layoutSubviews()
            displayLayer.frame = bounds
        }
    }
}

private struct ConnectionCard: View {
    @Environment(AppModel.self) private var model
    var onOpen: (() -> Void)? = nil

    var body: some View {
        if let state = AppModel.statusCardState(for: model.connection) {
            MacStatusCardView(card: MacStatusCard(macName: model.server?.displayName, state: state),
                              onRetry: { Task { await model.checkConnection() } }, onOpen: onOpen)
        } else {
            HStack(spacing: 10) {
                Image(systemName: "laptopcomputer.slash").foregroundStyle(.secondary)
                Text(VoxtL10n.shared.string("Not paired")).foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            .voxtCard()
        }
    }
}

/// Which Mac, and its state: a Mac glyph, the name as the headline, then a state dot and
/// one line ("Connected · Voxt 1.16"). Read by VoiceOver as one element.
struct MacStatusCardView: View {
    let card: MacStatusCard
    var onRetry: () -> Void = {}
    var onOpen: (() -> Void)? = nil
    /// nil: "connectedLabel" while connected (UI tests wait on it), else "connectionStatus".
    var identifier: String? = nil
    /// false inside a section that is already a card (Settings).
    var framed = true

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""
    @Environment(\.dynamicTypeSize) private var typeSize
    /// Half the subheadline's x-height: the dot and the spinner sit in the middle of the
    /// first line of the status text at every text size.
    @ScaledMetric(relativeTo: .subheadline) private var markLift: CGFloat = 4.5

    var body: some View {
        let _ = interfaceLanguage
        HStack(alignment: .top, spacing: 12) {
            Button { onOpen?() } label: {
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: card.symbol)
                        .font(.system(size: 28))
                        .foregroundStyle(.secondary)
                        .frame(width: 44, height: 44)
                        .background(Color(.tertiarySystemFill), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    VStack(alignment: .leading, spacing: 3) {
                        Text(card.title)
                            .font(.headline)
                            .foregroundStyle(VoxtTheme.Colors.ink)
                            .lineLimit(typeSize.isAccessibilitySize ? 2 : 1)
                            .truncationMode(.middle)
                        HStack(alignment: .firstTextBaseline, spacing: 6) {
                            if let dot = card.dot {
                                Circle().fill(dot == .green ? Color.green : Color.orange)
                                    .frame(width: 8, height: 8)
                                    // On the first line of the status text, at every text size.
                                    .alignmentGuide(.firstTextBaseline) { $0[VerticalAlignment.center] + markLift }
                            } else {
                                ProgressView().controlSize(.mini)
                                    .alignmentGuide(.firstTextBaseline) { $0[VerticalAlignment.center] + markLift }
                            }
                            // Wraps instead of cutting off: Spanish takes three lines beside Retry
                            // on an iPhone SE.
                            Text(card.detail)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        if let reason = card.reason {
                            Text(reason)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .padding(.top, 2)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            // Not .disabled: that greys the card out where it only shows status (Settings).
            .allowsHitTesting(onOpen != nil)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(card.accessibilityLabel)
            .accessibilityHint(card.reason ?? "")
            .accessibilityAddTraits(onOpen == nil ? [] : .isButton)
            // UI tests wait for "connectedLabel" to mean connected.
            .accessibilityIdentifier(identifier ?? (card.dot == .green ? "connectedLabel" : "connectionStatus"))
            if card.showsRetry {
                Button(VoxtL10n.shared.string("Retry"), action: onRetry)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .tint(.primary)
                    .padding(.top, 2)
                    .accessibilityIdentifier("connectionRetry")
            }
        }
        .modifier(CardFrame(framed: framed))
    }
}

private struct CardFrame: ViewModifier {
    let framed: Bool
    func body(content: Content) -> some View {
        if framed { content.voxtCard() } else { content }
    }
}

#if DEBUG
/// VOXT_TEST_STATUS_GALLERY=1: every status card state on one screen, for design review
/// screenshots (VOXT_TEST_APPEARANCE=dark|light). Fake Mac names only.
struct StatusCardGallery: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                MacStatusCardView(card: .init(macName: "Studio Mac mini", state: .connected(version: "1.16.0")), onOpen: {})
                MacStatusCardView(card: .init(macName: "Studio Mac mini", state: .connecting), onOpen: {})
                MacStatusCardView(card: .init(macName: "Studio Mac mini", state: .unreachable(reason: nil)), onOpen: {})
                MacStatusCardView(card: .init(macName: "Studio Mac mini", state: .unreachable(
                    reason: "The connection to your Mac is not ready. Reconnect and try again.")), onOpen: {})
                MacStatusCardView(card: .init(macName: "Studio Mac mini", state: .cloudModel(version: "1.16.0")), onOpen: {})
                MacStatusCardView(card: .init(macName: "Someone's Extremely Long Office MacBook Pro", state: .connected(version: "1.16.0")), onOpen: {})
                MacStatusCardView(card: .init(macName: nil, state: .connected(version: "1.16.0")), onOpen: {})
                VStack(alignment: .leading, spacing: 8) {
                    Text(VoxtL10n.shared.string("Mac")).font(VoxtTheme.Typography.cardTitle)
                    MacStatusCardView(card: .init(macName: "Office Mac Studio", state: .connected(version: "1.16.0")), framed: false)
                }
                .voxtCard()
                MacStatusCardView(card: .init(macName: "Studio Mac mini", state: .connected(version: "1.16.0")), onOpen: {})
                    .dynamicTypeSize(.accessibility3)
            }
            .padding(16)
            .padding(.top, 44)
        }
        .background(VoxtTheme.Colors.screenBackground)
    }
}
#endif
