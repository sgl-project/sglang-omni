// RecordingOverlayWindow.swift
// Provides Recording Overlay Window for window and overlay UI.

import AppKit
import SwiftUI
import Combine

/// A borderless, non-activating floating panel that sits at the bottom-center
/// of the main screen and hosts the WaveformView.
class RecordingOverlayWindow: NSPanel {

    private var hostingView: NSHostingView<OverlayContent>?
    private var visibilityToken: UInt64 = 0
    private var appearanceStateCancellable: AnyCancellable?
    private var overlayAppearanceCancellable: AnyCancellable?
    private weak var observedState: OverlayState?
    private var currentPosition: OverlayPosition = .bottom
    var onRequestClose: (() -> Void)?
    var onRequestInject: (() -> Void)?
    var onRequestContinue: (() -> Void)?
    var onRequestConversationRecordToggle: (() -> Void)?
    var onRequestDetail: (() -> Void)?
    var onRequestSessionTranslationTargetPickerToggle: (() -> Void)?
    var onRequestSessionTranslationTargetLanguageSelect: ((TranslationTargetLanguage) -> Void)?
    var onRequestSessionTranslationTargetPickerDismiss: (() -> Void)?

    init() {
        super.init(
            contentRect: .zero,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        level = .floating
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false
        isMovableByWindowBackground = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        ignoresMouseEvents = true

        overlayAppearanceCancellable = NotificationCenter.default.publisher(for: .voxtOverlayAppearanceDidChange)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                guard let self, let state = self.observedState else { return }
                let raw = UserDefaults.standard.string(forKey: AppPreferenceKey.overlayPosition) ?? OverlayPosition.bottom.rawValue
                self.currentPosition = OverlayPosition(rawValue: raw) ?? .bottom
                self.updateAppearance(for: state, animated: self.isVisible)
            }
    }

    override var canBecomeKey: Bool { true }

    func show(state: OverlayState, position: OverlayPosition) {
        VoxtLog.info(
            "Overlay show requested. wasVisible=\(isVisible), displayMode=\(state.displayMode), position=\(position.rawValue)",
            verbose: true
        )
        visibilityToken &+= 1
        currentPosition = position
        state.isPresented = true

        let content = OverlayContent(
            state: state,
            onInject: { [weak self] in self?.onRequestInject?() },
            onContinue: { [weak self] in self?.onRequestContinue?() },
            onToggleConversationRecording: { [weak self] in self?.onRequestConversationRecordToggle?() },
            onShowDetail: { [weak self] in self?.onRequestDetail?() },
            onClose: { [weak self] in self?.onRequestClose?() },
            onToggleSessionTranslationTargetPicker: { [weak self] in
                self?.onRequestSessionTranslationTargetPickerToggle?()
            },
            onSelectSessionTranslationTargetLanguage: { [weak self] language in
                self?.onRequestSessionTranslationTargetLanguageSelect?(language)
            },
            onDismissSessionTranslationTargetPicker: { [weak self] in
                self?.onRequestSessionTranslationTargetPickerDismiss?()
            }
        )

        if let hostingView {
            hostingView.rootView = content
        } else {
            let hosting = NSHostingView(rootView: content)
            hosting.translatesAutoresizingMaskIntoConstraints = true
            hosting.autoresizingMask = [.width, .height]
            contentView = hosting
            self.hostingView = hosting
        }

        observe(state: state)
        updateAppearance(for: state, animated: isVisible)
        DispatchQueue.main.async { [weak self] in
            self?.hostingView?.needsLayout = true
            self?.contentView?.needsLayout = true
        }

        if !isVisible {
            alphaValue = 1
            orderFrontRegardless()
        }
    }

    @discardableResult
    func handleAnswerSpaceShortcut() -> Bool {
        guard let state = observedState,
              let action = state.answerSpaceShortcutAction
        else {
            return false
        }

        switch action {
        case .continueAndRecord:
            onRequestContinue?()
        case .toggleConversationRecording:
            onRequestConversationRecordToggle?()
        }
        return true
    }

    func hide(animated: Bool = true, completion: (() -> Void)? = nil) {
        VoxtLog.info("Overlay hide requested. isVisible=\(isVisible)", verbose: true)
        observedState?.isPresented = false
        observedState?.audioLevel = 0
        observedState?.clearSessionTranslationLanguageHover()

        guard isVisible else {
            orderOut(nil)
            completion?()
            return
        }

        guard animated else {
            alphaValue = 0
            orderOut(nil)
            completion?()
            return
        }

        let token = visibilityToken
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.3
            animator().alphaValue = 0
        }, completionHandler: { [weak self] in
            guard let self else { return }
            guard token == self.visibilityToken else { return }
            self.orderOut(nil)
            completion?()
        })
    }

    private func observe(state: OverlayState) {
        guard observedState !== state else { return }
        observedState = state
        appearanceStateCancellable = Publishers.CombineLatest4(
            state.$displayMode,
            state.$allowsSessionTranslationLanguageSwitching,
            state.$isSessionTranslationTargetPickerPresented,
            state.$answerInteractionMode
        )
        .receive(on: RunLoop.main)
        .sink { [weak self, weak state] _ in
            guard let self, let state else { return }
            self.updateAppearance(for: state, animated: true)
        }
    }

    private func updateAppearance(for state: OverlayState, animated _: Bool) {
        updateMouseInteraction(for: state)
        let targetFrame = frame(for: panelSize(for: state), position: currentPosition)

        guard !targetFrame.isEmpty else { return }
        guard !frame.isApproximatelyEqual(to: targetFrame) else {
            return
        }
        // Keep window geometry changes outside AppKit's animation/layout transaction.
        // macOS 26 can re-enter SwiftUI DesignLibrary layout while the overlay view is
        // rebuilding after a hotkey transition, which has produced main-thread crashes.
        setFrame(targetFrame, display: true)
    }

    private func panelSize(for state: OverlayState) -> CGSize {
        switch state.displayMode {
        case .recording, .processing:
            let allowsRealtimeText = UserDefaults.standard.object(forKey: AppPreferenceKey.realtimeTextDisplayEnabled) as? Bool ?? true
            let width: CGFloat = allowsRealtimeText ? 360 : 220
            return CGSize(width: width, height: 140)
        case .answer:
            // Keep the answer window size stable while the translation language picker
            // opens, otherwise AppKit re-lays out the hosting window mid-update and the
            // whole overlay appears to jump upward.
            return CGSize(width: 560, height: 340)
        }
    }

    private func frame(for size: CGSize, position: OverlayPosition) -> CGRect {
        let fixedEdgeDistance = overlayScreenEdgeInset
        let visibleFrame = NSScreen.main?.visibleFrame ?? .zero
        guard !visibleFrame.isEmpty else {
            return CGRect(origin: frame.origin, size: size)
        }

        let x = visibleFrame.midX - size.width / 2
        let y: CGFloat
        switch position {
        case .bottom:
            y = visibleFrame.minY + fixedEdgeDistance
        case .top:
            y = visibleFrame.maxY - size.height - fixedEdgeDistance
        }
        return CGRect(origin: CGPoint(x: x, y: y), size: size)
    }

    private var overlayScreenEdgeInset: CGFloat {
        let storedValue = UserDefaults.standard.object(forKey: AppPreferenceKey.overlayScreenEdgeInset) as? Int ?? 30
        return CGFloat(min(max(storedValue, 0), 120))
    }

    private func updateMouseInteraction(for state: OverlayState) {
        ignoresMouseEvents = !(
            state.displayMode == .answer ||
            (state.displayMode == .recording && state.allowsSessionTranslationLanguageSwitching)
        )
    }
}

enum FloatingToastKind {
    case success
    case warning

    var systemImageName: String {
        switch self {
        case .success:
            return "checkmark.circle.fill"
        case .warning:
            return "exclamationmark.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .success:
            return Color(nsColor: .systemGreen)
        case .warning:
            return Color(nsColor: .systemOrange)
        }
    }
}

class FloatingToastWindow: NSPanel {
    private var hostingView: NSHostingView<FloatingToastContent>?
    private let panelSize = CGSize(width: 240, height: 52)

    init() {
        super.init(
            contentRect: .zero,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        level = .floating
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false
        isMovableByWindowBackground = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        ignoresMouseEvents = true
    }

    override var canBecomeKey: Bool { false }

    func show(message: String, kind: FloatingToastKind, position: OverlayPosition) {
        let content = FloatingToastContent(message: message, kind: kind)
        if let hostingView {
            hostingView.rootView = content
        } else {
            let hosting = NSHostingView(rootView: content)
            hosting.translatesAutoresizingMaskIntoConstraints = true
            hosting.autoresizingMask = [.width, .height]
            contentView = hosting
            hostingView = hosting
        }

        setFrame(frame(for: panelSize, position: position), display: true)
        alphaValue = 1
        orderFrontRegardless()
    }

    func hide() {
        orderOut(nil)
    }

    private func frame(for size: CGSize, position: OverlayPosition) -> CGRect {
        let visibleFrame = NSScreen.main?.visibleFrame ?? .zero
        guard !visibleFrame.isEmpty else {
            return CGRect(origin: frame.origin, size: size)
        }

        let edgeInset = overlayScreenEdgeInset
        let x = visibleFrame.midX - size.width / 2
        let y: CGFloat
        switch position {
        case .bottom:
            y = visibleFrame.minY + edgeInset
        case .top:
            y = visibleFrame.maxY - size.height - edgeInset
        }
        return CGRect(origin: CGPoint(x: x, y: y), size: size)
    }

    private var overlayScreenEdgeInset: CGFloat {
        let storedValue = UserDefaults.standard.object(forKey: AppPreferenceKey.overlayScreenEdgeInset) as? Int ?? 30
        return CGFloat(min(max(storedValue, 0), 120))
    }
}

private struct FloatingToastContent: View {
    @AppStorage(AppPreferenceKey.overlayCardOpacity) private var overlayCardOpacity = 82
    @AppStorage(AppPreferenceKey.overlayCardCornerRadius) private var overlayCardCornerRadius = 24

    let message: String
    let kind: FloatingToastKind
    @State private var appeared = false

    private var cornerRadius: CGFloat { CGFloat(min(max(overlayCardCornerRadius, 0), 40)) }
    private var cardOpacity: Double { Double(min(max(overlayCardOpacity, 0), 100)) / 100.0 }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: kind.systemImageName)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(kind.color)
            Text(message)
                .font(.system(size: 13, weight: .semibold))
                .lineLimit(1)
                .truncationMode(.middle)
                .foregroundStyle(.white.opacity(0.94))
        }
        .frame(width: 240, height: 52)
        .background(cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .shadow(color: .black.opacity(0.18), radius: 18, y: 10)
        .scaleEffect(appeared ? 1.0 : 0.88, anchor: .bottom)
        .opacity(appeared ? 1.0 : 0.0)
        .animation(.spring(response: 0.35, dampingFraction: 0.62, blendDuration: 0.1), value: appeared)
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.02) {
                appeared = true
            }
        }
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .fill(.black.opacity(cardOpacity))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
            )
    }
}

private extension CGRect {
    func isApproximatelyEqual(to other: CGRect, tolerance: CGFloat = 0.5) -> Bool {
        abs(origin.x - other.origin.x) <= tolerance &&
            abs(origin.y - other.origin.y) <= tolerance &&
            abs(size.width - other.size.width) <= tolerance &&
            abs(size.height - other.size.height) <= tolerance
    }
}
