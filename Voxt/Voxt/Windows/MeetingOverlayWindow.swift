// MeetingOverlayWindow.swift
// Provides Meeting Overlay Window for window and overlay UI.

import AppKit
import SwiftUI
import Combine
import QuartzCore

final class MeetingOverlayWindow: NSPanel {
    private var hostingView: NSHostingView<MeetingOverlayContainerView>?
    private var collapseCancellable: AnyCancellable?
    private var pickerStateCancellable: AnyCancellable?
    private var appearanceCancellable: AnyCancellable?
    private weak var observedState: MeetingOverlayState?
    private var localClickMonitor: Any?
    private var globalClickMonitor: Any?
    private var currentPosition: OverlayPosition = .bottom

    var onRequestClose: (() -> Void)?
    var onRequestCollapseToggle: (() -> Void)?
    var onRequestPauseToggle: (() -> Void)?
    var onRequestDetail: (() -> Void)?
    var onRequestRealtimeTranslateToggle: ((Bool) -> Void)?
    var onRequestCaptureModeChange: ((MeetingCaptureMode) -> Void)?
    var onRequestCaptureModePickerToggle: (() -> Void)?
    var onRequestCaptureModePickerDismiss: (() -> Void)?
    var onRequestRealtimeTranslationLanguageConfirm: (() -> Void)?
    var onRequestRealtimeTranslationLanguageCancel: (() -> Void)?
    var onRequestCancelMeeting: (() -> Void)?
    var onRequestFinishMeeting: (() -> Void)?
    var onRequestDismissCloseConfirmation: (() -> Void)?
    var onRequestCopySegment: ((MeetingTranscriptSegment) -> Void)?

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
        applyScreenSharingPreference()
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        ignoresMouseEvents = false

        appearanceCancellable = NotificationCenter.default.publisher(for: .voxtOverlayAppearanceDidChange)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                guard let self, let state = self.observedState else { return }
                let raw = UserDefaults.standard.string(forKey: AppPreferenceKey.overlayPosition) ?? OverlayPosition.bottom.rawValue
                self.currentPosition = OverlayPosition(rawValue: raw) ?? .bottom
                self.updateAppearance(for: state, animated: self.isVisible)
            }
    }

    override var canBecomeKey: Bool { true }

    func show(state: MeetingOverlayState, position: OverlayPosition) {
        currentPosition = position
        state.isPresented = true
        applyScreenSharingPreference()

        let content = MeetingOverlayContainerView(
            state: state,
            onClose: { [weak self] in self?.onRequestClose?() },
            onToggleCollapse: { [weak self] in self?.onRequestCollapseToggle?() },
            onTogglePause: { [weak self] in self?.onRequestPauseToggle?() },
            onShowDetail: { [weak self] in self?.onRequestDetail?() },
            onRealtimeTranslateToggle: { [weak self] isEnabled in
                self?.onRequestRealtimeTranslateToggle?(isEnabled)
            },
            onCaptureModeChange: { [weak self] mode in
                self?.onRequestCaptureModeChange?(mode)
            },
            onToggleCaptureModePicker: { [weak self] in
                self?.onRequestCaptureModePickerToggle?()
            },
            onDismissCaptureModePicker: { [weak self] in
                self?.onRequestCaptureModePickerDismiss?()
            },
            onConfirmRealtimeTranslationLanguage: { [weak self] in
                self?.onRequestRealtimeTranslationLanguageConfirm?()
            },
            onCancelRealtimeTranslationLanguage: { [weak self] in
                self?.onRequestRealtimeTranslationLanguageCancel?()
            },
            onConfirmCancelMeeting: { [weak self] in
                self?.onRequestCancelMeeting?()
            },
            onConfirmFinishMeeting: { [weak self] in
                self?.onRequestFinishMeeting?()
            },
            onDismissCloseConfirmation: { [weak self] in
                self?.onRequestDismissCloseConfirmation?()
            },
            onCopySegment: { [weak self] segment in
                self?.onRequestCopySegment?(segment)
            }
        )

        if let hostingView {
            hostingView.rootView = content
        } else {
            let hosting = NSHostingView(rootView: content)
            hosting.translatesAutoresizingMaskIntoConstraints = false
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

    func hide(completion: (() -> Void)? = nil) {
        observedState?.isPresented = false
        observedState?.isCaptureModePickerPresented = false
        removeOutsideClickMonitors()

        guard isVisible else {
            orderOut(nil)
            completion?()
            return
        }

        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.22
            animator().alphaValue = 0
        }, completionHandler: { [weak self] in
            self?.orderOut(nil)
            completion?()
        })
    }

    private func observe(state: MeetingOverlayState) {
        guard observedState !== state else { return }
        observedState = state
        collapseCancellable = Publishers.CombineLatest(state.$isCollapsed, state.$segments)
            .receive(on: RunLoop.main)
            .sink { [weak self, weak state] _ in
                guard let self, let state else { return }
                self.updateAppearance(for: state, animated: true)
            }
        pickerStateCancellable = state.$isCaptureModePickerPresented
            .receive(on: RunLoop.main)
            .sink { [weak self] isPresented in
                if isPresented {
                    self?.installOutsideClickMonitors()
                } else {
                    self?.removeOutsideClickMonitors()
                }
            }
    }

    private func updateAppearance(for state: MeetingOverlayState, animated: Bool) {
        let targetFrame = frame(for: panelSize(for: state), position: currentPosition)
        guard !targetFrame.isEmpty else { return }
        if animated, isVisible {
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.22
                context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                animator().setFrame(targetFrame, display: true)
            }
        } else {
            setFrame(targetFrame, display: true)
        }
    }

    private func panelSize(for state: MeetingOverlayState) -> CGSize {
        if state.isCollapsed {
            return CGSize(width: 340, height: 48)
        }
        return CGSize(width: 560, height: 360)
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

    private func applyScreenSharingPreference() {
        let hideFromScreenSharing = UserDefaults.standard.bool(
            forKey: AppPreferenceKey.hideMeetingOverlayFromScreenSharing
        )
        sharingType = hideFromScreenSharing ? .none : .readOnly
    }

    private func installOutsideClickMonitors() {
        guard localClickMonitor == nil, globalClickMonitor == nil else { return }

        localClickMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]
        ) { [weak self] event in
            self?.handleOutsideClickIfNeeded(eventLocationInScreen: NSEvent.mouseLocation, sourceWindow: event.window)
            return event
        }

        globalClickMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleOutsideClickIfNeeded(eventLocationInScreen: NSEvent.mouseLocation, sourceWindow: nil)
            }
        }
    }

    private func removeOutsideClickMonitors() {
        if let localClickMonitor {
            NSEvent.removeMonitor(localClickMonitor)
            self.localClickMonitor = nil
        }
        if let globalClickMonitor {
            NSEvent.removeMonitor(globalClickMonitor)
            self.globalClickMonitor = nil
        }
    }

    private func handleOutsideClickIfNeeded(eventLocationInScreen: NSPoint, sourceWindow: NSWindow?) {
        guard let state = observedState,
              state.isCaptureModePickerPresented
        else {
            return
        }

        if sourceWindow === self {
            return
        }

        guard !frame.contains(eventLocationInScreen) else { return }
        onRequestCaptureModePickerDismiss?()
    }

    deinit {
        removeOutsideClickMonitors()
    }
}
