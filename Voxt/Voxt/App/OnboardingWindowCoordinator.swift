// OnboardingWindowCoordinator.swift
// Provides Onboarding Window Coordinator for app lifecycle and routing.

import SwiftUI
import AppKit

private struct OnboardingGuideWindowRoot: View {
    @State private var currentStep: OnboardingGuideStep

    @ObservedObject var mlxModelManager: MLXModelManager
    @ObservedObject var customLLMManager: CustomLLMModelManager
    @ObservedObject var overlayState: OverlayState

    let onClose: () -> Void
    let onFinish: () -> Void

    init(
        initialStep: OnboardingGuideStep,
        mlxModelManager: MLXModelManager,
        customLLMManager: CustomLLMModelManager,
        overlayState: OverlayState,
        onClose: @escaping () -> Void,
        onFinish: @escaping () -> Void
    ) {
        _currentStep = State(initialValue: initialStep)
        self.mlxModelManager = mlxModelManager
        self.customLLMManager = customLLMManager
        self.overlayState = overlayState
        self.onClose = onClose
        self.onFinish = onFinish
    }

    var body: some View {
        OnboardingGuideView(
            currentStep: $currentStep,
            mlxModelManager: mlxModelManager,
            customLLMManager: customLLMManager,
            overlayState: overlayState,
            onClose: onClose,
            onFinish: onFinish
        )
    }
}

extension AppDelegate {
    private var onboardingWindowContentSize: NSSize {
        NSSize(width: 880, height: 600)
    }

    private final class OnboardingHostWindow: NSWindow {
        override var canBecomeKey: Bool { true }
        override var canBecomeMain: Bool { true }
    }

    func openOnboardingWindow(step requestedStep: OnboardingGuideStep? = nil) {
        let initialStep = requestedStep
            ?? OnboardingPreferenceManager.savedLastGuideStep()
            ?? .permissions
        if let window = onboardingWindowController?.window {
            if !window.isVisible {
                window.center()
            }
            AppBehaviorController.bringUserInvokedWindowToFront(window)
            return
        }

        OnboardingPreferenceManager.saveLastGuideStep(initialStep)
        let window = OnboardingHostWindow(
            contentRect: NSRect(origin: .zero, size: onboardingWindowContentSize),
            styleMask: [.borderless, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = AppLocalization.localizedString("Setup Guide")
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true
        window.isMovableByWindowBackground = true
        window.contentMinSize = onboardingWindowContentSize
        window.contentMaxSize = onboardingWindowContentSize
        window.isReleasedWhenClosed = false
        window.isRestorable = false
        window.level = .normal
        window.collectionBehavior = [.managed, .participatesInCycle]
        window.delegate = self

        let controller = NSWindowController(window: window)
        controller.shouldCascadeWindows = false
        onboardingWindowController = controller

        let contentView = OnboardingGuideWindowRoot(
            initialStep: initialStep,
            mlxModelManager: mlxModelManager,
            customLLMManager: customLLMManager,
            overlayState: overlayState,
            onClose: { [weak self, weak window] in
                window?.close()
                self?.onboardingWindowController = nil
            },
            onFinish: { [weak self, weak window] in
                OnboardingPreferenceManager.markCompleted()
                window?.close()
                self?.onboardingWindowController = nil
            }
        )

        window.contentViewController = NSHostingController(rootView: contentView)
        window.setContentSize(onboardingWindowContentSize)
        window.center()
        AppBehaviorController.bringUserInvokedWindowToFront(window)
    }
}
