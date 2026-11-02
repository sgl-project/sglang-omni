// ModelDebugWindowControllers.swift
// Provides Model Debug Window Controllers for window and overlay UI.

import AppKit
import SwiftUI
import AVFoundation
import Combine

@MainActor
final class ASRDebugWindowManager {
    static let shared = ASRDebugWindowManager()

    private var controller: ASRDebugWindowController?

    func present(appDelegate: AppDelegate) {
        let controller = resolvedController(appDelegate: appDelegate)
        controller.refresh()
        controller.showWindow(nil)
        AppBehaviorController.bringStandardWindowToFront(controller.window)
    }

    private func resolvedController(appDelegate: AppDelegate) -> ASRDebugWindowController {
        if let controller {
            return controller
        }
        let controller = ASRDebugWindowController(viewModel: ASRDebugViewModel(appDelegate: appDelegate)) { [weak self] in
            self?.controller = nil
        }
        self.controller = controller
        return controller
    }
}

@MainActor
final class LLMDebugWindowManager {
    static let shared = LLMDebugWindowManager()

    private var controller: LLMDebugWindowController?

    func present(appDelegate: AppDelegate) {
        let controller = resolvedController(appDelegate: appDelegate)
        controller.refresh()
        controller.showWindow(nil)
        AppBehaviorController.bringStandardWindowToFront(controller.window)
    }

    private func resolvedController(appDelegate: AppDelegate) -> LLMDebugWindowController {
        if let controller {
            return controller
        }
        let controller = LLMDebugWindowController(viewModel: LLMDebugViewModel(appDelegate: appDelegate)) { [weak self] in
            self?.controller = nil
        }
        self.controller = controller
        return controller
    }
}

@MainActor
private final class ASRDebugWindowController: NSWindowController, NSWindowDelegate {
    private let viewModel: ASRDebugViewModel
    private let onClose: () -> Void

    init(viewModel: ASRDebugViewModel, onClose: @escaping () -> Void) {
        self.viewModel = viewModel
        self.onClose = onClose

        let rootView = ASRDebugWindowView(viewModel: viewModel)
        let hostingController = NSHostingController(rootView: rootView)
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: NSSize(width: ModelDebugWindowStyle.width, height: ModelDebugWindowStyle.height)),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.contentViewController = hostingController
        window.title = modelDebugLocalized("ASR Debug")
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.toolbar = nil
        window.collectionBehavior = []
        window.level = .normal
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: ModelDebugWindowStyle.minWidth, height: ModelDebugWindowStyle.minHeight)
        window.setContentSize(NSSize(width: ModelDebugWindowStyle.width, height: ModelDebugWindowStyle.height))
        window.center()
        modelDebugConfigureWindowChrome(window)

        super.init(window: window)
        window.delegate = self
        modelDebugPositionWindowTrafficLightButtons(window)
        modelDebugScheduleTrafficLightButtonPositionUpdate(for: window)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func refresh() {
        viewModel.refreshOptions()
    }

    func windowWillClose(_ notification: Notification) {
        viewModel.handleWindowClose()
        onClose()
    }

    func windowDidResize(_ notification: Notification) {
        guard let window else { return }
        modelDebugScheduleTrafficLightButtonPositionUpdate(for: window)
    }

    func windowDidBecomeKey(_ notification: Notification) {
        guard let window else { return }
        modelDebugScheduleTrafficLightButtonPositionUpdate(for: window)
    }
}

@MainActor
private final class LLMDebugWindowController: NSWindowController, NSWindowDelegate {
    private let viewModel: LLMDebugViewModel
    private let onClose: () -> Void

    init(viewModel: LLMDebugViewModel, onClose: @escaping () -> Void) {
        self.viewModel = viewModel
        self.onClose = onClose

        let rootView = LLMDebugWindowView(viewModel: viewModel)
        let hostingController = NSHostingController(rootView: rootView)
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: NSSize(width: ModelDebugWindowStyle.width, height: ModelDebugWindowStyle.height)),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.contentViewController = hostingController
        window.title = modelDebugLocalized("LLM Debug")
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.toolbar = nil
        window.collectionBehavior = []
        window.level = .normal
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: ModelDebugWindowStyle.minWidth, height: ModelDebugWindowStyle.minHeight)
        window.setContentSize(NSSize(width: ModelDebugWindowStyle.width, height: ModelDebugWindowStyle.height))
        window.center()
        modelDebugConfigureWindowChrome(window)

        super.init(window: window)
        window.delegate = self
        modelDebugPositionWindowTrafficLightButtons(window)
        modelDebugScheduleTrafficLightButtonPositionUpdate(for: window)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func refresh() {
        viewModel.refreshOptions()
    }

    func windowWillClose(_ notification: Notification) {
        viewModel.handleWindowClose()
        onClose()
    }

    func windowDidResize(_ notification: Notification) {
        guard let window else { return }
        modelDebugScheduleTrafficLightButtonPositionUpdate(for: window)
    }

    func windowDidBecomeKey(_ notification: Notification) {
        guard let window else { return }
        modelDebugScheduleTrafficLightButtonPositionUpdate(for: window)
    }
}
