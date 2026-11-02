// MenuWindowCoordinator.swift
// Provides Menu Window Coordinator for app lifecycle and routing.

import SwiftUI
import AppKit
import CoreAudio
import Combine

extension AppDelegate {
    private var mainWindowContentSize: NSSize {
        NSSize(width: 820, height: 560)
    }

    private var mainWindowMinimumContentSize: NSSize {
        NSSize(width: 820, height: 560)
    }

    private func setMainWindowVisibility(_ isVisible: Bool) {
        synchronizeAppActivationPolicy(mainWindowVisible: isVisible)
        guard mainWindowVisibilityState.isVisible != isVisible else { return }
        mainWindowVisibilityState.isVisible = isVisible
    }

    func synchronizeAppActivationPolicy() {
        synchronizeAppActivationPolicy(mainWindowVisible: mainWindowVisibilityState.isVisible)
    }

    private func synchronizeAppActivationPolicy(mainWindowVisible: Bool) {
        AppBehaviorController.applyDockVisibility(
            showInDock: showInDock,
            mainWindowVisible: mainWindowVisible
        )
    }

    private func repairedMainWindowFrame(for window: NSWindow) -> NSRect {
        let contentRect = NSRect(origin: .zero, size: mainWindowContentSize)
        let frameSize = window.frameRect(forContentRect: contentRect).size
        return NSRect(origin: window.frame.origin, size: frameSize)
    }

    private func minimumMainWindowFrame(for window: NSWindow) -> NSRect {
        let contentRect = NSRect(origin: .zero, size: mainWindowMinimumContentSize)
        let frameSize = window.frameRect(forContentRect: contentRect).size
        return NSRect(origin: window.frame.origin, size: frameSize)
    }

    private func repairMainWindowFrameIfNeeded(_ window: NSWindow) {
        let repairedFrame = repairedMainWindowFrame(for: window)
        let minimumFrame = minimumMainWindowFrame(for: window)
        let needsRepair =
            window.frame.width < minimumFrame.width ||
            window.frame.height < minimumFrame.height

        guard needsRepair else { return }
        window.setFrame(repairedFrame, display: false)
    }

    private func recenterMainWindowIfNeeded(_ window: NSWindow) {
        repairMainWindowFrameIfNeeded(window)

        let candidateScreen = window.screen ?? NSScreen.main ?? NSScreen.screens.first
        guard let visibleFrame = candidateScreen?.visibleFrame else { return }

        let needsPositionRepair =
            !visibleFrame.intersects(window.frame) ||
            window.frame.maxX <= visibleFrame.minX ||
            window.frame.maxY <= visibleFrame.minY ||
            window.frame.minX >= visibleFrame.maxX ||
            window.frame.minY >= visibleFrame.maxY

        guard needsPositionRepair else { return }

        centerMainWindow(window, on: candidateScreen)
    }

    private var feedbackURL: URL {
        URL(string: "https://github.com/hehehai/voxt/issues/new/choose")!
    }

    func buildMenu() {
        // Replacing a menu while AppKit tracks the pointer disrupts highlighting
        // and can invalidate its open submenu. Coalesce updates until dismissal.
        guard !isStatusMenuOpen else {
            statusMenuNeedsRebuild = true
            return
        }
        statusMenuNeedsRebuild = false
        let availability = FeatureSettingsStore.availability()
        let menu = NSMenu()
        menu.autoenablesItems = false
        menu.delegate = self

        let dashboardItem = NSMenuItem(title: AppLocalization.localizedString("Dashboard"), action: #selector(openDashboardFromMenu), keyEquivalent: "")
        dashboardItem.target = self
        menu.addItem(dashboardItem)

        let generalItem = NSMenuItem(title: AppLocalization.localizedString("General"), action: #selector(openGeneralFromMenu), keyEquivalent: ",")
        generalItem.target = self
        menu.addItem(generalItem)

        let featureItem = NSMenuItem(
            title: AppLocalization.localizedString("Feature"),
            action: #selector(openFeatureFromMenu),
            keyEquivalent: ""
        )
        featureItem.target = self
        menu.addItem(featureItem)

        let historyItem = NSMenuItem(title: AppLocalization.localizedString("History"), action: nil, keyEquivalent: "")
        historyItem.submenu = buildHistoryMenu(availability: availability)
        menu.addItem(historyItem)

        let dictionaryItem = NSMenuItem(
            title: AppLocalization.localizedString("Dictionary"),
            action: #selector(openDictionarySettings),
            keyEquivalent: ""
        )
        dictionaryItem.target = self
        menu.addItem(dictionaryItem)

        let microphoneItem = NSMenuItem(title: AppLocalization.localizedString("Microphone"), action: nil, keyEquivalent: "")
        microphoneItem.submenu = buildMicrophoneMenu()
        menu.addItem(microphoneItem)

        addRecentHistoryItems(to: menu, availability: availability)
        menu.addItem(NSMenuItem.separator())

        let checkUpdatesItem = NSMenuItem(
            title: AppLocalization.localizedString("Check for Updates…"),
            action: #selector(checkForUpdates),
            keyEquivalent: ""
        )
        checkUpdatesItem.target = self
        checkUpdatesItem.isEnabled = !appUpdateManager.shouldDisableInteractiveUpdateTrigger
        menu.addItem(checkUpdatesItem)

        let feedbackItem = NSMenuItem(
            title: AppLocalization.localizedString("Feedback"),
            action: #selector(openFeedbackPage),
            keyEquivalent: ""
        )
        feedbackItem.target = self
        menu.addItem(feedbackItem)

        if appUpdateManager.hasUpdate, let latestVersion = appUpdateManager.latestVersion {
            let updateInfoItem = NSMenuItem(
                title: AppLocalization.format("New version: %@", latestVersion),
                action: nil,
                keyEquivalent: ""
            )
            updateInfoItem.isEnabled = false
            menu.addItem(updateInfoItem)
        }

        menu.addItem(NSMenuItem.separator())
        let quitItem = NSMenuItem(title: AppLocalization.localizedString("Quit Voxt"), action: #selector(quit), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)
        statusItem?.menu = menu
    }

    func startObservingStatusMenuHistory() {
        statusMenuHistoryCancellable = historyStore.$entries
            // Deliver after store mutations/persistence finish, and collapse bursts.
            .debounce(for: .milliseconds(50), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.refreshStatusMenuHistory()
            }
    }

    private func refreshStatusMenuHistory() {
        statusMenuHistoryGeneration += 1
        let generation = statusMenuHistoryGeneration
        historyStore.loadRecentMenuEntries(limitPerKind: StatusMenuHistorySupport.recentLimit) { [weak self] entries in
            guard let self, generation == self.statusMenuHistoryGeneration else { return }
            guard entries != self.statusMenuRecentEntries else { return }
            self.statusMenuRecentEntries = entries
            self.buildMenu()
        }
    }

    private func buildHistoryMenu(availability: FeatureAvailabilitySettings) -> NSMenu {
        let submenu = NSMenu()
        submenu.autoenablesItems = false
        for filter in StatusMenuHistorySupport.filters(availability: availability) {
            let item = NSMenuItem(title: filter.title, action: #selector(openHistoryFromMenu(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = filter.rawValue
            submenu.addItem(item)
        }
        return submenu
    }

    private func addRecentHistoryItems(to menu: NSMenu, availability: FeatureAvailabilitySettings) {
        let entries = StatusMenuHistorySupport.recentEntries(from: statusMenuRecentEntries, availability: availability)
        guard !entries.isEmpty else { return }

        menu.addItem(NSMenuItem.separator())
        for entry in entries {
            let item = NSMenuItem(
                title: StatusMenuHistorySupport.previewTitle(for: entry),
                action: #selector(copyRecentHistoryFromMenu(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = entry.id.uuidString
            item.toolTip = AppLocalization.localizedString("Click to copy full text")
            menu.addItem(item)
        }
    }

    private func buildMicrophoneMenu() -> NSMenu {
        let submenu = NSMenu()
        submenu.autoenablesItems = false
        let resolvedSelectedUID = selectedInputDeviceUID

        let autoSwitchItem = NSMenuItem(
            title: AppLocalization.localizedString("Auto Switch"),
            action: #selector(toggleMicrophoneAutoSwitch(_:)),
            keyEquivalent: ""
        )
        autoSwitchItem.target = self
        autoSwitchItem.state = microphoneResolvedState.autoSwitchEnabled ? .on : .off
        submenu.addItem(autoSwitchItem)
        submenu.addItem(NSMenuItem.separator())

        for device in inputDevicesSnapshot {
            let item = NSMenuItem(title: device.name, action: #selector(selectMicrophoneFromMenu(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = device.uid
            item.state = device.uid == resolvedSelectedUID ? .on : .off
            submenu.addItem(item)
        }

        if inputDevicesSnapshot.isEmpty {
            let emptyItem = NSMenuItem(title: AppLocalization.localizedString("No microphone available"), action: nil, keyEquivalent: "")
            emptyItem.isEnabled = false
            submenu.addItem(emptyItem)
        }

        return submenu
    }

    func startObservingAudioInputDevices() {
        audioInputDevicesObserver = AudioInputDeviceManager.makeDevicesObserver { [weak self] in
            Task { @MainActor [weak self] in
                self?.refreshInputDevicesSnapshot(reason: "hardware change")
            }
        }
    }

    func refreshInputDevicesSnapshot(reason: String) {
        inputDevicesRefreshTask?.cancel()
        VoxtLog.info("Refreshing audio input snapshot. reason=\(reason)", verbose: true)

        inputDevicesRefreshTask = Task { [weak self] in
            let devices = await Task.detached(priority: .utility) {
                AudioInputDeviceManager.snapshotAvailableInputDevices()
            }.value
            guard !Task.isCancelled else { return }

            await MainActor.run {
                guard let self else { return }
                self.applyInputDevicesSnapshot(devices, reason: reason)
            }
        }
    }

    private func applyInputDevicesSnapshot(_ devices: [AudioInputDevice], reason: String) {
        let previousDevices = inputDevicesSnapshot
        let previousSelectedUID = selectedInputDeviceUID
        let previousResolvedState = microphoneResolvedState
        inputDevicesSnapshot = devices
        let lockedActiveUID = lockedActiveInputDeviceUID(
            previousResolvedState: previousResolvedState,
            availableDevices: devices,
            reason: reason
        )
        let resolvedState = MicrophonePreferenceManager.syncState(
            defaults: .standard,
            availableDevices: devices,
            previousAvailableUIDs: previousDevices.isEmpty ? nil : Set(previousDevices.map(\.uid)),
            lockedActiveUID: lockedActiveUID
        )
        microphoneResolvedState = resolvedState

        let devicesChanged = previousDevices != devices
        let selectionChanged = previousSelectedUID != resolvedState.activeUID
        let previousUIDs = Set(previousDevices.map(\.uid))
        let currentUIDs = Set(devices.map(\.uid))
        let addedDevices = devices.filter { !previousUIDs.contains($0.uid) }
        let removedDevices = previousDevices.filter { !currentUIDs.contains($0.uid) }

        if !addedDevices.isEmpty || !removedDevices.isEmpty {
            VoxtLog.info(
                """
                Audio input hardware change applied. reason=\(reason), previousCount=\(previousDevices.count), currentCount=\(devices.count), added=\(describeDevices(addedDevices)), removed=\(describeDevices(removedDevices))
                """
            )
        }

        if (devicesChanged || selectionChanged), isSessionActive {
            VoxtLog.warning(
                """
                Audio input change observed during active session. reason=\(reason), recordingActive=\(isSessionActive), previousSelected=\(previousSelectedUID ?? "none"), newSelected=\(resolvedState.activeUID ?? "none"), devicesChanged=\(devicesChanged), selectionChanged=\(selectionChanged), added=\(describeDevices(addedDevices)), removed=\(describeDevices(removedDevices))
                """
            )
        }

        guard devicesChanged || selectionChanged else { return }

        if devicesChanged {
            NotificationCenter.default.post(name: .voxtAudioInputDevicesDidChange, object: nil)
        }

        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)

        VoxtLog.info(
            "Audio input snapshot refreshed. reason=\(reason), devices=\(devices.count), selected=\(resolvedState.activeUID ?? "none"), autoSwitch=\(resolvedState.autoSwitchEnabled)",
            verbose: true
        )
        handleResolvedMicrophoneStateChange(
            from: previousResolvedState,
            to: resolvedState,
            reason: reason
        )
        buildMenu()
    }

    private func describeDevices(_ devices: [AudioInputDevice]) -> String {
        guard !devices.isEmpty else { return "[]" }
        return "[" + devices.map { "\($0.name){uid=\($0.uid),id=\($0.id)}" }.joined(separator: ", ") + "]"
    }

    private func lockedActiveInputDeviceUID(
        previousResolvedState: MicrophoneResolvedState,
        availableDevices: [AudioInputDevice],
        reason: String
    ) -> String? {
        guard reason == "hardware change" else { return nil }
        guard let currentUID = previousResolvedState.activeUID else { return nil }
        guard availableDevices.contains(where: { $0.uid == currentUID }) else { return nil }

        guard isSessionActive, recordingStoppedAt == nil else { return nil }
        VoxtLog.info(
            "Preserving active microphone during recording hardware change. uid=\(currentUID), output=\(sessionOutputMode)"
        )
        return currentUID
    }

    @objc private func checkForUpdates() {
        performAfterStatusMenuDismissal {
            VoxtLog.info("Manual update check triggered from menu.")
            self.appUpdateManager.checkForUpdatesWithUserInterface()
        }
    }

    @objc private func openFeedbackPage() {
        performAfterStatusMenuDismissal {
            VoxtLog.info("Feedback page opened from menu.")
            NSWorkspace.shared.open(self.feedbackURL)
        }
    }

    @objc private func openDashboardFromMenu() {
        performAfterStatusMenuDismissal {
            self.openMainWindow(target: SettingsNavigationTarget(tab: .report))
        }
    }

    @objc private func openGeneralFromMenu() {
        performAfterStatusMenuDismissal {
            self.openMainWindow(target: SettingsNavigationTarget(tab: .general))
        }
    }

    @objc private func openDictionarySettings() {
        performAfterStatusMenuDismissal {
            self.openMainWindow(target: SettingsNavigationTarget(tab: .dictionary))
        }
    }

    @objc private func openFeatureFromMenu() {
        performAfterStatusMenuDismissal {
            self.openMainWindow(target: SettingsNavigationTarget(tab: .feature, featureTab: .features))
        }
    }

    @objc private func openHistoryFromMenu(_ sender: NSMenuItem) {
        guard let rawValue = sender.representedObject as? String,
              let filter = HistoryFilterTab(rawValue: rawValue)
        else { return }
        performAfterStatusMenuDismissal {
            self.openMainWindow(
                target: SettingsNavigationTarget(
                    tab: .history,
                    section: .historyEntries,
                    historyFilter: filter
                )
            )
        }
    }

    @objc private func copyRecentHistoryFromMenu(_ sender: NSMenuItem) {
        guard let rawID = sender.representedObject as? String,
              let id = UUID(uuidString: rawID)
        else { return }
        performAfterStatusMenuDismissal {
            // Fetch the full, current text only on selection, never during hover.
            self.historyStore.loadEntry(id: id) { [weak self] entry in
                guard let self, let entry else { return }
                _ = self.pasteboardTextWriter.write(entry.text, to: .general, restorePrevious: false)
            }
        }
    }

    @objc private func selectMicrophoneFromMenu(_ sender: NSMenuItem) {
        guard let uid = sender.representedObject as? String else { return }
        let previousState = microphoneResolvedState
        if let device = inputDevicesSnapshot.first(where: { $0.uid == uid }) {
            VoxtLog.info("Microphone focus changed from tray menu. uid=\(uid), name=\(device.name)")
        } else {
            VoxtLog.info("Microphone focus changed from tray menu. uid=\(uid)")
        }
        microphoneResolvedState = MicrophonePreferenceManager.setFocusedDevice(
            uid: uid,
            defaults: .standard,
            availableDevices: inputDevicesSnapshot
        )
        handleResolvedMicrophoneStateChange(
            from: previousState,
            to: microphoneResolvedState,
            reason: "tray microphone selection"
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
        buildMenu()
    }

    @objc private func toggleMicrophoneAutoSwitch(_ sender: NSMenuItem) {
        let newValue = sender.state != .on
        let previousState = microphoneResolvedState
        VoxtLog.info("Microphone auto switch updated from tray menu. enabled=\(newValue)")
        microphoneResolvedState = MicrophonePreferenceManager.setAutoSwitchEnabled(
            newValue,
            defaults: .standard,
            availableDevices: inputDevicesSnapshot
        )
        handleResolvedMicrophoneStateChange(
            from: previousState,
            to: microphoneResolvedState,
            reason: "tray microphone auto-switch"
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
        buildMenu()
    }

    func applyMicrophonePriorityOrder(_ orderedUIDs: [String]) {
        let previousState = microphoneResolvedState
        microphoneResolvedState = MicrophonePreferenceManager.reorderPriority(
            orderedUIDs: orderedUIDs,
            defaults: .standard,
            availableDevices: inputDevicesSnapshot
        )
        handleResolvedMicrophoneStateChange(
            from: previousState,
            to: microphoneResolvedState,
            reason: "microphone priority reorder"
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
        buildMenu()
    }

    func selectMicrophoneManually(uid: String) {
        let previousState = microphoneResolvedState
        if let device = inputDevicesSnapshot.first(where: { $0.uid == uid }) {
            VoxtLog.info("Microphone focus changed. uid=\(uid), name=\(device.name)")
        } else {
            VoxtLog.info("Microphone focus changed. uid=\(uid)")
        }
        microphoneResolvedState = MicrophonePreferenceManager.setFocusedDevice(
            uid: uid,
            defaults: .standard,
            availableDevices: inputDevicesSnapshot
        )
        handleResolvedMicrophoneStateChange(
            from: previousState,
            to: microphoneResolvedState,
            reason: "manual microphone selection"
        )
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
        buildMenu()
    }

    func handleResolvedMicrophoneStateChange(
        from previousState: MicrophoneResolvedState,
        to newState: MicrophoneResolvedState,
        reason: String
    ) {
        guard previousState.activeUID != newState.activeUID else {
            applyPreferredInputDevice()
            return
        }

        if let activeDevice = newState.activeDevice {
            VoxtLog.info(
                "Resolved microphone changed. reason=\(reason), uid=\(activeDevice.uid), name=\(activeDevice.name), autoSwitch=\(newState.autoSwitchEnabled)"
            )
        } else {
            VoxtLog.warning("Resolved microphone cleared. reason=\(reason)")
        }

        handlePreferredInputDeviceChange(
            previousUID: previousState.activeUID,
            newUID: newState.activeUID,
            reason: reason
        )
    }

    func presentMainWindowOnLaunchIfNeeded() {
        if OnboardingPreferenceManager.shouldPresentOnLaunch() {
            openOnboardingWindow()
            return
        }
        guard LaunchPresentationPolicy.shouldPresentMainWindowOnLaunch() else { return }
        openMainWindow()
    }

    func openMainWindow() {
        openMainWindow(target: SettingsNavigationTarget(tab: .report))
    }

    func openMainWindow(target: SettingsNavigationTarget) {
        let navigationRequest = SettingsNavigationRequest(target: target)

        if let window = mainWindowController?.window {
            NotificationCenter.default.post(
                name: .voxtSettingsNavigate,
                object: nil,
                userInfo: navigationRequest.target.userInfo
            )
            if !window.isVisible {
                recenterMainWindowIfNeeded(window)
            }
            setMainWindowVisibility(true)
            bringWindowToFront(window)
            return
        }

        let contentView = SettingsView(
            availableDictionaryHistoryScanModels: {
                self.availableDictionaryHistoryScanModelOptions()
            },
            onIngestDictionarySuggestionsFromHistory: { request, persistSettings in
                self.startDictionaryHistorySuggestionScan(
                    request: request,
                    persistSettings: persistSettings
                )
            },
            onCancelDictionarySuggestionsFromHistory: {
                self.cancelDictionaryHistorySuggestionScan()
            },
            mlxModelManager: mlxModelManager,
            customLLMManager: customLLMManager,
            ggufTranslationModelManager: ggufTranslationModelManager,
            historyStore: historyStore,
            meetingFileTaskQueue: meetingFileTaskQueue,
            noteStore: noteStore,
            dictionaryStore: dictionaryStore,
            dictionarySuggestionStore: dictionarySuggestionStore,
            appUpdateManager: appUpdateManager,
            mainWindowState: mainWindowVisibilityState,
            initialNavigationTarget: navigationRequest.target,
            initialDisplayMode: resolvedInitialDisplayMode(for: navigationRequest.target)
        )
        .frame(
            minWidth: mainWindowMinimumContentSize.width,
            idealWidth: mainWindowContentSize.width,
            maxWidth: .infinity,
            minHeight: mainWindowMinimumContentSize.height,
            idealHeight: mainWindowContentSize.height,
            maxHeight: .infinity
        )

        let hostingController = NSHostingController(rootView: contentView)

        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: mainWindowContentSize),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Voxt"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.toolbar = nil
        window.isOpaque = true
        window.backgroundColor = SettingsUIStyle.windowBackgroundNSColor
        window.isMovableByWindowBackground = false
        // Keep the main window on its assigned Space, matching normal AppKit
        // application behavior when switching between Spaces.
        window.collectionBehavior = []
        window.contentViewController = hostingController
        window.setContentSize(mainWindowContentSize)
        window.contentMinSize = mainWindowMinimumContentSize
        window.isReleasedWhenClosed = false
        window.isRestorable = false
        window.level = .normal
        window.delegate = self

        let controller = NSWindowController(window: window)
        controller.shouldCascadeWindows = false
        mainWindowController = controller
        repairMainWindowFrameIfNeeded(window)
        centerMainWindow(window, on: NSScreen.main)
        setMainWindowVisibility(true)
        bringWindowToFront(window)
        scheduleTrafficLightButtonPositionUpdate(for: window)
    }

    func openMainWindow(selectTab: SettingsTab?) {
        openMainWindow(target: SettingsNavigationTarget(tab: selectTab ?? .report))
    }

    private func resolvedInitialDisplayMode(for target: SettingsNavigationTarget) -> SettingsDisplayMode {
        _ = target
        return .normal
    }

    private func bringWindowToFront(_ window: NSWindow) {
        var collectionBehavior = window.collectionBehavior
        collectionBehavior.remove(.moveToActiveSpace)
        window.collectionBehavior = collectionBehavior
        AppBehaviorController.bringStandardWindowToFront(window)
    }

    private func centerMainWindow(_ window: NSWindow, on screen: NSScreen?) {
        guard let screen = screen ?? NSScreen.screens.first else {
            window.center()
            return
        }

        let visibleFrame = screen.visibleFrame
        let windowFrame = window.frame
        let origin = CGPoint(
            x: round(visibleFrame.midX - (windowFrame.width / 2)),
            y: round(visibleFrame.midY - (windowFrame.height / 2))
        )
        window.setFrameOrigin(origin)
    }

    private func performAfterStatusMenuDismissal(_ action: @escaping @MainActor () -> Void) {
        let wrappedAction: () -> Void = {
            Task { @MainActor in
                action()
            }
        }

        guard isStatusMenuOpen else {
            wrappedAction()
            return
        }

        pendingStatusMenuActions.append(wrappedAction)
    }

    private func positionWindowTrafficLightButtons(_ window: NSWindow) {
        guard let closeButton = window.standardWindowButton(.closeButton),
              let miniaturizeButton = window.standardWindowButton(.miniaturizeButton),
              let zoomButton = window.standardWindowButton(.zoomButton),
              let container = closeButton.superview
        else {
            return
        }

        let leftInset: CGFloat = 14
        let topInset: CGFloat = 14
        let spacing: CGFloat = 6

        let buttonSize = closeButton.frame.size
        let y = container.bounds.height - topInset - buttonSize.height
        let closeX = leftInset
        let miniaturizeX = closeX + buttonSize.width + spacing
        let zoomX = miniaturizeX + buttonSize.width + spacing

        closeButton.translatesAutoresizingMaskIntoConstraints = true
        miniaturizeButton.translatesAutoresizingMaskIntoConstraints = true
        zoomButton.translatesAutoresizingMaskIntoConstraints = true

        closeButton.setFrameOrigin(CGPoint(x: closeX, y: y))
        miniaturizeButton.setFrameOrigin(CGPoint(x: miniaturizeX, y: y))
        zoomButton.setFrameOrigin(CGPoint(x: zoomX, y: y))
    }

    private func setTrafficLightButtonsHidden(_ isHidden: Bool, for window: NSWindow) {
        window.standardWindowButton(.closeButton)?.isHidden = isHidden
        window.standardWindowButton(.miniaturizeButton)?.isHidden = isHidden
        window.standardWindowButton(.zoomButton)?.isHidden = isHidden
    }

    private func scheduleTrafficLightButtonPositionUpdate(for window: NSWindow) {
        DispatchQueue.main.async { [weak self, weak window] in
            guard let self, let window else { return }
            self.positionWindowTrafficLightButtons(window)
        }
    }

    @objc private func quit() {
        VoxtLog.info("Quit requested from menu.")
        requestApplicationTermination()
    }

    func prepareMainWindowForUpdatePresentation() {
        mainWindowPresentationState = MainWindowPresentationState()
        synchronizeAppActivationPolicy(mainWindowVisible: true)
        AppBehaviorController.activateCurrentApp()
        VoxtLog.info("Preparing Sparkle update UI without hiding main window.")
    }

    func restoreMainWindowAfterUpdateSessionIfNeeded() {
        mainWindowPresentationState = MainWindowPresentationState()
        VoxtLog.info("Sparkle update UI finished; leaving main window state unchanged.")
    }

    func showPermissionAlert() {
        let alert = NSAlert()
        alert.messageText = AppLocalization.localizedString("Permissions Required")
        alert.informativeText = AppLocalization.localizedString("Voxt needs Microphone access. If you use Direct Dictation, enable Speech Recognition in System Settings → Privacy & Security.")
        alert.addButton(withTitle: AppLocalization.localizedString("Open System Settings"))
        alert.addButton(withTitle: AppLocalization.localizedString("Quit"))
        if alert.runModal() == .alertFirstButtonReturn {
            PermissionGuidance.openSettings(for: SettingsPermissionKind.speechRecognition)
        }
        requestApplicationTermination()
    }
}

extension AppDelegate: NSWindowDelegate {
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        guard sender == mainWindowController?.window else { return true }
        sender.orderOut(nil)
        setMainWindowVisibility(false)
        return false
    }

    func windowWillClose(_ notification: Notification) {
        guard notification.object as? NSWindow == onboardingWindowController?.window else { return }
        onboardingWindowController = nil
    }

    func windowWillStartLiveResize(_ notification: Notification) {
        guard let window = notification.object as? NSWindow,
              window == mainWindowController?.window
        else {
            return
        }
        setTrafficLightButtonsHidden(true, for: window)
    }

    func windowDidEndLiveResize(_ notification: Notification) {
        guard let window = notification.object as? NSWindow,
              window == mainWindowController?.window
        else {
            return
        }
        setTrafficLightButtonsHidden(false, for: window)
        scheduleTrafficLightButtonPositionUpdate(for: window)
    }

    func windowDidMiniaturize(_ notification: Notification) {
        guard notification.object as? NSWindow == mainWindowController?.window else { return }
        setMainWindowVisibility(false)
    }

    func windowDidDeminiaturize(_ notification: Notification) {
        guard notification.object as? NSWindow == mainWindowController?.window else { return }
        setMainWindowVisibility(true)
    }
}

extension AppDelegate: NSMenuDelegate {
    func menuWillOpen(_ menu: NSMenu) {
        guard menu == statusItem?.menu else { return }
        isStatusMenuOpen = true
    }

    func menuDidClose(_ menu: NSMenu) {
        guard menu == statusItem?.menu else { return }
        isStatusMenuOpen = false

        let actions = pendingStatusMenuActions
        pendingStatusMenuActions.removeAll()

        // Let AppKit finish unwinding menu tracking before replacing the menu.
        DispatchQueue.main.async { [weak self] in
            if let self, self.statusMenuNeedsRebuild {
                self.buildMenu()
            }
            actions.forEach { $0() }
        }
    }
}
