import SwiftUI
import AppKit
import AVFoundation
import CoreAudio

extension OnboardingGuideView {
    private var allRequiredPermissions: [OnboardingContextualPermission] {
        var permissions: [OnboardingContextualPermission] = [.microphone, .accessibility]
        if featureSettings.transcription.asrSelectionID.asrSelection == .dictation {
            permissions.append(.speechRecognition)
        }
        return permissions
    }

    var areRequiredPermissionsGranted: Bool {
        allRequiredPermissions.allSatisfy { isPermissionGranted($0) }
    }

    var permissionsGuidePanel: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 0)

            VStack(spacing: 16) {
                Text(currentStep.subtitle)
                    .font(.callout)
                    .foregroundStyle(OnboardingGuideStyle.secondaryText)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 560)

                permissionsActions
                    .frame(maxWidth: 520)
            }
            .frame(maxWidth: 560)

            Spacer(minLength: 0)

            permissionsFooter
                .frame(maxWidth: 520)
                .padding(.bottom, 12)
        }
        .padding(.horizontal, 24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var permissionsActions: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(allRequiredPermissions, id: \.self) { permission in
                permissionRow(permission)
            }

            userMainLanguageRow
        }
    }

    private var userMainLanguageRow: some View {
        HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(AppLocalization.localizedString("Your Main Language"))
                    .font(.subheadline.weight(.medium))
                Text(AppLocalization.localizedString("Languages prioritized for recognition. You can select multiple."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            Button {
                isUserMainLanguageDialogPresented = true
            } label: {
                OnboardingLanguageSelectLabel(summary: userMainLanguageSummary)
            }
            .buttonStyle(.plain)
            .help(AppLocalization.localizedString("Select User Languages"))
            .accessibilityLabel(AppLocalization.localizedString("Your Main Language"))
            .accessibilityValue(userMainLanguageSummary)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(SettingsUIStyle.controlFillColor)
        )
    }

    private func permissionRow(_ permission: OnboardingContextualPermission) -> some View {
        let isGranted = isPermissionGranted(permission)

        return HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(permission.titleKey)
                    .font(.subheadline.weight(.medium))
                Text(permission.descriptionKey)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            if permissionMonitoringKinds.contains(permission) {
                ProgressView()
                    .controlSize(.small)
            }

            if permission == .microphone, isGranted {
                microphonePermissionControl
            } else {
                OnboardingPermissionStatusBadge(isGranted: isGranted)
            }

            if !isGranted {
                Button(AppLocalization.localizedString("Allow")) {
                    requestPermission(permission)
                }
                .buttonStyle(SettingsCompactActionButtonStyle())
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(SettingsUIStyle.controlFillColor)
        )
    }

    @ViewBuilder
    private var microphonePermissionControl: some View {
        if let activeDevice = microphoneState.activeDevice,
           microphoneState.hasAvailableDevices {
            Button {
                isMicrophoneDialogPresented = true
            } label: {
                OnboardingMicrophoneSelectLabel(
                    deviceName: activeDevice.name,
                    hasDetectedAudio: microphoneHasDetectedAudio,
                    showsMenuIndicator: true
                )
            }
            .buttonStyle(.plain)
            .help(AppLocalization.localizedString("Switch Microphone"))
            .accessibilityLabel(AppLocalization.localizedString("Current Microphone"))
            .accessibilityValue(activeDevice.name)
        } else {
            OnboardingMicrophoneSelectLabel(
                deviceName: AppLocalization.localizedString("No valid microphone"),
                hasDetectedAudio: false,
                showsMenuIndicator: false
            )
            .help(AppLocalization.localizedString("No available microphone devices."))
        }
    }

    // MARK: - Follow-along practice

    private var permissionsFooter: some View {
        HStack {
            Spacer(minLength: 0)

            if let next = currentStep.next {
                Button {
                    currentStep = next
                } label: {
                    Label(AppLocalization.localizedString("Continue"), systemImage: "chevron.right")
                        .labelStyle(OnboardingGuideNextLabelStyle())
                }
                .buttonStyle(OnboardingGuidePrimaryButtonStyle())
                .disabled(!canContinue)
                .help(canContinue ? "" : continueDisabledHelp)
            }

            Spacer(minLength: 0)
        }
    }

    func isPermissionGranted(_ permission: OnboardingContextualPermission) -> Bool {
        _ = permissionRefreshRevision
        return OnboardingPermissionGrantResolver.isGranted(permission)
    }

    private func requestPermission(_ permission: OnboardingContextualPermission) {
        permissionMonitoringKinds.insert(permission)
        switch permission {
        case .microphone:
            AVCaptureDevice.requestAccess(for: .audio) { _ in
                Task { @MainActor in
                    permissionRefreshRevision += 1
                    startPermissionMonitoring(permission)
                    restartMicrophoneMeterIfNeeded()
                }
            }
        case .speechRecognition:
            startPermissionMonitoring(permission)
        case .accessibility:
            let granted = AccessibilityPermissionManager.request(prompt: true)
            if !granted {
                PermissionGuidance.openSettings(for: permission)
            }
            startPermissionMonitoring(permission)
        }
    }

    private func startPermissionMonitoring(_ permission: OnboardingContextualPermission) {
        permissionMonitorTasks[permission]?.cancel()
        permissionMonitorTasks[permission] = Task { @MainActor in
            for _ in 0..<30 {
                try? await Task.sleep(for: .milliseconds(500))
                permissionRefreshRevision += 1
                if isPermissionGranted(permission) {
                    permissionMonitoringKinds.remove(permission)
                    permissionMonitorTasks[permission] = nil
                    if permission == .microphone, microphoneCapture == nil {
                        restartMicrophoneMeterIfNeeded()
                    }
                    return
                }
            }
            permissionMonitoringKinds.remove(permission)
            permissionMonitorTasks[permission] = nil
        }
    }

    func refreshInputDevices() {
        let previousActiveUID = microphoneState.activeUID
        inputDevices = AudioInputDeviceManager.availableInputDevices()
        microphoneState = MicrophonePreferenceManager.syncState(
            defaults: .standard,
            availableDevices: inputDevices
        )
        if previousActiveUID != microphoneState.activeUID {
            resetMicrophoneDetection()
        }
    }

    func focusMicrophone(uid: String) {
        let previousActiveUID = microphoneState.activeUID
        microphoneState = MicrophonePreferenceManager.setFocusedDevice(
            uid: uid,
            defaults: .standard,
            availableDevices: inputDevices
        )
        if previousActiveUID != microphoneState.activeUID {
            resetMicrophoneDetection()
        }
        NotificationCenter.default.post(name: .voxtSelectedInputDeviceDidChange, object: nil)
    }

    func updateMicrophoneCapture() {
        refreshInputDevices()
        if shouldRunMicrophoneMeter {
            startMicrophoneMeter(
                preferredDeviceID: microphoneState.activeDevice?.id,
                resetStartupRetry: true
            )
        } else {
            stopMicrophoneMeter()
        }
    }

    private func startMicrophoneMeter(preferredDeviceID: AudioDeviceID?, resetStartupRetry: Bool) {
        guard currentStep == .permissions,
              OnboardingPermissionGrantResolver.isGranted(.microphone),
              microphoneState.activeDevice != nil
        else {
            stopMicrophoneMeter()
            return
        }

        stopMicrophoneMeter(resetStartupRetry: resetStartupRetry)
        if resetStartupRetry {
            microphoneStartupRetryCount = 0
        }
        microphoneReceivedInitialBuffer = false

        let capture = MeetingMicrophoneCapture()
        capture.setPreferredInputDevice(preferredDeviceID)
        microphoneCapture = capture

        do {
            try capture.start { _, level in
                Task { @MainActor in
                    guard currentStep == .permissions, microphoneCapture === capture else { return }
                    microphoneReceivedInitialBuffer = true
                    microphoneStartupWatchdogTask?.cancel()
                    microphoneStartupWatchdogTask = nil
                    guard !microphoneHasDetectedAudio else { return }
                    if level >= Self.microphoneSignalThreshold {
                        microphoneSignalFrameCount += 1
                    } else {
                        microphoneSignalFrameCount = 0
                    }
                    if microphoneSignalFrameCount >= Self.microphoneRequiredSignalFrames {
                        withAnimation(.easeOut(duration: 0.16)) {
                            microphoneHasDetectedAudio = true
                        }
                    }
                }
            }
            scheduleMicrophoneStartupWatchdog(preferredDeviceID: preferredDeviceID)
        } catch {
            VoxtLog.settingsWarning("Guide microphone meter failed: \(error.localizedDescription)")
            stopMicrophoneMeter()
        }
    }

    func stopMicrophoneMeter(resetStartupRetry: Bool = true) {
        microphoneStartupWatchdogTask?.cancel()
        microphoneStartupWatchdogTask = nil
        microphoneCapture?.stop()
        microphoneCapture = nil
        microphoneReceivedInitialBuffer = false
        if resetStartupRetry {
            microphoneStartupRetryCount = 0
        }
    }

    private func scheduleMicrophoneStartupWatchdog(preferredDeviceID: AudioDeviceID?) {
        microphoneStartupWatchdogTask?.cancel()
        microphoneStartupWatchdogTask = Task { @MainActor in
            do {
                try await Task.sleep(for: Self.microphoneStartupWatchdogDelay)
            } catch {
                return
            }

            guard !Task.isCancelled,
                  currentStep == .permissions,
                  !microphoneReceivedInitialBuffer,
                  microphoneStartupRetryCount < 1
            else {
                return
            }

            microphoneStartupRetryCount += 1
            VoxtLog.settingsWarning("Guide microphone meter restarting after missing initial callback.")
            startMicrophoneMeter(preferredDeviceID: nil, resetStartupRetry: false)
        }
    }

    private func restartMicrophoneMeterIfNeeded() {
        refreshInputDevices()
        if shouldRunMicrophoneMeter {
            startMicrophoneMeter(
                preferredDeviceID: microphoneState.activeDevice?.id,
                resetStartupRetry: true
            )
        } else {
            stopMicrophoneMeter()
            if currentStep == .permissions {
                resetMicrophoneDetection()
            }
        }
    }

    func scheduleMicrophoneRefresh() {
        microphoneRefreshTask?.cancel()
        microphoneRefreshTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .milliseconds(80))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            restartMicrophoneMeterIfNeeded()
            microphoneRefreshTask = nil
        }
    }

    private var shouldRunMicrophoneMeter: Bool {
        currentStep == .permissions &&
            OnboardingPermissionGrantResolver.isGranted(.microphone) &&
            microphoneState.activeDevice != nil
    }

    private func resetMicrophoneDetection() {
        microphoneSignalFrameCount = 0
        microphoneHasDetectedAudio = false
    }
}
