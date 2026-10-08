// AudioInputDeviceManager.swift
// Provides Audio Input Device Manager for core app behavior.

import Foundation
import CoreAudio
import AVFoundation

struct AudioInputDevice: Identifiable, Hashable, Sendable {
    let id: AudioDeviceID
    let uid: String
    let name: String

    var identifier: String { uid }
}

enum AudioInputDeviceManager {
    static func availableInputDevices() -> [AudioInputDevice] {
        let devices = snapshotAvailableInputDevices()
        VoxtLog.audio("Audio input devices discovered: \(devices.count)", verbose: true)
        return devices
    }

    nonisolated static func snapshotAvailableInputDevices() -> [AudioInputDevice] {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var dataSize: UInt32 = 0
        let sizeStatus = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &dataSize
        )
        guard sizeStatus == noErr, dataSize > 0 else { return [] }

        let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = Array(repeating: AudioDeviceID(0), count: deviceCount)
        let listStatus = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &dataSize,
            &deviceIDs
        )
        guard listStatus == noErr else { return [] }

        let discoveredDevices: [AudioInputDevice] = deviceIDs.compactMap { (id: AudioDeviceID) -> AudioInputDevice? in
            guard hasInputStream(deviceID: id) else { return nil }
            guard let uid = deviceUID(deviceID: id), !uid.isEmpty else { return nil }
            guard let name = deviceName(deviceID: id), !name.isEmpty else { return nil }
            return AudioInputDevice(id: id, uid: uid, name: name)
        }

        let devices = discoveredDevices
            .filter { shouldIncludeInSnapshot(uid: $0.uid, name: $0.name) }
            .sorted { (lhs: AudioInputDevice, rhs: AudioInputDevice) in
                lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
        return devices
    }

    nonisolated static func shouldIncludeInSnapshot(uid: String, name: String) -> Bool {
        let trimmedUID = uid.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedUID.isEmpty, !trimmedName.isEmpty else { return false }

        if trimmedUID.hasPrefix("voxt-process-tap-") || trimmedName == "VoxtProcessTap" {
            return false
        }

        // CoreAudio can expose internal aggregate routing devices as input streams.
        // They are not real microphones and should not participate in auto-switch.
        if trimmedUID.hasPrefix("CADefaultDeviceAggregate-"),
           trimmedName.hasPrefix("CADefaultDeviceAggregate-") {
            return false
        }

        return true
    }

    static func defaultInputDeviceID() -> AudioDeviceID? {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var deviceID = AudioDeviceID(0)
        var dataSize = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &dataSize,
            &deviceID
        )
        guard status == noErr, deviceID != 0 else {
            VoxtLog.audioWarning("Failed to read default input device. status=\(status), deviceID=\(deviceID)")
            return nil
        }
        return deviceID
    }

    static func defaultInputDeviceUID(from devices: [AudioInputDevice]? = nil) -> String? {
        guard let defaultID = defaultInputDeviceID() else { return nil }
        if let devices, let device = devices.first(where: { $0.id == defaultID }) {
            return device.uid
        }
        return deviceUID(deviceID: defaultID)
    }

    static func resolvedInputDeviceID(
        from devices: [AudioInputDevice],
        preferredID: AudioDeviceID?
    ) -> AudioDeviceID? {
        resolvedInputDeviceID(
            from: devices,
            preferredID: preferredID,
            defaultDeviceID: defaultInputDeviceID()
        )
    }

    static func resolvedInputDeviceID(
        from devices: [AudioInputDevice],
        preferredID: AudioDeviceID?,
        defaultDeviceID: AudioDeviceID?
    ) -> AudioDeviceID? {
        if let preferredID,
           devices.contains(where: { $0.id == preferredID }) {
            return preferredID
        }

        if let defaultDeviceID,
           devices.contains(where: { $0.id == defaultDeviceID }) {
            return defaultDeviceID
        }

        return devices.first?.id
    }

    static func isAvailableInputDevice(_ deviceID: AudioDeviceID) -> Bool {
        snapshotAvailableInputDevices().contains(where: { $0.id == deviceID })
    }

    nonisolated static func nominalSampleRate(for deviceID: AudioDeviceID?) -> Double? {
        guard let deviceID,
              deviceID != AudioDeviceID(kAudioObjectUnknown)
        else {
            return nil
        }

        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var sampleRate = Float64(0)
        var dataSize = UInt32(MemoryLayout<Float64>.size)
        let status = AudioObjectGetPropertyData(
            deviceID,
            &propertyAddress,
            0,
            nil,
            &dataSize,
            &sampleRate
        )
        guard status == noErr, sampleRate.isFinite, sampleRate > 0 else {
            return nil
        }
        return sampleRate
    }

    nonisolated static func captureTapFormat(
        nodeOutputFormat: AVAudioFormat,
        hardwareSampleRate: Double?
    ) -> AVAudioFormat {
        // AVAudioInputNode can report 48 kHz even when the active input hardware is
        // running at 44.1 kHz. Installing a tap with the node format in that state
        // raises "Input HW format and tap format not matching", so prefer the
        // hardware sample rate while preserving the node channel/layout format.
        guard let hardwareSampleRate,
              hardwareSampleRate.isFinite,
              hardwareSampleRate > 0,
              abs(hardwareSampleRate - nodeOutputFormat.sampleRate) > 1
        else {
            return nodeOutputFormat
        }

        return AVAudioFormat(
            commonFormat: nodeOutputFormat.commonFormat,
            sampleRate: hardwareSampleRate,
            channels: nodeOutputFormat.channelCount,
            interleaved: nodeOutputFormat.isInterleaved
        ) ?? nodeOutputFormat
    }

    static func makeDevicesObserver(onChange: @escaping @Sendable () -> Void) -> AudioInputDeviceObserver? {
        AudioInputDeviceObserver(onChange: onChange)
    }

    nonisolated static func selectorDescription(_ selector: AudioObjectPropertySelector) -> String {
        switch selector {
        case kAudioHardwarePropertyDevices:
            return "devices"
        case kAudioHardwarePropertyDefaultInputDevice:
            return "defaultInputDevice"
        default:
            return String(selector)
        }
    }

    nonisolated private static func hasInputStream(deviceID: AudioDeviceID) -> Bool {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )

        var dataSize: UInt32 = 0
        let status = AudioObjectGetPropertyDataSize(deviceID, &propertyAddress, 0, nil, &dataSize)
        return status == noErr && dataSize > 0
    }

    nonisolated private static func deviceName(deviceID: AudioDeviceID) -> String? {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var buffer = [CChar](repeating: 0, count: 256)
        var dataSize = UInt32(buffer.count * MemoryLayout<CChar>.size)
        let status = AudioObjectGetPropertyData(deviceID, &propertyAddress, 0, nil, &dataSize, &buffer)
        guard status == noErr else { return nil }
        return String(cString: buffer)
    }

    nonisolated private static func deviceUID(deviceID: AudioDeviceID) -> String? {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var unmanagedUID: Unmanaged<CFString>?
        var dataSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = withUnsafeMutablePointer(to: &unmanagedUID) { pointer in
            AudioObjectGetPropertyData(
                deviceID,
                &propertyAddress,
                0,
                nil,
                &dataSize,
                pointer
            )
        }
        guard status == noErr, let unmanagedUID else { return nil }
        return unmanagedUID.takeUnretainedValue() as String
    }
}

final class AudioInputDeviceObserver {
    private let queue = DispatchQueue(label: "com.voxt.audio-input-devices")
    private let onChange: @Sendable () -> Void
    private let block: AudioObjectPropertyListenerBlock
    private var devicesAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    private var defaultInputAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    private var isRegistered = false

    init?(onChange: @escaping @Sendable () -> Void) {
        self.onChange = onChange
        self.block = { numberAddresses, addresses in
            let selectorValues = UnsafeBufferPointer(start: addresses, count: Int(numberAddresses)).map {
                AudioInputDeviceManager.selectorDescription($0.mSelector)
            }
            let selectors = selectorValues.isEmpty ? "unknown" : selectorValues.joined(separator: ",")
            VoxtLog.audio("Audio input device observer fired. selectors=\(selectors)", verbose: true)
            onChange()
        }

        let systemObjectID = AudioObjectID(kAudioObjectSystemObject)
        let devicesStatus = AudioObjectAddPropertyListenerBlock(
            systemObjectID,
            &devicesAddress,
            queue,
            block
        )
        let defaultInputStatus = AudioObjectAddPropertyListenerBlock(
            systemObjectID,
            &defaultInputAddress,
            queue,
            block
        )

        guard devicesStatus == noErr, defaultInputStatus == noErr else {
            if devicesStatus == noErr {
                AudioObjectRemovePropertyListenerBlock(systemObjectID, &devicesAddress, queue, block)
            }
            if defaultInputStatus == noErr {
                AudioObjectRemovePropertyListenerBlock(systemObjectID, &defaultInputAddress, queue, block)
            }
            VoxtLog.audioWarning(
                "Failed to register audio device observer. devicesStatus=\(devicesStatus), defaultInputStatus=\(defaultInputStatus)"
            )
            return nil
        }

        isRegistered = true
        VoxtLog.audio("Audio input device observer registered.")
    }

    deinit {
        stop()
    }

    func stop() {
        guard isRegistered else { return }
        isRegistered = false
        let systemObjectID = AudioObjectID(kAudioObjectSystemObject)
        AudioObjectRemovePropertyListenerBlock(systemObjectID, &devicesAddress, queue, block)
        AudioObjectRemovePropertyListenerBlock(systemObjectID, &defaultInputAddress, queue, block)
        VoxtLog.audio("Audio input device observer removed.", verbose: true)
    }
}
