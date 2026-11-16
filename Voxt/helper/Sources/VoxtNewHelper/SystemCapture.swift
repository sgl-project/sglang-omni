import VoxtNewHelperKit
@preconcurrency import AVFoundation
import CoreAudio
import Foundation

// capture.system's input: a Core Audio process tap (macOS 14.2+), as
// Voxt 1.x's MeetingSystemAudioCapture. The tap takes every process's
// sound but the helper's and VoxtNew's, mixed to mono, private and left
// audible; a private aggregate device around the default output carries
// it (tap auto-start, drift compensation) and an IOProc reads it. Unlike
// 1.x it follows the default output: a change calls outputChanged, and
// CaptureRun builds a new tap. The permission is "System Audio
// Recording" (NSAudioCaptureUsageDescription), which macOS asks for
// itself when the tap is made; never Screen Recording.
//
// Logs name the output by its number only: a Bluetooth device's name
// (and so its UID's text) may hold a person's name.

/// The default output, nil when there is none.
func defaultOutput() -> (id: AudioObjectID, uid: String)? {
    var id = AudioObjectID(0)
    guard property(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice, &id), id != 0,
          let uid = stringProperty(id, kAudioDevicePropertyDeviceUID) else { return nil }
    return (id, uid)
}

/// The Core Audio process object of pid, nil when it has none (a process
/// that never played sound).
private func processObject(_ pid: Int32) -> AudioObjectID? {
    var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
                                             mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var pid = pid
    var object = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                            UInt32(MemoryLayout<Int32>.size), &pid, &size, &object)
    return status == noErr && object != kAudioObjectUnknown ? object : nil
}

private func check(_ status: OSStatus, _ what: String) throws {
    if status != noErr { throw RPCError(Code.deviceUnavailable, "system audio: \(what) failed (\(status))") }
}

/// One tap on the default output, from init to stop().
final class SystemTap: @unchecked Sendable {
    let deviceID: AudioObjectID
    let outputUID: String
    /// The tap's audio: float, mono.
    let format: AVAudioFormat
    private let queue = DispatchQueue(label: "voxtnew.capture.system", qos: .userInteractive)
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var listener: AudioObjectPropertyListenerBlock?
    private var defaultOutputAddress = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                                                  mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)

    /// Makes the tap and its aggregate device; device_unavailable without
    /// an output.
    init(ownPIDs: [Int32], outputChanged: @escaping @Sendable () -> Void) throws {
        guard let out = defaultOutput() else { throw RPCError(Code.deviceUnavailable, "No output device to record from") }
        deviceID = out.id
        outputUID = out.uid
        let plan = systemTapPlan(ownPIDs: ownPIDs, outputUID: out.uid)
        let desc = CATapDescription(monoGlobalTapButExcludeProcesses: plan.excludedPIDs.compactMap(processObject))
        desc.name = "VoxtNew system audio"
        desc.isPrivate = plan.isPrivate
        desc.isMixdown = plan.mixdown
        desc.isMono = plan.mono
        desc.isExclusive = plan.exclusive
        desc.muteBehavior = plan.muted ? .muted : .unmuted
        desc.deviceUID = plan.outputUID
        var tap = AudioObjectID(kAudioObjectUnknown), agg = AudioObjectID(kAudioObjectUnknown)
        let f: AVAudioFormat
        do {
            try check(AudioHardwareCreateProcessTap(desc, &tap), "the tap")
            var asbd = AudioStreamBasicDescription()
            guard property(tap, kAudioTapPropertyFormat, &asbd), let tf = AVAudioFormat(streamDescription: &asbd) else {
                throw RPCError(Code.deviceUnavailable, "system audio: the tap has no format")
            }
            f = tf
            let aggregate: [String: Any] = [
                kAudioAggregateDeviceNameKey: "VoxtNew system audio",
                kAudioAggregateDeviceUIDKey: "voxtnew.system." + UUID().uuidString,
                kAudioAggregateDeviceMainSubDeviceKey: plan.outputUID,
                kAudioAggregateDeviceIsPrivateKey: true,
                kAudioAggregateDeviceIsStackedKey: false,
                kAudioAggregateDeviceTapAutoStartKey: true,
                kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: plan.outputUID]],
                kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true, kAudioSubTapUIDKey: desc.uuid.uuidString]],
            ]
            try check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &agg), "the aggregate device")
        } catch {
            if tap != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tap) }
            throw error
        }
        tapID = tap
        aggregateID = agg
        format = f
        let block: AudioObjectPropertyListenerBlock = { _, _ in outputChanged() }
        if AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &defaultOutputAddress, queue, block) == noErr {
            listener = block
        }
    }

    /// Starts the IOProc: each buffer goes to deliver, on the tap's queue.
    func start(_ deliver: @escaping @Sendable (AVAudioPCMBuffer) -> Void) throws {
        let format = format
        try check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, queue) { _, input, _, _, _ in
            if let b = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: input, deallocator: nil) { deliver(b) }
        }, "the IOProc")
        try check(AudioDeviceStart(aggregateID, procID), "starting")
    }

    /// Ends the tap: the IOProc, the aggregate device, then the tap.
    func stop() {
        if let l = listener {
            AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &defaultOutputAddress, queue, l)
            listener = nil
        }
        if let p = procID {
            AudioDeviceStop(aggregateID, p)
            AudioDeviceDestroyIOProcID(aggregateID, p)
            procID = nil
        }
        if aggregateID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }
}
