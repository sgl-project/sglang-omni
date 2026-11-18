import VoxtNewHelperKit
import CoreAudio
import Foundation

// The Mac's audio inputs, from Core Audio: capture.devices lists them
// (the default first, with the transport kind), findDevice finds one by
// its UID for capture.start, isAlive tells a connected one from a gone
// one. cpuSeconds is a recording's CPU use, for its log line.

// MARK: Devices (Core Audio)

func property<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                         scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal, _ value: inout T) -> Bool {
    var address = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<T>.size)
    return withUnsafeMutablePointer(to: &value) { AudioObjectGetPropertyData(object, &address, 0, nil, &size, $0) } == noErr
}

func stringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let value = UnsafeMutablePointer<Unmanaged<CFString>?>.allocate(capacity: 1)
    defer { value.deallocate() }
    value.initialize(to: nil)
    guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, value) == noErr, let s = value.pointee?.takeRetainedValue() else { return nil }
    return s as String
}

private func inputChannelCount(_ device: AudioObjectID) -> Int {
    var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamConfiguration,
                                             mScope: kAudioDevicePropertyScopeInput, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, raw) == noErr else { return 0 }
    let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
    return list.reduce(0) { $0 + Int($1.mNumberChannels) }
}

private func transportName(_ t: UInt32) -> String? {
    switch t {
    case kAudioDeviceTransportTypeBuiltIn: "builtin"
    case kAudioDeviceTransportTypeUSB: "usb"
    case kAudioDeviceTransportTypeBluetooth, kAudioDeviceTransportTypeBluetoothLE: "bluetooth"
    case kAudioDeviceTransportTypeAggregate, kAudioDeviceTransportTypeAutoAggregate: "aggregate"
    case kAudioDeviceTransportTypeVirtual: "virtual"
    case kAudioDeviceTransportTypeContinuityCaptureWired, kAudioDeviceTransportTypeContinuityCaptureWireless: "continuity"
    default: nil
    }
}

func inputDeviceIDs() -> (all: [AudioObjectID], defaultID: AudioObjectID) {
    var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                             mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    var ids: [AudioObjectID] = []
    if AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr {
        ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids)
    }
    var def = AudioObjectID(0)
    _ = property(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultInputDevice, &def)
    return (ids.filter { inputChannelCount($0) > 0 }, def)
}

/// The audio inputs, the default one first.
func captureDevices() -> [CaptureDevice] {
    let (ids, def) = inputDeviceIDs()
    var out: [CaptureDevice] = []
    for id in ids {
        guard let uid = stringProperty(id, kAudioDevicePropertyDeviceUID) else { continue }
        var rate = Float64(0)
        _ = property(id, kAudioDevicePropertyNominalSampleRate, &rate)
        var transport = UInt32(0)
        _ = property(id, kAudioDevicePropertyTransportType, &transport)
        out.append(CaptureDevice(id: uid, name: stringProperty(id, kAudioObjectPropertyName) ?? uid,
                                 channels: inputChannelCount(id), sampleRate: Int(rate), default: id == def,
                                 transport: transportName(transport)))
    }
    return out.sorted { $0.default && !$1.default }
}

func findDevice(uid: String) -> AudioObjectID? {
    inputDeviceIDs().all.first { stringProperty($0, kAudioDevicePropertyDeviceUID) == uid }
}

func deviceName(_ id: AudioObjectID) -> String {
    stringProperty(id, kAudioObjectPropertyName) ?? stringProperty(id, kAudioDevicePropertyDeviceUID) ?? "input \(id)"
}

func isAlive(_ id: AudioObjectID) -> Bool {
    var alive = UInt32(0)
    return property(id, kAudioDevicePropertyDeviceIsAlive, &alive) && alive != 0
}

/// CPU time of the process, in seconds.
func cpuSeconds() -> Double {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    func s(_ tv: timeval) -> Double { Double(tv.tv_sec) + Double(tv.tv_usec) / 1e6 }
    return s(u.ru_utime) + s(u.ru_stime)
}
