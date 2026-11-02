// MeetingSystemAudioCapture.swift
// Provides Meeting System Audio Capture for meeting capture.

import Foundation
import AVFoundation
import AudioToolbox
import CoreAudio

final class MeetingSystemAudioCapture: @unchecked Sendable {
    enum CaptureError: LocalizedError {
        case outputDeviceUnavailable(OSStatus)
        case tapCreationFailed(OSStatus)
        case aggregateDeviceCreationFailed(OSStatus)
        case ioProcCreationFailed(OSStatus)
        case startFailed(OSStatus)
        case invalidTapFormat

        var errorDescription: String? {
            switch self {
            case .outputDeviceUnavailable(let status):
                return AppLocalization.format("System output device unavailable (%d).", Int(status))
            case .tapCreationFailed(let status):
                return AppLocalization.format("System audio tap creation failed (%d).", Int(status)) + " " + Self.accessGuidance
            case .aggregateDeviceCreationFailed(let status):
                return AppLocalization.format("System audio aggregate device failed (%d).", Int(status))
            case .ioProcCreationFailed(let status):
                return AppLocalization.format("System audio callback setup failed (%d).", Int(status))
            case .startFailed(let status):
                return AppLocalization.format("System audio capture failed to start (%d).", Int(status)) + " " + Self.accessGuidance
            case .invalidTapFormat:
                return AppLocalization.localizedString("System audio tap returned an invalid format.")
            }
        }

        private static var accessGuidance: String {
            AppLocalization.localizedString("If access was denied, enable System Audio Recording for Voxt in System Settings, then retry the meeting.")
        }
    }

    private let callbackQueue = DispatchQueue(label: "com.voxt.meeting.system-audio", qos: .userInteractive)
    private let lock = NSLock()
    private var aggregateDeviceID = AudioDeviceID(kAudioObjectUnknown)
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var callback: ((AVAudioPCMBuffer, Float) -> Void)?
    private var hasLoggedFirstCallback = false

    deinit {
        stop()
    }

    func start(onBuffer: @escaping (AVAudioPCMBuffer, Float) -> Void) throws {
        stop()
        callback = onBuffer
        hasLoggedFirstCallback = false

        let outputDeviceID = try Self.defaultOutputDeviceID()
        let outputUID = try Self.deviceUID(for: outputDeviceID)
        let tapUUID = UUID()

        let tapDescription = CATapDescription()
        tapDescription.name = "Voxt Meeting System Audio"
        tapDescription.uuid = tapUUID
        tapDescription.processes = Self.currentProcessObjectID().map { [$0] } ?? []
        tapDescription.isPrivate = true
        tapDescription.muteBehavior = .unmuted
        tapDescription.isMixdown = true
        tapDescription.isMono = true
        tapDescription.isExclusive = true
        tapDescription.deviceUID = outputUID
        tapDescription.stream = 0

        var tapID = AudioObjectID(kAudioObjectUnknown)
        let tapCreateStatus = AudioHardwareCreateProcessTap(tapDescription, &tapID)
        guard tapCreateStatus == noErr else {
            throw CaptureError.tapCreationFailed(tapCreateStatus)
        }

        let aggregateUID = UUID().uuidString
        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey: "VoxtMeetingSystemAudio",
            kAudioAggregateDeviceUIDKey: aggregateUID,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [
                [kAudioSubDeviceUIDKey: outputUID]
            ],
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapDriftCompensationKey: true,
                    kAudioSubTapUIDKey: tapUUID.uuidString
                ]
            ]
        ]

        var aggregateDeviceID = AudioDeviceID(kAudioObjectUnknown)
        let aggregateStatus = AudioHardwareCreateAggregateDevice(
            aggregateDescription as CFDictionary,
            &aggregateDeviceID
        )
        guard aggregateStatus == noErr else {
            AudioHardwareDestroyProcessTap(tapID)
            throw CaptureError.aggregateDeviceCreationFailed(aggregateStatus)
        }

        let streamDescription: AudioStreamBasicDescription
        do {
            streamDescription = try Self.tapStreamDescription(for: tapID)
        } catch {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            AudioHardwareDestroyProcessTap(tapID)
            throw error
        }
        var mutableStreamDescription = streamDescription
        guard let format = AVAudioFormat(streamDescription: &mutableStreamDescription) else {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            AudioHardwareDestroyProcessTap(tapID)
            throw CaptureError.invalidTapFormat
        }

        var ioProcID: AudioDeviceIOProcID?
        let ioProcStatus = AudioDeviceCreateIOProcIDWithBlock(
            &ioProcID,
            aggregateDeviceID,
            callbackQueue
        ) { [weak self] _, inInputData, _, _, _ in
            self?.handleInputData(inInputData, format: format)
        }
        guard ioProcStatus == noErr, let ioProcID else {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            AudioHardwareDestroyProcessTap(tapID)
            throw CaptureError.ioProcCreationFailed(ioProcStatus)
        }

        let startStatus = AudioDeviceStart(aggregateDeviceID, ioProcID)
        guard startStatus == noErr else {
            AudioDeviceDestroyIOProcID(aggregateDeviceID, ioProcID)
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            AudioHardwareDestroyProcessTap(tapID)
            throw CaptureError.startFailed(startStatus)
        }

        lock.lock()
        self.tapID = tapID
        self.aggregateDeviceID = aggregateDeviceID
        self.ioProcID = ioProcID
        lock.unlock()
    }

    func stop() {
        lock.lock()
        let aggregateDeviceID = self.aggregateDeviceID
        let tapID = self.tapID
        let ioProcID = self.ioProcID
        self.aggregateDeviceID = AudioDeviceID(kAudioObjectUnknown)
        self.tapID = AudioObjectID(kAudioObjectUnknown)
        self.ioProcID = nil
        callback = nil
        hasLoggedFirstCallback = false
        lock.unlock()

        if aggregateDeviceID != AudioDeviceID(kAudioObjectUnknown) {
            if let ioProcID {
                AudioDeviceStop(aggregateDeviceID, ioProcID)
                AudioDeviceDestroyIOProcID(aggregateDeviceID, ioProcID)
            }
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
        }

        if tapID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tapID)
        }
    }

    private func handleInputData(
        _ inputData: UnsafePointer<AudioBufferList>,
        format: AVAudioFormat
    ) {
        let sourceBuffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inputData))
        let streamDescription = format.streamDescription
        let bytesPerFrame = Int(streamDescription.pointee.mBytesPerFrame)
        guard bytesPerFrame > 0, let firstBuffer = sourceBuffers.first else { return }

        let frameCount = AVAudioFrameCount(Int(firstBuffer.mDataByteSize) / bytesPerFrame)
        guard frameCount > 0 else { return }

        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return
        }
        pcmBuffer.frameLength = frameCount

        let destinationBuffers = UnsafeMutableAudioBufferListPointer(pcmBuffer.mutableAudioBufferList)
        guard destinationBuffers.count == sourceBuffers.count else { return }

        for index in 0..<sourceBuffers.count {
            let source = sourceBuffers[index]
            let copySize = min(Int(source.mDataByteSize), Int(destinationBuffers[index].mDataByteSize))
            guard copySize > 0,
                  let sourceData = source.mData,
                  let destinationData = destinationBuffers[index].mData
            else {
                continue
            }

            memcpy(destinationData, sourceData, copySize)
            destinationBuffers[index].mDataByteSize = UInt32(copySize)
        }

        if !hasLoggedFirstCallback {
            hasLoggedFirstCallback = true
            VoxtLog.meeting(
                "Meeting system audio callback received. sampleRate=\(Int(format.sampleRate)), channels=\(format.channelCount), frames=\(pcmBuffer.frameLength)",
                verbose: true
            )
        }
        callback?(pcmBuffer, Self.normalizedRMS(from: pcmBuffer))
    }

    private static func tapStreamDescription(for tapID: AudioObjectID) throws -> AudioStreamBasicDescription {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var description = AudioStreamBasicDescription()
        var dataSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let status = AudioObjectGetPropertyData(
            tapID,
            &propertyAddress,
            0,
            nil,
            &dataSize,
            &description
        )
        guard status == noErr else {
            throw CaptureError.invalidTapFormat
        }
        return description
    }

    private static func defaultOutputDeviceID() throws -> AudioDeviceID {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = AudioDeviceID(kAudioObjectUnknown)
        var dataSize = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &dataSize,
            &deviceID
        )
        guard status == noErr, deviceID != AudioDeviceID(kAudioObjectUnknown) else {
            throw CaptureError.outputDeviceUnavailable(status)
        }
        return deviceID
    }

    private static func deviceUID(for deviceID: AudioDeviceID) throws -> String {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: Unmanaged<CFString>?
        var dataSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = AudioObjectGetPropertyData(
            deviceID,
            &propertyAddress,
            0,
            nil,
            &dataSize,
            &uid
        )
        guard status == noErr else {
            throw CaptureError.outputDeviceUnavailable(status)
        }
        guard let uid else {
            throw CaptureError.outputDeviceUnavailable(kAudioHardwareUnspecifiedError)
        }
        return uid.takeUnretainedValue() as String
    }

    private static func currentProcessObjectID() -> AudioObjectID? {
        var pid = getpid()
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var processObjectID = AudioObjectID(kAudioObjectUnknown)
        var dataSize = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = withUnsafePointer(to: &pid) { pidPointer in
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject),
                &address,
                UInt32(MemoryLayout<pid_t>.size),
                pidPointer,
                &dataSize,
                &processObjectID
            )
        }
        guard status == noErr, processObjectID != AudioObjectID(kAudioObjectUnknown) else {
            return nil
        }
        return processObjectID
    }

    private static func normalizedRMS(from buffer: AVAudioPCMBuffer) -> Float {
        AudioLevelMeter.normalizedLevel(from: buffer)
    }
}
