//go:build windows

package main

import (
	"hash/crc32"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

const (
	dataFlowRender                = 0
	endpointVolumeSetMute         = 14
	endpointVolumeGetMute         = 15
	endpointVolumeHardwareSupport = 19
	endpointHardwareMute          = 0x2
)

var interfaceEndpointVolume = windows.GUID{Data1: 0x5cdf2c82, Data2: 0x841e, Data3: 0x4546, Data4: [8]byte{0x97, 0x22, 0x0c, 0xf7, 0x40, 0x78, 0x22, 0x9a}}

type wasapiOutputHardware struct{}

func (hardware *wasapiOutputHardware) defaultOutput() (proto.OutputDevice, bool) {
	enumerator, err := newMMDeviceEnumerator()
	if err != nil {
		return proto.OutputDevice{}, false
	}
	defer enumerator.release()
	device, err := enumerator.output(mmDeviceDefaultEndpoint, dataFlowRender, roleConsole)
	if err != nil {
		return proto.OutputDevice{}, false
	}
	defer device.release()
	return describeOutputDevice(device)
}

func (hardware *wasapiOutputHardware) device(uid string) (proto.OutputDevice, bool) {
	device, err := openOutputMMDevice(uid)
	if err != nil {
		return proto.OutputDevice{}, false
	}
	defer device.release()
	return describeOutputDevice(device)
}

func (hardware *wasapiOutputHardware) muted(device proto.OutputDevice) (bool, bool) {
	volume, err := openEndpointVolume(device.UID)
	if err != nil {
		return false, false
	}
	defer volume.release()
	var muted int32
	result := volume.call(endpointVolumeGetMute, uintptr(unsafe.Pointer(&muted)))
	return muted != 0, int32(result) >= 0
}

func (hardware *wasapiOutputHardware) settable(device proto.OutputDevice) bool {
	volume, err := openEndpointVolume(device.UID)
	if err != nil {
		return false
	}
	defer volume.release()
	var support uint32
	result := volume.call(endpointVolumeHardwareSupport, uintptr(unsafe.Pointer(&support)))
	return int32(result) >= 0 && support&endpointHardwareMute != 0
}

func (hardware *wasapiOutputHardware) setMuted(device proto.OutputDevice, muted bool) bool {
	volume, err := openEndpointVolume(device.UID)
	if err != nil {
		return false
	}
	defer volume.release()
	value := uintptr(0)
	if muted {
		value = 1
	}
	result := volume.call(endpointVolumeSetMute, value, 0)
	return int32(result) >= 0
}

func describeOutputDevice(device *comObject) (proto.OutputDevice, bool) {
	details, err := describeMMDevice(device)
	if err != nil {
		return proto.OutputDevice{}, false
	}
	identifier := crc32.ChecksumIEEE([]byte(details.ID))
	if identifier == 0 {
		identifier = 1
	}
	return proto.OutputDevice{ID: identifier, UID: details.ID, Name: details.Name}, true
}

func openOutputMMDevice(uid string) (*comObject, error) {
	enumerator, err := newMMDeviceEnumerator()
	if err != nil {
		return nil, err
	}
	defer enumerator.release()
	if uid == "" {
		return enumerator.output(mmDeviceDefaultEndpoint, dataFlowRender, roleConsole)
	}
	uidPointer, err := windows.UTF16PtrFromString(uid)
	if err != nil {
		return nil, err
	}
	return enumerator.output(mmDeviceGetDevice, uintptr(unsafe.Pointer(uidPointer)))
}

func openEndpointVolume(uid string) (*comObject, error) {
	device, err := openOutputMMDevice(uid)
	if err != nil {
		return nil, err
	}
	defer device.release()
	var volume *comObject
	result := device.call(mmDeviceActivate,
		uintptr(unsafe.Pointer(&interfaceEndpointVolume)),
		comClassContextAll,
		0,
		uintptr(unsafe.Pointer(&volume)),
	)
	if int32(result) < 0 {
		return nil, hresultError("IMMDevice.Activate(IAudioEndpointVolume)", result)
	}
	return volume, nil
}
