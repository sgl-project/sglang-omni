//go:build windows

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

const (
	deviceStateActive            = 1
	comClassContextAll           = 23
	dataFlowCapture              = 1
	roleConsole                  = 0
	storageRead                  = 0
	audClientShareModeShared     = 0
	audClientStreamEventCallback = 0x00040000
	audClientBufferSilent        = 0x2
	propertyVariantLPWSTR        = 31
	mmDeviceEnumEndpoints        = 3
	mmDeviceDefaultEndpoint      = 4
	mmDeviceGetDevice            = 5
	mmDeviceActivate             = 3
	mmDeviceOpenPropertyStore    = 4
	mmDeviceGetID                = 5
	mmDeviceGetState             = 6
	mmDeviceCollectionCount      = 3
	mmDeviceCollectionItem       = 4
	propertyStoreGetValue        = 5
	audioClientInitialize        = 3
	audioClientGetMixFormat      = 8
	audioClientStart             = 10
	audioClientStop              = 11
	audioClientSetEventHandle    = 13
	audioClientGetService        = 14
	audioCaptureGetBuffer        = 3
	audioCaptureReleaseBuffer    = 4
	audioCaptureNextPacketSize   = 5
	waveFormatPCM                = 1
	waveFormatIEEEFloat          = 3
	waveFormatExtensible         = 0xfffe
)

var (
	propertyDeviceFriendlyName = propertyKey{
		format: windows.GUID{Data1: 0xa45c254e, Data2: 0xdf1c, Data3: 0x4efd, Data4: [8]byte{0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}},
		id:     14,
	}
	classMMDeviceEnumerator     = windows.GUID{Data1: 0xbcde0395, Data2: 0xe52f, Data3: 0x467c, Data4: [8]byte{0x8e, 0x3d, 0xc4, 0x57, 0x92, 0x91, 0x69, 0x2e}}
	interfaceMMDeviceEnumerator = windows.GUID{Data1: 0xa95664d2, Data2: 0x9614, Data3: 0x4f35, Data4: [8]byte{0xa7, 0x46, 0xde, 0x8d, 0xb6, 0x36, 0x17, 0xe6}}
	interfaceAudioClient        = windows.GUID{Data1: 0x1cb9ad4c, Data2: 0xdbfa, Data3: 0x4c32, Data4: [8]byte{0xb1, 0x78, 0xc2, 0xf5, 0x68, 0xa7, 0x03, 0xb2}}
	interfaceAudioCaptureClient = windows.GUID{Data1: 0xc8adbd64, Data2: 0xe71e, Data3: 0x48a0, Data4: [8]byte{0xa4, 0xde, 0x18, 0x5c, 0x27, 0xcd, 0x42, 0x7b}}
	procCoTaskMemFree           = ole32.NewProc("CoTaskMemFree")
	procPropVariantClear        = ole32.NewProc("PropVariantClear")
)

type propertyKey struct {
	format windows.GUID
	id     uint32
}

type propertyVariant struct {
	valueType uint16
	reserved1 uint16
	reserved2 uint16
	reserved3 uint16
	value     [2]uint64
}

type waveFormat struct {
	tag           uint16
	channels      int
	sampleRate    int
	bitsPerSample int
	blockAlign    int
}

func newMMDeviceEnumerator() (*comObject, error) {
	var enumerator *comObject
	result, _, _ := procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(&classMMDeviceEnumerator)),
		0,
		comClassContextAll,
		uintptr(unsafe.Pointer(&interfaceMMDeviceEnumerator)),
		uintptr(unsafe.Pointer(&enumerator)),
	)
	if int32(result) < 0 {
		return nil, hresultError("CoCreateInstance(MMDeviceEnumerator)", result)
	}
	return enumerator, nil
}

func captureDevices() ([]proto.CaptureDevice, error) {
	enumerator, err := newMMDeviceEnumerator()
	if err != nil {
		return nil, err
	}
	defer enumerator.release()
	defaultDevice, defaultErr := enumerator.output(mmDeviceDefaultEndpoint, dataFlowCapture, roleConsole)
	defaultID := ""
	if defaultErr == nil {
		defaultID, _ = mmDeviceID(defaultDevice)
		defaultDevice.release()
	}
	var collection *comObject
	result := enumerator.call(mmDeviceEnumEndpoints, dataFlowCapture, deviceStateActive, uintptr(unsafe.Pointer(&collection)))
	if int32(result) < 0 {
		return nil, hresultError("EnumAudioEndpoints", result)
	}
	defer collection.release()
	var count uint32
	if result := collection.call(mmDeviceCollectionCount, uintptr(unsafe.Pointer(&count))); int32(result) < 0 {
		return nil, hresultError("IMMDeviceCollection.GetCount", result)
	}
	devices := make([]proto.CaptureDevice, 0, count)
	for index := uint32(0); index < count; index++ {
		device, err := collection.output(mmDeviceCollectionItem, uintptr(index))
		if err != nil {
			continue
		}
		details, err := describeMMDevice(device)
		device.release()
		if err != nil {
			continue
		}
		details.Default = details.ID == defaultID
		devices = append(devices, details)
	}
	if len(devices) == 0 && defaultErr != nil {
		return nil, defaultErr
	}
	return devices, nil
}

func describeMMDevice(device *comObject) (proto.CaptureDevice, error) {
	id, err := mmDeviceID(device)
	if err != nil {
		return proto.CaptureDevice{}, err
	}
	details := proto.CaptureDevice{ID: id}
	var store *comObject
	result := device.call(mmDeviceOpenPropertyStore, storageRead, uintptr(unsafe.Pointer(&store)))
	if int32(result) < 0 {
		return details, hresultError("OpenPropertyStore", result)
	}
	var property propertyVariant
	result = store.call(propertyStoreGetValue, uintptr(unsafe.Pointer(&propertyDeviceFriendlyName)), uintptr(unsafe.Pointer(&property)))
	store.release()
	if int32(result) >= 0 {
		if property.valueType == propertyVariantLPWSTR {
			address := *(**uint16)(unsafe.Pointer(&property.value[0]))
			details.Name = windows.UTF16PtrToString(address)
		}
		procPropVariantClear.Call(uintptr(unsafe.Pointer(&property)))
	}
	client, err := activateAudioClient(device)
	if err == nil {
		format, formatErr := audioClientMixFormat(client)
		client.release()
		if formatErr == nil {
			details.Channels = format.channels
			details.SampleRate = format.sampleRate
		}
	}
	if details.Name == "" {
		details.Name = id
	}
	return details, nil
}

func mmDeviceID(device *comObject) (string, error) {
	var address *uint16
	result := device.call(mmDeviceGetID, uintptr(unsafe.Pointer(&address)))
	if int32(result) < 0 {
		return "", hresultError("IMMDevice.GetId", result)
	}
	defer procCoTaskMemFree.Call(uintptr(unsafe.Pointer(address)))
	return windows.UTF16PtrToString(address), nil
}

func activateAudioClient(device *comObject) (*comObject, error) {
	var client *comObject
	result := device.call(mmDeviceActivate,
		uintptr(unsafe.Pointer(&interfaceAudioClient)),
		comClassContextAll,
		0,
		uintptr(unsafe.Pointer(&client)),
	)
	if int32(result) < 0 {
		return nil, hresultError("IMMDevice.Activate(IAudioClient)", result)
	}
	return client, nil
}

func audioClientMixFormat(client *comObject) (waveFormat, error) {
	var address *byte
	result := client.call(audioClientGetMixFormat, uintptr(unsafe.Pointer(&address)))
	if int32(result) < 0 {
		return waveFormat{}, hresultError("IAudioClient.GetMixFormat", result)
	}
	defer procCoTaskMemFree.Call(uintptr(unsafe.Pointer(address)))
	return parseWaveFormat(address), nil
}

func parseWaveFormat(address *byte) waveFormat {
	header := unsafe.Slice(address, 18)
	length := 18
	if binary.LittleEndian.Uint16(header) == waveFormatExtensible && binary.LittleEndian.Uint16(header[16:]) >= 22 {
		length = 40
	}
	format, _ := parseFileWaveFormat(unsafe.Slice(address, length))
	return format
}

func (format waveFormat) decode(address *byte, frameCount int) ([]float64, error) {
	if frameCount == 0 {
		return nil, nil
	}
	if format.channels <= 0 || format.blockAlign <= 0 {
		return nil, fmt.Errorf("invalid capture format")
	}
	mono := make([]float64, frameCount)
	if address == nil {
		return mono, nil
	}
	bytes := unsafe.Slice(address, frameCount*format.blockAlign)
	for frame := 0; frame < frameCount; frame++ {
		var sum float64
		for channel := 0; channel < format.channels; channel++ {
			offset := frame*format.blockAlign + channel*(format.bitsPerSample/8)
			var sample float64
			switch {
			case format.tag == waveFormatIEEEFloat && format.bitsPerSample == 32:
				sample = float64(math.Float32frombits(binary.LittleEndian.Uint32(bytes[offset:])))
			case format.tag == waveFormatIEEEFloat && format.bitsPerSample == 64:
				sample = math.Float64frombits(binary.LittleEndian.Uint64(bytes[offset:]))
			case format.tag == waveFormatPCM && format.bitsPerSample == 16:
				sample = float64(int16(binary.LittleEndian.Uint16(bytes[offset:]))) / 32768
			case format.tag == waveFormatPCM && format.bitsPerSample == 24:
				value := int32(bytes[offset]) | int32(bytes[offset+1])<<8 | int32(bytes[offset+2])<<16
				if value&0x800000 != 0 {
					value |= ^0xffffff
				}
				sample = float64(value) / 8388608
			case format.tag == waveFormatPCM && format.bitsPerSample == 32:
				sample = float64(int32(binary.LittleEndian.Uint32(bytes[offset:]))) / 2147483648
			default:
				return nil, fmt.Errorf("unsupported capture format tag=%d bits=%d", format.tag, format.bitsPerSample)
			}
			sum += sample
		}
		mono[frame] = sum / float64(format.channels)
	}
	return mono, nil
}

type wasapiStream struct {
	device  *comObject
	client  *comObject
	capture *comObject
	event   windows.Handle
	format  waveFormat
}

func openWasapiStream(deviceID string) (*wasapiStream, proto.CaptureDevice, error) {
	enumerator, err := newMMDeviceEnumerator()
	if err != nil {
		return nil, proto.CaptureDevice{}, err
	}
	defer enumerator.release()
	deviceIDPointer, err := windows.UTF16PtrFromString(deviceID)
	if err != nil {
		return nil, proto.CaptureDevice{}, err
	}
	device, err := enumerator.output(mmDeviceGetDevice, uintptr(unsafe.Pointer(deviceIDPointer)))
	if err != nil {
		return nil, proto.CaptureDevice{}, err
	}
	stream := &wasapiStream{device: device}
	opened := false
	defer func() {
		if !opened {
			stream.close()
		}
	}()
	details, err := describeMMDevice(device)
	if err != nil {
		return nil, proto.CaptureDevice{}, err
	}
	client, err := activateAudioClient(device)
	if err != nil {
		return nil, proto.CaptureDevice{}, err
	}
	stream.client = client
	formatAddress, err := mixFormatAddress(client)
	if err != nil {
		stream.close()
		return nil, proto.CaptureDevice{}, err
	}
	format := parseWaveFormat(formatAddress)
	result := client.call(audioClientInitialize,
		audClientShareModeShared,
		audClientStreamEventCallback,
		0,
		0,
		uintptr(unsafe.Pointer(formatAddress)),
		0,
	)
	procCoTaskMemFree.Call(uintptr(unsafe.Pointer(formatAddress)))
	if int32(result) < 0 {
		stream.close()
		return nil, proto.CaptureDevice{}, hresultError("IAudioClient.Initialize", result)
	}
	stream.format = format
	event, err := windows.CreateEventEx(nil, nil, 0, windows.EVENT_MODIFY_STATE|windows.SYNCHRONIZE)
	if err != nil {
		stream.close()
		return nil, proto.CaptureDevice{}, err
	}
	stream.event = event
	if result := client.call(audioClientSetEventHandle, uintptr(event)); int32(result) < 0 {
		stream.close()
		return nil, proto.CaptureDevice{}, hresultError("IAudioClient.SetEventHandle", result)
	}
	result = client.call(audioClientGetService,
		uintptr(unsafe.Pointer(&interfaceAudioCaptureClient)),
		uintptr(unsafe.Pointer(&stream.capture)),
	)
	if int32(result) < 0 {
		stream.close()
		return nil, proto.CaptureDevice{}, hresultError("IAudioClient.GetService", result)
	}
	if result := client.call(audioClientStart); int32(result) < 0 {
		stream.close()
		return nil, proto.CaptureDevice{}, hresultError("IAudioClient.Start", result)
	}
	opened = true
	return stream, details, nil
}

func mixFormatAddress(client *comObject) (*byte, error) {
	var address *byte
	result := client.call(audioClientGetMixFormat, uintptr(unsafe.Pointer(&address)))
	if int32(result) < 0 {
		return nil, hresultError("IAudioClient.GetMixFormat", result)
	}
	return address, nil
}

func (stream *wasapiStream) readPacket() ([]float64, error) {
	var packetFrames uint32
	result := stream.capture.call(audioCaptureNextPacketSize, uintptr(unsafe.Pointer(&packetFrames)))
	if int32(result) < 0 {
		return nil, hresultError("IAudioCaptureClient.GetNextPacketSize", result)
	}
	if packetFrames == 0 {
		return nil, nil
	}
	var address *byte
	var frames, flags uint32
	var devicePosition, queryPosition uint64
	result = stream.capture.call(audioCaptureGetBuffer,
		uintptr(unsafe.Pointer(&address)),
		uintptr(unsafe.Pointer(&frames)),
		uintptr(unsafe.Pointer(&flags)),
		uintptr(unsafe.Pointer(&devicePosition)),
		uintptr(unsafe.Pointer(&queryPosition)),
	)
	if int32(result) < 0 {
		return nil, hresultError("IAudioCaptureClient.GetBuffer", result)
	}
	var samples []float64
	var decodeErr error
	if flags&audClientBufferSilent != 0 {
		samples = make([]float64, frames)
	} else {
		samples, decodeErr = stream.format.decode(address, int(frames))
	}
	releaseResult := stream.capture.call(audioCaptureReleaseBuffer, uintptr(frames))
	if int32(releaseResult) < 0 && decodeErr == nil {
		decodeErr = hresultError("IAudioCaptureClient.ReleaseBuffer", releaseResult)
	}
	return samples, decodeErr
}

func (stream *wasapiStream) active() bool {
	var state uint32
	result := stream.device.call(mmDeviceGetState, uintptr(unsafe.Pointer(&state)))
	return int32(result) >= 0 && state == deviceStateActive
}

func (stream *wasapiStream) close() {
	if stream == nil {
		return
	}
	if stream.client != nil {
		stream.client.call(audioClientStop)
		stream.client.release()
		stream.client = nil
	}
	if stream.capture != nil {
		stream.capture.release()
		stream.capture = nil
	}
	if stream.device != nil {
		stream.device.release()
		stream.device = nil
	}
	if stream.event != 0 {
		windows.CloseHandle(stream.event)
		stream.event = 0
	}
}
