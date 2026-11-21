package main

import (
	"testing"

	"golang.org/x/sys/windows"
)

// Hand-declared COM interface IDs must equal the published Windows SDK
// values: a wrong byte is invisible until GetService answers E_NOINTERFACE
// for every device (see the capture IID fix).
func TestWellKnownInterfaceIDs(t *testing.T) {
	ids := map[string]windows.GUID{
		"IID_IAudioClient (1CB9AD4C-DBFA-4C32-B178-C2F568A703B2)":        interfaceAudioClient,
		"IID_IAudioCaptureClient (C8ADBD64-E71E-48A0-A4DE-185C395CD317)": interfaceAudioCaptureClient,
	}
	want := map[string]windows.GUID{
		"IID_IAudioClient (1CB9AD4C-DBFA-4C32-B178-C2F568A703B2)":        {Data1: 0x1cb9ad4c, Data2: 0xdbfa, Data3: 0x4c32, Data4: [8]byte{0xb1, 0x78, 0xc2, 0xf5, 0x68, 0xa7, 0x03, 0xb2}},
		"IID_IAudioCaptureClient (C8ADBD64-E71E-48A0-A4DE-185C395CD317)": {Data1: 0xc8adbd64, Data2: 0xe71e, Data3: 0x48a0, Data4: [8]byte{0xa4, 0xde, 0x18, 0x5c, 0x39, 0x5c, 0xd3, 0x17}},
	}
	for name, got := range ids {
		if got != want[name] {
			t.Errorf("%s: got %v, want the SDK value", name, got)
		}
	}
}
