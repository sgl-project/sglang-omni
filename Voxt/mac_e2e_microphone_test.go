package main

import (
	"os"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// TestMacHelperMicrophoneSelection exercises the real input with and
// without its UID. It requires microphone permission granted in advance.
// VOXTNEW_MAC_MIC_UID selects a non-default device for the format-mismatch
// regression: its reported rate must match the hardware and produce PCM
// without an engine restart.
func TestMacHelperMicrophoneSelection(t *testing.T) {
	if os.Getenv("VOXTNEW_MAC_MIC_CAPTURE") != "1" {
		t.Skip("set VOXTNEW_MAC_MIC_CAPTURE=1 to record from the real microphone")
	}
	bin := macE2E(t)
	h := startMacHelper(t, bin, envCaptureFile+"=")
	if perms := macCall[proto.PermissionsResult](t, h, proto.MethodPermissionsStatus, nil); perms.Microphone != "granted" {
		t.Fatalf("the helper needs microphone permission, got %q", perms.Microphone)
	}
	devices := macCall[proto.DevicesResult](t, h, proto.MethodCaptureDevices, nil).Devices
	requestedDeviceID := os.Getenv("VOXTNEW_MAC_MIC_UID")
	index := slices.IndexFunc(devices, func(device proto.CaptureDevice) bool { return device.Default })
	if requestedDeviceID != "" {
		index = slices.IndexFunc(devices, func(device proto.CaptureDevice) bool { return device.ID == requestedDeviceID })
		if index < 0 {
			t.Fatalf("VOXTNEW_MAC_MIC_UID did not match an input: %q", requestedDeviceID)
		}
		if devices[index].Default {
			t.Fatal("VOXTNEW_MAC_MIC_UID must name a non-default microphone")
		}
		if devices[index].SampleRate <= 0 {
			t.Fatalf("selected microphone has no hardware sample rate: %+v", devices[index])
		}
	} else if index < 0 {
		t.Fatal("no default microphone")
	}
	device := devices[index]
	tests := []struct {
		name    string
		devices []string
	}{
		{name: "default"},
		{name: "selected", devices: []string{device.ID}},
		{name: "selected_again", devices: []string{device.ID}},
		{name: "back_to_default"},
	}
	if requestedDeviceID != "" {
		tests = tests[1:3]
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := proto.CaptureParams{Devices: tc.devices, SampleRate: 16000, MaxSeconds: 0.6}
			c := record(t, h, tc.name, params, 0, 5*time.Second)
			if c.err != nil {
				t.Fatal(c.err)
			}
			if c.started == nil || c.started.Device != device.ID || c.started.FellBack {
				t.Fatalf("expected audio from %s without fallback, got %+v", device.ID, c.started)
			}
			if c.result.Reason != proto.CaptureMaxSeconds || len(c.frames) == 0 {
				t.Fatalf("capture ended without the expected audio: %+v", c.result)
			}
			c.checkFrames(t)
			if requestedDeviceID != "" {
				if c.started.DeviceSampleRate != device.SampleRate {
					t.Fatalf("capture used %d Hz, hardware reported %d Hz", c.started.DeviceSampleRate, device.SampleRate)
				}
				if c.result.Restarts != 0 {
					t.Fatalf("selected microphone restarted %d times", c.result.Restarts)
				}
			}
			t.Logf("%s at %d Hz started in %.1f ms, first PCM in %.1f ms, %d frames, %d restarts",
				device.ID, c.started.DeviceSampleRate, c.started.StartMS, c.result.FirstFrameMS, c.result.Frames, c.result.Restarts)
		})
	}
}
