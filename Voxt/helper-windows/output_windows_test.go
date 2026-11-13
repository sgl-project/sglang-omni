//go:build windows

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

type fakeOutputDevice struct {
	device    proto.OutputDevice
	muted     bool
	readable  bool
	settable  bool
	refuseSet bool
}

type fakeOutputHardware struct {
	defaultUID string
	devices    map[string]*fakeOutputDevice
}

func (hardware *fakeOutputHardware) defaultOutput() (proto.OutputDevice, bool) {
	if hardware.defaultUID == "" {
		return proto.OutputDevice{}, false
	}
	return hardware.device(hardware.defaultUID)
}

func (hardware *fakeOutputHardware) device(uid string) (proto.OutputDevice, bool) {
	device, found := hardware.devices[uid]
	if !found {
		return proto.OutputDevice{}, false
	}
	return device.device, true
}

func (hardware *fakeOutputHardware) muted(device proto.OutputDevice) (bool, bool) {
	return hardware.devices[device.UID].muted, hardware.devices[device.UID].readable
}

func (hardware *fakeOutputHardware) settable(device proto.OutputDevice) bool {
	return hardware.devices[device.UID].settable
}

func (hardware *fakeOutputHardware) setMuted(device proto.OutputDevice, muted bool) bool {
	state := hardware.devices[device.UID]
	if state.refuseSet {
		return false
	}
	state.muted = muted
	return true
}

func fakeOutput(uid string, muted bool) *fakeOutputDevice {
	return &fakeOutputDevice{
		device: proto.OutputDevice{ID: 7, UID: uid, Name: uid}, muted: muted,
		readable: true, settable: true,
	}
}

func TestOutputClaimsNeverOwnUserMute(t *testing.T) {
	first := fakeOutput("first", false)
	userMuted := fakeOutput("user", true)
	hardware := &fakeOutputHardware{defaultUID: "first", devices: map[string]*fakeOutputDevice{
		"first": first, "user": userMuted,
	}}
	journal := t.TempDir()
	claims := newOutputClaims(hardware, 4242, func(int) bool { return false })
	result, outputErr := claims.mute(proto.OutputMuteParams{Device: first.device, Journal: journal})
	if outputErr != nil || !result.Took || !first.muted {
		t.Fatalf("mute = %+v, %v; device muted %v", result, outputErr, first.muted)
	}
	contents, err := os.ReadFile(filepath.Join(journal, "4242.json"))
	if err != nil {
		t.Fatal(err)
	}
	var saved outputJournal
	if json.Unmarshal(contents, &saved) != nil || saved.PID != 4242 || len(saved.Devices) != 1 || saved.Devices[0].UID != "first" {
		t.Fatalf("journal = %+v", saved)
	}
	result, outputErr = claims.mute(proto.OutputMuteParams{Device: userMuted.device})
	if outputErr != nil || result.Took {
		t.Fatalf("pre-muted claim = %+v, %v", result, outputErr)
	}
	if released := claims.release(proto.OutputReleaseParams{Device: userMuted.device, Unmute: true}); released.Unmuted || !userMuted.muted {
		t.Fatalf("release claimed a user's mute: %+v", released)
	}
	if released := claims.release(proto.OutputReleaseParams{Device: first.device, Unmute: true}); !released.Unmuted || first.muted {
		t.Fatalf("release own mute = %+v, device muted %v", released, first.muted)
	}
	if _, err := os.Stat(filepath.Join(journal, "4242.json")); !os.IsNotExist(err) {
		t.Fatalf("empty journal was not removed: %v", err)
	}
}

func TestOutputJournalFailureDoesNotMute(t *testing.T) {
	device := fakeOutput("device", false)
	hardware := &fakeOutputHardware{devices: map[string]*fakeOutputDevice{"device": device}}
	blocked := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocked, []byte("not a directory"), 0o600); err != nil {
		t.Fatal(err)
	}
	claims := newOutputClaims(hardware, 1, func(int) bool { return false })
	_, outputErr := claims.mute(proto.OutputMuteParams{Device: device.device, Journal: blocked})
	if outputErr == nil || outputErr.Code != proto.CodeInternal || device.muted {
		t.Fatalf("journal failure = %v, device muted %v", outputErr, device.muted)
	}
}

func TestOutputRecoverOnlyDeadHelpers(t *testing.T) {
	living := fakeOutput("living", true)
	dead := fakeOutput("dead", true)
	hardware := &fakeOutputHardware{devices: map[string]*fakeOutputDevice{"living": living, "dead": dead}}
	journal := t.TempDir()
	if err := writeOutputJournal(journal, 200, []proto.OutputDevice{living.device}); err != nil {
		t.Fatal(err)
	}
	if err := writeOutputJournal(journal, 300, []proto.OutputDevice{dead.device}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(journal, "999.json"), []byte("broken"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(journal, "notes.json"), []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	claims := newOutputClaims(hardware, 100, func(pid int) bool { return pid == 200 })
	result := claims.recover(proto.OutputRecoverParams{Journal: journal})
	if len(result.Restored) != 1 || result.Restored[0].UID != "dead" || dead.muted || !living.muted {
		t.Fatalf("recover = %+v; living %v dead %v", result.Restored, living.muted, dead.muted)
	}
	if _, err := os.Stat(filepath.Join(journal, "200.json")); err != nil {
		t.Fatalf("living helper's journal was removed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(journal, "300.json")); !os.IsNotExist(err) {
		t.Fatalf("recovered journal remains: %v", err)
	}
	if _, err := os.Stat(filepath.Join(journal, "999.json")); !os.IsNotExist(err) {
		t.Fatalf("unreadable numeric journal remains: %v", err)
	}
	if _, err := os.Stat(filepath.Join(journal, "notes.json")); err != nil {
		t.Fatalf("unrelated file was removed: %v", err)
	}
}

func TestOutputRecoverKeepsUnavailableDevice(t *testing.T) {
	hardware := &fakeOutputHardware{devices: map[string]*fakeOutputDevice{}}
	journal := t.TempDir()
	device := proto.OutputDevice{ID: 9, UID: "missing"}
	if err := writeOutputJournal(journal, 300, []proto.OutputDevice{device}); err != nil {
		t.Fatal(err)
	}
	claims := newOutputClaims(hardware, 100, func(int) bool { return false })
	if result := claims.recover(proto.OutputRecoverParams{Journal: journal}); len(result.Restored) != 0 {
		t.Fatalf("missing device restored: %+v", result.Restored)
	}
	contents, err := os.ReadFile(filepath.Join(journal, "300.json"))
	if err != nil || !json.Valid(contents) {
		t.Fatalf("unresolved journal was not retained: %v", err)
	}
}

func TestWindowsProcessRuns(t *testing.T) {
	if !windowsProcessRuns(os.Getpid()) {
		t.Fatal("the current process must count as alive")
	}
}
