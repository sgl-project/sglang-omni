//go:build windows

package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

type outputHardware interface {
	defaultOutput() (proto.OutputDevice, bool)
	device(uid string) (proto.OutputDevice, bool)
	muted(device proto.OutputDevice) (bool, bool)
	settable(device proto.OutputDevice) bool
	setMuted(device proto.OutputDevice, muted bool) bool
}

type outputClaims struct {
	hardware outputHardware
	pid      int
	alive    func(pid int) bool

	mutex   sync.Mutex
	claims  []proto.OutputDevice
	journal string
	wrote   bool
}

type outputJournal struct {
	PID     int                  `json:"pid"`
	Devices []proto.OutputDevice `json:"devices"`
}

func newOutputClaims(hardware outputHardware, pid int, alive func(pid int) bool) *outputClaims {
	return &outputClaims{hardware: hardware, pid: pid, alive: alive}
}

func (claims *outputClaims) state() proto.OutputState {
	claims.mutex.Lock()
	defer claims.mutex.Unlock()
	return claims.stateLocked()
}

func (claims *outputClaims) stateLocked() proto.OutputState {
	device, found := claims.hardware.defaultOutput()
	if !found {
		return proto.OutputState{}
	}
	return claims.stateOf(device)
}

func (claims *outputClaims) stateOf(device proto.OutputDevice) proto.OutputState {
	muted, readable := claims.hardware.muted(device)
	if !readable {
		return proto.OutputState{Device: &device}
	}
	return proto.OutputState{Device: &device, Muted: &muted, Settable: claims.hardware.settable(device)}
}

func (claims *outputClaims) mute(params proto.OutputMuteParams) (proto.OutputMuteResult, *proto.Error) {
	claims.mutex.Lock()
	defer claims.mutex.Unlock()
	device, found := claims.hardware.device(params.Device.UID)
	if !found {
		return proto.OutputMuteResult{}, deviceUnavailable(params.Device.UID, "No output device has the UID "+params.Device.UID)
	}
	state := claims.stateOf(device)
	if state.Muted == nil || !state.Settable {
		return proto.OutputMuteResult{}, unsupportedOutput("The device has no mute the helper can set")
	}
	if *state.Muted {
		return proto.OutputMuteResult{Took: false, State: state}, nil
	}
	if params.Journal != "" {
		claims.journal = params.Journal
	}
	next := appendClaim(nil, claims.claims, device)
	if err := claims.writeJournal(next); err != nil {
		return proto.OutputMuteResult{}, proto.Errorf(proto.CodeInternal, "Could not write the journal: %v", err)
	}
	if !claims.hardware.setMuted(device, true) {
		_ = claims.writeJournal(claims.claims)
		return proto.OutputMuteResult{}, unsupportedOutput("The device did not take the mute")
	}
	claims.claims = next
	return proto.OutputMuteResult{Took: true, State: claims.stateOf(device)}, nil
}

func (claims *outputClaims) release(params proto.OutputReleaseParams) proto.OutputReleaseResult {
	claims.mutex.Lock()
	defer claims.mutex.Unlock()
	index := claimIndex(claims.claims, params.Device.UID)
	if index < 0 {
		return proto.OutputReleaseResult{}
	}
	result := proto.OutputReleaseResult{}
	if params.Unmute {
		switch claims.undo(params.Device.UID) {
		case undoFailed:
			return result
		case undoUnmuted:
			result.Unmuted = true
		case undoNotMuted:
		}
	}
	claims.claims = append(claims.claims[:index], claims.claims[index+1:]...)
	_ = claims.writeJournal(claims.claims)
	return result
}

func (claims *outputClaims) releaseAll() []proto.OutputDevice {
	claims.mutex.Lock()
	defer claims.mutex.Unlock()
	released := make([]proto.OutputDevice, 0, len(claims.claims))
	kept := make([]proto.OutputDevice, 0)
	for _, device := range claims.claims {
		if claims.undo(device.UID) == undoFailed {
			kept = append(kept, device)
		} else {
			released = append(released, device)
		}
	}
	claims.claims = kept
	_ = claims.writeJournal(kept)
	return released
}

func (claims *outputClaims) recover(params proto.OutputRecoverParams) proto.OutputRecoverResult {
	claims.mutex.Lock()
	defer claims.mutex.Unlock()
	result := proto.OutputRecoverResult{}
	if params.Journal == "" {
		return result
	}
	entries, err := os.ReadDir(params.Journal)
	if err != nil {
		return result
	}
	sort.Slice(entries, func(first, second int) bool { return entries[first].Name() < entries[second].Name() })
	for _, entry := range entries {
		_, valid := journalPID(entry.Name())
		if entry.IsDir() || !valid {
			continue
		}
		path := filepath.Join(params.Journal, entry.Name())
		contents, err := os.ReadFile(path)
		var journal outputJournal
		if err != nil || json.Unmarshal(contents, &journal) != nil {
			_ = os.Remove(path)
			continue
		}
		if journal.PID == claims.pid && claims.wrote || journal.PID != claims.pid && claims.alive(journal.PID) {
			continue
		}
		left := make([]proto.OutputDevice, 0)
		for _, device := range journal.Devices {
			switch claims.undo(device.UID) {
			case undoUnmuted:
				result.Restored = append(result.Restored, device)
			case undoNotMuted:
			case undoFailed:
				left = append(left, device)
			}
		}
		if len(left) == 0 {
			_ = os.Remove(path)
		} else {
			_ = writeOutputJournal(params.Journal, journal.PID, left)
		}
	}
	return result
}

type undoResult int

const (
	undoFailed undoResult = iota
	undoUnmuted
	undoNotMuted
)

func (claims *outputClaims) undo(uid string) undoResult {
	device, found := claims.hardware.device(uid)
	if !found {
		return undoFailed
	}
	muted, readable := claims.hardware.muted(device)
	if !readable {
		return undoFailed
	}
	if !muted {
		return undoNotMuted
	}
	if !claims.hardware.setMuted(device, false) {
		return undoFailed
	}
	return undoUnmuted
}

func (claims *outputClaims) writeJournal(devices []proto.OutputDevice) error {
	if claims.journal == "" {
		return nil
	}
	if len(devices) == 0 {
		err := os.Remove(filepath.Join(claims.journal, strconv.Itoa(claims.pid)+".json"))
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if err := writeOutputJournal(claims.journal, claims.pid, devices); err != nil {
		return err
	}
	claims.wrote = true
	return nil
}

func writeOutputJournal(directory string, pid int, devices []proto.OutputDevice) (returnErr error) {
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, strconv.Itoa(pid)+"-*.json")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() {
		if temporary != nil {
			temporary.Close()
		}
		if returnErr != nil {
			_ = os.Remove(temporaryPath)
		}
	}()
	encoder := json.NewEncoder(temporary)
	if err := encoder.Encode(outputJournal{PID: pid, Devices: devices}); err != nil {
		return err
	}
	if err := temporary.Sync(); err != nil {
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	temporary = nil
	path := filepath.Join(directory, strconv.Itoa(pid)+".json")
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	return nil
}

func appendClaim(claims []proto.OutputDevice, existing []proto.OutputDevice, device proto.OutputDevice) []proto.OutputDevice {
	for _, claimed := range existing {
		if claimed.UID != device.UID {
			claims = append(claims, claimed)
		}
	}
	return append(claims, device)
}

func claimIndex(claims []proto.OutputDevice, uid string) int {
	for index, device := range claims {
		if device.UID == uid {
			return index
		}
	}
	return -1
}

func journalPID(name string) (int, bool) {
	if filepath.Ext(name) != ".json" {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSuffix(name, ".json"))
	return pid, err == nil
}

func unsupportedOutput(message string) *proto.Error {
	data, _ := json.Marshal(map[string]string{"capability": "mute"})
	return &proto.Error{Code: proto.CodeUnsupported, Message: message, Data: data}
}
