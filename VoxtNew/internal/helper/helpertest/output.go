package helpertest

// The output.* methods of the fake: OutputSystem is the Mac's Core Audio
// output (its devices, the default one, their mutes), OutputHelper a
// helper's claims on it, with the rules of the Swift helper's MuteClaims
// (helper/Sources/VoxtNewHelperKit/OutputMute.swift): a mute the user
// made is never claimed, only claimed mutes are undone, each claim is in
// the journal (<dir>/<pid>.json) before the mute, and output.recover undoes
// the journals of helpers that are gone. Tests use both in process
// (internal/outmute), and the fake helper process serves them, with the
// system kept in a file (out=PATH) so that it outlives a crash.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// OutputDevice is a device of an OutputSystem.
type OutputDevice struct {
	ID   uint32 `json:"id"`
	UID  string `json:"uid"`
	Name string `json:"name"`
	// NoMute: it has no mute; Fixed: a mute that cannot be set.
	NoMute bool `json:"noMute,omitempty"`
	Fixed  bool `json:"fixed,omitempty"`
	Muted  bool `json:"muted,omitempty"`
}

// The fake's devices: the speakers (the default) and AirPods.
var (
	Speakers = OutputDevice{ID: 52, UID: "BuiltInSpeakerDevice", Name: "MacBook Air Speakers"}
	AirPods  = OutputDevice{ID: 91, UID: "5C-52-30-AA-01-02:output", Name: "AirPods Pro"}
	Display  = OutputDevice{ID: 61, UID: "DisplayPortAudio:1", Name: "LG UltraFine", Fixed: true}
)

// OutputSystem is the output as Core Audio has it. Goroutine-safe.
type OutputSystem struct {
	mu      sync.Mutex
	Devices []OutputDevice `json:"devices"`
	Default string         `json:"default"` // a UID; "" for none
	// Writes are the mutes set, in order: "uid:true", "uid:false".
	Writes []string `json:"writes,omitempty"`
	path   string
	subs   []func()
}

// NewOutputSystem is the speakers (the default, unmuted) and AirPods.
func NewOutputSystem() *OutputSystem {
	return &OutputSystem{Devices: []OutputDevice{Speakers, AirPods}, Default: Speakers.UID}
}

// LoadOutputSystem reads the system kept in path, or makes the default one
// there: the fake helper's, outliving the process.
func LoadOutputSystem(path string) *OutputSystem {
	s := NewOutputSystem()
	data, err := os.ReadFile(path)
	if err == nil {
		json.Unmarshal(data, s)
	}
	s.path = path
	if err != nil {
		s.saveLocked() // reading writes nothing: another process may be writing
	}
	return s
}

func (s *OutputSystem) saveLocked() {
	if s.path == "" {
		return
	}
	data, _ := json.Marshal(s)
	tmp := s.path + ".tmp"
	if os.WriteFile(tmp, data, 0o644) == nil {
		os.Rename(tmp, s.path)
	}
}

// reload reads the file again (another process may have written it).
func (s *OutputSystem) reloadLocked() {
	if s.path == "" {
		return
	}
	if data, err := os.ReadFile(s.path); err == nil {
		var v OutputSystem
		if json.Unmarshal(data, &v) == nil {
			s.Devices, s.Default, s.Writes = v.Devices, v.Default, v.Writes
		}
	}
}

// subscribe calls fn after each change (the helper's listeners).
func (s *OutputSystem) subscribe(fn func()) {
	s.mu.Lock()
	s.subs = append(s.subs, fn)
	s.mu.Unlock()
}

func (s *OutputSystem) changed() {
	s.mu.Lock()
	subs := slices.Clone(s.subs)
	s.mu.Unlock()
	for _, fn := range subs {
		fn()
	}
}

// Device is the device of uid now (false: none).
func (s *OutputSystem) Device(uid string) (OutputDevice, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reloadLocked()
	return s.deviceLocked(uid)
}

func (s *OutputSystem) deviceLocked(uid string) (OutputDevice, bool) {
	for _, d := range s.Devices {
		if d.UID == uid {
			return d, true
		}
	}
	return OutputDevice{}, false
}

// Muted reports whether uid's device is muted.
func (s *OutputSystem) Muted(uid string) bool {
	d, _ := s.Device(uid)
	return d.Muted
}

// SetDefault makes uid the default output (the user picked it, a device
// came or went), adding d when it is not there.
func (s *OutputSystem) SetDefault(d OutputDevice) {
	s.mu.Lock()
	s.reloadLocked()
	if _, ok := s.deviceLocked(d.UID); !ok {
		s.Devices = append(s.Devices, d)
	}
	s.Default = d.UID
	s.saveLocked()
	s.mu.Unlock()
	s.changed()
}

// Remove unplugs uid's device (the default goes to the first left).
func (s *OutputSystem) Remove(uid string) {
	s.mu.Lock()
	s.reloadLocked()
	s.Devices = slices.DeleteFunc(s.Devices, func(d OutputDevice) bool { return d.UID == uid })
	if s.Default == uid {
		s.Default = ""
		if len(s.Devices) > 0 {
			s.Default = s.Devices[0].UID
		}
	}
	s.saveLocked()
	s.mu.Unlock()
	s.changed()
}

// UserMute is the user muting or unmuting uid's device (the menu bar,
// the keyboard): it is not a write of a helper's.
func (s *OutputSystem) UserMute(uid string, muted bool) {
	s.set(uid, muted, false)
}

// set changes uid's mute; write records it in Writes (a helper's).
func (s *OutputSystem) set(uid string, muted, write bool) bool {
	s.mu.Lock()
	s.reloadLocked()
	ok := false
	for i, d := range s.Devices {
		if d.UID == uid && !d.NoMute && !d.Fixed {
			s.Devices[i].Muted, ok = muted, true
			if write {
				s.Writes = append(s.Writes, uid+":"+strconv.FormatBool(muted))
			}
		}
	}
	s.saveLocked()
	s.mu.Unlock()
	if ok {
		s.changed()
	}
	return ok
}

// WriteLog is Writes as the file has it now.
func (s *OutputSystem) WriteLog() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reloadLocked()
	return slices.Clone(s.Writes)
}

// State is the default output and its mute.
func (s *OutputSystem) State() proto.OutputState {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reloadLocked()
	d, ok := s.deviceLocked(s.Default)
	if !ok {
		return proto.OutputState{}
	}
	return deviceState(d)
}

func deviceState(d OutputDevice) proto.OutputState {
	st := proto.OutputState{Device: &proto.OutputDevice{ID: d.ID, UID: d.UID, Name: d.Name}}
	if !d.NoMute {
		m := d.Muted
		st.Muted, st.Settable = &m, !d.Fixed
	}
	return st
}

// OutputHelper is one helper's side: its claims and its watch.
type OutputHelper struct {
	Sys *OutputSystem
	// PID names its journal file; Alive tells whether another helper's
	// process runs (nil: a signal 0 to it).
	PID   int
	Alive func(pid int) bool
	// Notify sends a notification (output.changed).
	Notify func(method string, params any)

	mu       sync.Mutex
	claims   []proto.OutputDevice
	journal  string
	watching bool
	last     proto.OutputState
	once     sync.Once
}

func (h *OutputHelper) alive(pid int) bool {
	if pid == h.PID {
		return true
	}
	if h.Alive != nil {
		return h.Alive(pid)
	}
	p, err := os.FindProcess(pid)
	return err == nil && p.Signal(syscall.Signal(0)) == nil
}

// Claims are the devices it muted.
func (h *OutputHelper) Claims() []proto.OutputDevice {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.claims)
}

func (h *OutputHelper) Watch(_ context.Context, on bool) (proto.OutputState, error) {
	h.once.Do(func() { h.Sys.subscribe(h.sysChanged) })
	st := h.Sys.State()
	h.mu.Lock()
	h.watching, h.last = on, st
	h.mu.Unlock()
	return st, nil
}

func (h *OutputHelper) sysChanged() {
	st := h.Sys.State()
	h.mu.Lock()
	send := h.watching && !sameState(st, h.last)
	if send {
		h.last = st
	}
	h.mu.Unlock()
	if send && h.Notify != nil {
		h.Notify(proto.MethodOutputChanged, st)
	}
}

func sameState(a, b proto.OutputState) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}

func (h *OutputHelper) Mute(_ context.Context, p proto.OutputMuteParams) (proto.OutputMuteResult, error) {
	d, ok := h.Sys.Device(p.Device.UID)
	if !ok {
		return proto.OutputMuteResult{}, &proto.Error{Code: proto.CodeDeviceUnavailable, Message: "No output device has the UID " + p.Device.UID}
	}
	st := deviceState(d)
	switch {
	case st.Muted == nil || !st.Settable:
		return proto.OutputMuteResult{}, &proto.Error{Code: proto.CodeUnsupported, Message: d.Name + " has no mute the helper can set"}
	case d.Muted:
		return proto.OutputMuteResult{Took: false, State: st}, nil
	}
	dev := *st.Device
	h.mu.Lock()
	claims := append(slices.Clone(h.claims), dev)
	if p.Journal != "" {
		h.journal = p.Journal
	}
	if err := h.writeJournal(claims); err != nil {
		h.mu.Unlock()
		return proto.OutputMuteResult{}, &proto.Error{Code: proto.CodeInternal, Message: "Could not write the journal: " + err.Error()}
	}
	h.claims = claims
	h.mu.Unlock()
	if !h.Sys.set(dev.UID, true, true) {
		h.mu.Lock()
		h.claims = slices.DeleteFunc(h.claims, func(c proto.OutputDevice) bool { return c.UID == dev.UID })
		h.writeJournal(h.claims)
		h.mu.Unlock()
		return proto.OutputMuteResult{}, &proto.Error{Code: proto.CodeUnsupported, Message: d.Name + " did not take the mute"}
	}
	d, _ = h.Sys.Device(dev.UID)
	return proto.OutputMuteResult{Took: true, State: deviceState(d)}, nil
}

func (h *OutputHelper) Release(_ context.Context, p proto.OutputReleaseParams) (proto.OutputReleaseResult, error) {
	h.mu.Lock()
	i := slices.IndexFunc(h.claims, func(c proto.OutputDevice) bool { return c.UID == p.Device.UID })
	if i < 0 {
		h.mu.Unlock()
		return proto.OutputReleaseResult{}, nil // not ours
	}
	h.claims = slices.Delete(slices.Clone(h.claims), i, i+1)
	h.writeJournal(h.claims)
	h.mu.Unlock()
	if p.Unmute && h.Sys.Muted(p.Device.UID) {
		return proto.OutputReleaseResult{Unmuted: h.Sys.set(p.Device.UID, false, true)}, nil
	}
	return proto.OutputReleaseResult{}, nil
}

// ReleaseAll undoes every claim (stdin closed, helper.shutdown, sleep)
// and says which it gave back.
func (h *OutputHelper) ReleaseAll() []proto.OutputDevice {
	h.mu.Lock()
	claims := h.claims
	h.claims = nil
	h.writeJournal(nil)
	h.mu.Unlock()
	for _, c := range claims {
		if h.Sys.Muted(c.UID) {
			h.Sys.set(c.UID, false, true)
		}
	}
	return claims
}

func (h *OutputHelper) Recover(_ context.Context, p proto.OutputRecoverParams) (proto.OutputRecoverResult, error) {
	var res proto.OutputRecoverResult
	if p.Journal == "" {
		return res, nil // no journal: never the working directory
	}
	files, _ := filepath.Glob(filepath.Join(p.Journal, "*.json"))
	for _, f := range files {
		if _, err := strconv.Atoi(strings.TrimSuffix(filepath.Base(f), ".json")); err != nil {
			continue // not a helper's journal (<pid>.json)
		}
		var j journalFile
		data, err := os.ReadFile(f)
		if err != nil || json.Unmarshal(data, &j) != nil {
			os.Remove(f) // unreadable: nothing to undo
			continue
		}
		if h.alive(j.PID) {
			continue // a helper that runs: its claims are its own
		}
		for _, d := range j.Devices {
			if h.Sys.Muted(d.UID) && h.Sys.set(d.UID, false, true) {
				res.Restored = append(res.Restored, d)
			}
		}
		os.Remove(f)
	}
	return res, nil
}

// journalFile is a helper's claims on disk (the Swift helper's format).
type journalFile struct {
	PID     int                  `json:"pid"`
	Devices []proto.OutputDevice `json:"devices"`
}

// writeJournal keeps claims in the journal (none: no file). h.mu held.
func (h *OutputHelper) writeJournal(claims []proto.OutputDevice) error {
	if h.journal == "" {
		return nil
	}
	path := filepath.Join(h.journal, strconv.Itoa(h.PID)+".json")
	if len(claims) == 0 {
		err := os.Remove(path)
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if err := os.MkdirAll(h.journal, 0o755); err != nil {
		return err
	}
	data, _ := json.Marshal(journalFile{PID: h.PID, Devices: claims})
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ---- the fake helper process ----

// output serves the output.* methods (out=PATH keeps the system in a file;
// out-default=muted|fixed|none|empty starts the default output so).
func (f *fake) output(m proto.Message) {
	h := f.outputHelper()
	ctx := context.Background()
	var res any
	var err error
	switch m.Method {
	case proto.MethodOutputWatch:
		var p proto.OutputWatchParams
		m.DecodeParams(&p)
		res, err = h.Watch(ctx, p.On)
	case proto.MethodOutputMute:
		var p proto.OutputMuteParams
		m.DecodeParams(&p)
		var r proto.OutputMuteResult
		r, err = h.Mute(ctx, p)
		res = r
		if err == nil && r.Took {
			f.afterMute(h, p.Device)
		}
	case proto.MethodOutputRelease:
		var p proto.OutputReleaseParams
		m.DecodeParams(&p)
		res, err = h.Release(ctx, p)
	case proto.MethodOutputRecover:
		var p proto.OutputRecoverParams
		m.DecodeParams(&p)
		res, err = h.Recover(ctx, p)
	}
	if pe, ok := err.(*proto.Error); ok {
		f.send(proto.NewError(m.ID, pe))
		return
	}
	f.result(m.ID, res)
}

func (f *fake) outputHelper() *OutputHelper {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.outSim == nil {
		var sys *OutputSystem
		if path := f.opts["out"]; path != "" {
			sys = LoadOutputSystem(path)
		} else {
			sys = NewOutputSystem()
		}
		switch f.opts["out-default"] {
		case "muted":
			sys.UserMute(sys.Default, true)
		case "fixed":
			sys.SetDefault(Display)
		case "none":
			sys.SetDefault(OutputDevice{ID: 70, UID: "NoMute:1", Name: "USB DAC", NoMute: true})
		case "empty":
			sys.mu.Lock()
			sys.Default = ""
			sys.saveLocked()
			sys.mu.Unlock()
		}
		f.outSim = &OutputHelper{Sys: sys, PID: os.Getpid(), Notify: f.notify}
	}
	return f.outSim
}

// afterMute plays what the options script after a mute: crash-mute
// exits at once (the claim stays in the journal), out-switch=MS makes
// AirPods the default MS ms later, out-unmute=MS has the user unmute the
// device MS ms later.
func (f *fake) afterMute(h *OutputHelper, d proto.OutputDevice) {
	if f.has("crash-mute") {
		fmt.Fprintln(os.Stderr, "fatal: crashed after muting")
		os.Exit(3)
	}
	if ms, err := strconv.Atoi(f.opts["out-switch"]); err == nil && d.UID != AirPods.UID {
		time.AfterFunc(time.Duration(ms)*time.Millisecond, func() { h.Sys.SetDefault(AirPods) })
	}
	if ms, err := strconv.Atoi(f.opts["out-unmute"]); err == nil {
		time.AfterFunc(time.Duration(ms)*time.Millisecond, func() { h.Sys.UserMute(d.UID, false) })
	}
}

// outputExit undoes the claims as the fake exits (stdin closed,
// helper.shutdown), as the helper does.
func (f *fake) outputExit() {
	f.mu.Lock()
	h := f.outSim
	f.mu.Unlock()
	if h != nil {
		if gone := h.ReleaseAll(); len(gone) > 0 {
			fmt.Fprintf(os.Stderr, "unmuted %s as the app went away\n", strings.Join(uids(gone), ", "))
		}
	}
}

func uids(ds []proto.OutputDevice) []string {
	var out []string
	for _, d := range ds {
		out = append(out, d.UID)
	}
	return out
}
