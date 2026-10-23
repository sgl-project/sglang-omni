package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// TestMacHelperOutputMute mutes an output device through the real helper
// and Core Audio, and checks that every way out unmutes it: output.release,
// a helper killed with SIGKILL (the next helper's output.recover), and a
// helper whose app goes away (helper.shutdown, stdin closed). Opt-in, as
// it changes a device: VOXTNEW_MAC_OUTPUT=<the device's UID>, which must
// not be muted, nor the default output (so that nothing anyone hears
// changes) unless VOXTNEW_MAC_OUTPUT_DEFAULT=1 allows it (asked for
// first: other work on the Mac hears it); each mute lasts well under a
// second.
func TestMacHelperOutputMute(t *testing.T) {
	bin := macE2E(t)
	uid := os.Getenv("VOXTNEW_MAC_OUTPUT")
	if uid == "" {
		t.Skip("set VOXTNEW_MAC_OUTPUT to the UID of an output device that is not the default (e.g. BuiltInSpeakerDevice)")
	}
	journal := filepath.Join(t.TempDir(), "output-mute")
	dev := proto.OutputDevice{UID: uid}
	h0 := startMacHelper(t, bin)
	// Whatever fails below, a helper's claim left in the journal is undone
	// (before h0 stops; helpers that stop undo their own).
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		h0.Call(ctx, "", proto.MethodOutputRecover, proto.OutputRecoverParams{Journal: journal}, nil)
	})
	def := macCall[proto.OutputState](t, h0, proto.MethodOutputWatch, proto.OutputWatchParams{On: false})
	if def.Device == nil || def.Device.UID == uid && os.Getenv("VOXTNEW_MAC_OUTPUT_DEFAULT") != "1" {
		t.Fatalf("%s is the default output (or there is none: %+v): pick another device", uid, def.Device)
	}
	t.Logf("default output: device %d, muted %v (not touched)", def.Device.ID, def.IsMuted())

	// mute takes it, or says why not; it is unmuted again at once.
	mute := func(h *macHelper) (proto.OutputMuteResult, time.Time) {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		raw, err := h.Call(ctx, "", proto.MethodOutputMute, proto.OutputMuteParams{Device: dev, Journal: journal}, nil)
		var pe *proto.Error
		if errors.As(err, &pe) {
			t.Skipf("%s cannot be muted here: %s %s", uid, pe.Code, pe.Message)
		}
		if err != nil {
			t.Fatal(err)
		}
		var r proto.OutputMuteResult
		json.Unmarshal(raw, &r)
		return r, time.Now()
	}
	release := func(h *macHelper) proto.OutputReleaseResult {
		t.Helper()
		return macCall[proto.OutputReleaseResult](t, h, proto.MethodOutputRelease, proto.OutputReleaseParams{Device: dev, Unmute: true})
	}
	// unmutedNow checks the device is unmuted: a mute takes it again
	// (took), and is given back at once.
	unmutedNow := func(what string) {
		t.Helper()
		r, at := mute(h0)
		if !r.Took {
			t.Fatalf("%s: %s is still muted", what, uid)
		}
		if u := release(h0); !u.Unmuted {
			t.Fatalf("%s: the check's mute was not undone", what)
		}
		t.Logf("%s: unmuted (checked with a mute of %v)", what, time.Since(at).Round(time.Millisecond))
	}

	// 1. mute and release.
	r, at := mute(h0)
	if !r.Took {
		t.Skipf("%s is muted already (the user's): left alone", uid)
	}
	if !r.State.IsMuted() || r.State.Device == nil || r.State.Device.UID != uid {
		t.Fatalf("mute: %+v", r)
	}
	if files, _ := filepath.Glob(filepath.Join(journal, "*.json")); len(files) != 1 {
		t.Errorf("journal while muted: %v", files)
	}
	if u := release(h0); !u.Unmuted {
		t.Fatal("release did not unmute")
	}
	t.Logf("muted device %d for %v, then released", r.State.Device.ID, time.Since(at).Round(time.Millisecond))
	if files, _ := filepath.Glob(filepath.Join(journal, "*.json")); len(files) != 0 {
		t.Errorf("journal after the release: %v", files)
	}

	// 2. a helper killed while it muted: the next one's output.recover.
	killed := startMacHelper(t, bin)
	r, at = mute(killed)
	pid := killed.Status().PID
	syscall.Kill(pid, syscall.SIGKILL)
	killed.Stop() // no restart
	rec := macCall[proto.OutputRecoverResult](t, h0, proto.MethodOutputRecover, proto.OutputRecoverParams{Journal: journal})
	if len(rec.Restored) != 1 || rec.Restored[0].UID != uid {
		t.Fatalf("recover after SIGKILL: %+v", rec)
	}
	t.Logf("killed helper %d: recovered after %v", pid, time.Since(at).Round(time.Millisecond))
	unmutedNow("after the recovery")

	// 3. the app goes away: the helper unmutes as it exits.
	gone := startMacHelper(t, bin)
	_, at = mute(gone)
	gone.Stop()
	t.Logf("helper stopped %v after its mute", time.Since(at).Round(time.Millisecond))
	unmutedNow("after helper.shutdown")
	if files, _ := filepath.Glob(filepath.Join(journal, "*.json")); len(files) != 0 {
		t.Errorf("journal left: %v", files)
	}
}
