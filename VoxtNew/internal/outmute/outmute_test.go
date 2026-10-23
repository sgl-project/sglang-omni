package outmute

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// testOutput is a helper in process (helpertest.OutputHelper, the Swift
// helper's rules) with what can go wrong between the app and it: a call
// that fails, or one whose answer never comes (lost: done, then the
// context's error).
type testOutput struct {
	*helpertest.OutputHelper
	mu    sync.Mutex
	fail  map[string]error
	lost  map[string]bool
	calls []string
}

func (o *testOutput) note(m string) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.calls = append(o.calls, m)
	return o.fail[m]
}

func (o *testOutput) setFail(m string, err error) {
	o.mu.Lock()
	o.fail[m] = err
	o.mu.Unlock()
}

func (o *testOutput) setLost(m string) {
	o.mu.Lock()
	o.lost[m] = true
	o.mu.Unlock()
}

func (o *testOutput) isLost(m string) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.lost[m]
}

func (o *testOutput) Calls() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return slices.Clone(o.calls)
}

func (o *testOutput) Watch(ctx context.Context, on bool) (proto.OutputState, error) {
	if err := o.note(proto.MethodOutputWatch); err != nil {
		return proto.OutputState{}, err
	}
	return o.OutputHelper.Watch(ctx, on)
}

func (o *testOutput) Mute(ctx context.Context, p proto.OutputMuteParams) (proto.OutputMuteResult, error) {
	if err := o.note(proto.MethodOutputMute); err != nil {
		return proto.OutputMuteResult{}, err
	}
	r, err := o.OutputHelper.Mute(ctx, p)
	if o.isLost(proto.MethodOutputMute) {
		return proto.OutputMuteResult{}, context.DeadlineExceeded
	}
	return r, err
}

func (o *testOutput) Release(ctx context.Context, p proto.OutputReleaseParams) (proto.OutputReleaseResult, error) {
	if err := o.note(proto.MethodOutputRelease); err != nil {
		return proto.OutputReleaseResult{}, err
	}
	return o.OutputHelper.Release(ctx, p)
}

func (o *testOutput) Recover(ctx context.Context, p proto.OutputRecoverParams) (proto.OutputRecoverResult, error) {
	if err := o.note(proto.MethodOutputRecover); err != nil {
		return proto.OutputRecoverResult{}, err
	}
	return o.OutputHelper.Recover(ctx, p)
}

// rig is a Controller over a helper in process, on sys.
type rig struct {
	t       *testing.T
	sys     *helpertest.OutputSystem
	out     *testOutput
	c       *Controller
	journal string
	dead    map[int]bool // helpers that are gone, by pid
	mu      sync.Mutex
	notices []string
	logs    []string
}

func newRig(t *testing.T) *rig {
	r := &rig{t: t, sys: helpertest.NewOutputSystem(), journal: filepath.Join(t.TempDir(), "output-mute"), dead: map[int]bool{}}
	r.c = r.controller()
	r.helper(1001)
	return r
}

func (r *rig) controller() *Controller {
	return New(Config{
		Output:  outputFunc(func() Output { return r.out }),
		Journal: r.journal,
		Log: func(level, msg string) {
			r.mu.Lock()
			r.logs = append(r.logs, level+": "+msg)
			r.mu.Unlock()
		},
		OnUnmutable: func(id string) {
			r.mu.Lock()
			r.notices = append(r.notices, id)
			r.mu.Unlock()
		},
	})
}

// helper starts a helper process (pid) and tells the controller it is
// ready.
func (r *rig) helper(pid int) {
	h := &helpertest.OutputHelper{Sys: r.sys, PID: pid, Alive: func(p int) bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		return !r.dead[p]
	}}
	h.Notify = func(method string, params any) {
		if method == proto.MethodOutputChanged {
			r.c.Changed(params.(proto.OutputState))
		}
	}
	r.out = &testOutput{OutputHelper: h, fail: map[string]error{}, lost: map[string]bool{}}
	r.c.HelperReady(true)
	r.c.Wait()
}

// crash: the helper process dies without undoing anything.
func (r *rig) crash() {
	r.mu.Lock()
	r.dead[r.out.PID] = true
	r.mu.Unlock()
	r.out.Notify = nil
	r.c.HelperGone()
}

func (r *rig) Notices() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.notices)
}

// writes is what the helpers set, in order.
func (r *rig) writes(want ...string) {
	r.t.Helper()
	r.c.Wait()
	if got := r.sys.WriteLog(); !slices.Equal(got, want) {
		r.t.Fatalf("writes %v, want %v\nlogs:\n%v", got, want, r.logs)
	}
}

func (r *rig) muted(uid string, want bool) {
	r.t.Helper()
	r.c.Wait()
	if got := r.sys.Muted(uid); got != want {
		r.t.Fatalf("%s muted %t, want %t", uid, got, want)
	}
}

func (r *rig) journalFiles() []string {
	files, _ := filepath.Glob(filepath.Join(r.journal, "*.json"))
	return files
}

// outputFunc lets the rig swap the helper behind a controller.
type outputFunc func() Output

func (f outputFunc) Watch(ctx context.Context, on bool) (proto.OutputState, error) {
	return f().Watch(ctx, on)
}
func (f outputFunc) Mute(ctx context.Context, p proto.OutputMuteParams) (proto.OutputMuteResult, error) {
	return f().Mute(ctx, p)
}
func (f outputFunc) Release(ctx context.Context, p proto.OutputReleaseParams) (proto.OutputReleaseResult, error) {
	return f().Release(ctx, p)
}
func (f outputFunc) Recover(ctx context.Context, p proto.OutputRecoverParams) (proto.OutputRecoverResult, error) {
	return f().Recover(ctx, p)
}

const spk, pods = "BuiltInSpeakerDevice", "5C-52-30-AA-01-02:output"

func TestMuteAndRestore(t *testing.T) {
	r := newRig(t)
	r.c.Engage("s1")
	r.muted(spk, true)
	if st := r.c.Status(); st.Session != "s1" || st.Claim == nil || st.Claim.UID != spk || st.Uncertain {
		t.Fatalf("status %+v", st)
	}
	if len(r.journalFiles()) != 1 {
		t.Fatalf("the claim is not in the journal: %v", r.journalFiles())
	}
	r.c.Engage("s1") // twice: nothing more
	ran := false
	r.c.Release("s1", func() { ran = r.sys.Muted(spk) == false })
	r.writes(spk+":true", spk+":false")
	if !ran {
		t.Error("then ran before the output was given back")
	}
	if st := r.c.Status(); st.Session != "" || st.Claim != nil {
		t.Fatalf("status after %+v", st)
	}
	if len(r.journalFiles()) != 0 {
		t.Errorf("journal left: %v", r.journalFiles())
	}
	r.c.Release("s1", nil) // twice: nothing more
	r.writes(spk+":true", spk+":false")
	if n := r.Notices(); len(n) != 0 {
		t.Errorf("notices %v", n)
	}
	// The mute's own output.changed is read again (a watch on), and
	// changes nothing.
	want := []string{proto.MethodOutputRecover, proto.MethodOutputWatch, proto.MethodOutputMute, proto.MethodOutputWatch, proto.MethodOutputRelease, proto.MethodOutputWatch}
	if got := r.out.Calls(); !slices.Equal(got, want) {
		t.Errorf("calls %v, want %v", got, want)
	}
}

func TestUserMutedAlready(t *testing.T) {
	r := newRig(t)
	r.sys.UserMute(spk, true)
	r.c.Engage("s1")
	r.c.Release("s1", nil)
	r.writes()
	r.muted(spk, true)
	if n := r.Notices(); len(n) != 0 {
		t.Errorf("a muted device is no problem: %v", n)
	}
}

func TestCannotMute(t *testing.T) {
	for _, d := range []helpertest.OutputDevice{
		helpertest.Display, // a mute it cannot set
		{ID: 70, UID: "NoMute:1", Name: "USB DAC", NoMute: true},
	} {
		t.Run(d.UID, func(t *testing.T) {
			r := newRig(t)
			r.sys.SetDefault(d)
			r.c.Engage("s1")
			r.c.Release("s1", nil)
			r.writes()
			if n := r.Notices(); !slices.Equal(n, []string{"s1"}) {
				t.Fatalf("notices %v, want s1 once", n)
			}
		})
	}
	t.Run("no output", func(t *testing.T) {
		r := newRig(t)
		r.sys.Remove(spk)
		r.sys.Remove(pods)
		r.c.Engage("s1")
		r.c.Release("s1", nil)
		r.writes()
		if n := r.Notices(); len(n) != 0 {
			t.Errorf("notices %v", n)
		}
	})
	t.Run("journal unwritable", func(t *testing.T) {
		r := newRig(t)
		os.WriteFile(r.journal, []byte("a file"), 0o644) // not a directory
		r.c.Engage("s1")
		r.c.Release("s1", nil)
		r.writes()
		if n := r.Notices(); !slices.Equal(n, []string{"s1"}) {
			t.Fatalf("notices %v", n)
		}
	})
}

func TestOutputChanges(t *testing.T) {
	r := newRig(t)
	r.c.Engage("s1")
	r.muted(spk, true)
	r.sys.SetDefault(helpertest.AirPods) // the user picks AirPods meanwhile
	r.muted(pods, true)
	r.muted(spk, false)
	r.c.Release("s1", nil)
	r.writes(spk+":true", spk+":false", pods+":true", pods+":false")
	// Unplugged while muted: nothing to restore; the next default is taken.
	r.sys.SetDefault(helpertest.Speakers)
	r.c.Engage("s2")
	r.muted(spk, true)
	r.sys.Remove(spk)
	r.c.Wait()
	r.c.Release("s2", nil)
	r.writes(spk+":true", spk+":false", pods+":true", pods+":false", spk+":true", pods+":true", pods+":false")
	if n := r.Notices(); len(n) != 0 {
		t.Errorf("no notice on a change: %v", n)
	}
}

func TestUserUnmutes(t *testing.T) {
	r := newRig(t)
	r.c.Engage("s1")
	r.muted(spk, true)
	r.sys.UserMute(spk, false) // the user wants to hear it
	r.c.Wait()
	if st := r.c.Status(); st.Claim != nil || !st.Yielded {
		t.Fatalf("status %+v", st)
	}
	r.sys.UserMute(spk, true) // and mutes it again, themselves
	r.sys.SetDefault(helpertest.AirPods)
	r.c.Release("s1", nil)
	r.writes(spk + ":true")
	r.muted(spk, true)   // theirs: not unmuted at the end
	r.muted(pods, false) // and no other device taken this session
	// The next session takes the output again.
	r.c.Engage("s2")
	r.muted(pods, true)
	r.c.Release("s2", nil)
	r.muted(pods, false)
}

func TestLostAnswers(t *testing.T) {
	t.Run("muted, the answer lost", func(t *testing.T) {
		r := newRig(t)
		r.out.setLost(proto.MethodOutputMute)
		notify := r.out.Notify
		r.out.Notify = nil // its output.changed comes later
		r.c.Engage("s1")
		r.c.Wait()
		if st := r.c.Status(); !st.Uncertain || st.Claim == nil {
			t.Fatalf("status %+v", st)
		}
		notify(proto.MethodOutputChanged, r.sys.State()) // it did mute: the claim is sure
		r.c.Wait()
		if st := r.c.Status(); st.Uncertain || st.Claim == nil {
			t.Fatalf("status after the change %+v", st)
		}
		r.c.Release("s1", nil)
		r.writes(spk+":true", spk+":false")
	})
	t.Run("not muted, no answer", func(t *testing.T) {
		r := newRig(t)
		r.out.setFail(proto.MethodOutputMute, core.ErrHelper)
		r.c.Engage("s1")
		r.c.Release("s1", nil)
		r.writes() // the helper had no claim: the release did nothing
	})
	t.Run("the watch fails", func(t *testing.T) {
		r := newRig(t)
		r.out.setFail(proto.MethodOutputWatch, context.DeadlineExceeded)
		r.c.Engage("s1")
		r.c.Release("s1", nil)
		r.writes()
	})
	t.Run("the release fails", func(t *testing.T) {
		r := newRig(t)
		r.c.Engage("s1")
		r.c.Wait()
		r.out.setFail(proto.MethodOutputRelease, context.DeadlineExceeded)
		r.c.Release("s1", nil)
		r.writes(spk + ":true")
		// The helper still holds it: it unmutes it as it exits.
		r.out.ReleaseAll()
		r.writes(spk+":true", spk+":false")
	})
}

func TestLateCalls(t *testing.T) {
	r := newRig(t)
	ran := false
	r.c.Release("s1", func() { ran = true }) // stopped before its cue ended
	r.c.Engage("s1")                         // the cue's late engage
	r.writes()
	if !ran {
		t.Error("then did not run")
	}
	r.c.Engage("s2")
	r.c.Release("s2", nil)
	r.c.Changed(proto.OutputState{Device: &proto.OutputDevice{ID: 91, UID: pods}}) // after the end
	r.c.Released(proto.OutputReleasedParams{Reason: proto.OutputReasonSleep})
	r.writes(spk+":true", spk+":false")
	// A new session's engage releases one left engaged.
	r.c.Engage("s3")
	r.c.Engage("s4")
	r.c.Wait()
	if st := r.c.Status(); st.Session != "s4" || st.Claim == nil {
		t.Fatalf("status %+v", st)
	}
	r.c.Engage("s3") // released by s4: never again
	r.c.Release("s4", nil)
	r.writes(spk+":true", spk+":false", spk+":true", spk+":false", spk+":true", spk+":false")
}

func TestHelperRestarts(t *testing.T) {
	r := newRig(t)
	r.c.Engage("s1")
	r.muted(spk, true)
	r.crash() // killed while muting (a restart, a crash)
	r.c.Release("s1", nil)
	r.muted(spk, true) // nobody to unmute it yet
	r.helper(1002)     // the supervisor's next one
	r.muted(spk, false)
	r.writes(spk+":true", spk+":false")
	if len(r.journalFiles()) != 0 {
		t.Errorf("journal left: %v", r.journalFiles())
	}
	// The new helper works as the old one did.
	r.c.Engage("s2")
	r.muted(spk, true)
	r.c.Release("s2", nil)
	r.muted(spk, false)
}

func TestAppRestarts(t *testing.T) {
	t.Run("the helper died too", func(t *testing.T) {
		r := newRig(t)
		r.c.Engage("s1")
		r.muted(spk, true)
		r.crash() // the app and its helper killed
		r.c = r.controller()
		r.helper(1002) // started again
		r.muted(spk, false)
	})
	t.Run("the helper undid it", func(t *testing.T) {
		r := newRig(t)
		r.c.Engage("s1")
		r.muted(spk, true)
		r.out.ReleaseAll() // the app crashed: stdin closed, the helper unmutes as it exits
		r.crash()
		r.muted(spk, false)
		r.sys.UserMute(spk, true) // the user mutes it before starting VoxtNew again
		r.c = r.controller()
		r.helper(1002)
		r.muted(spk, true) // theirs: recovery leaves it alone
	})
	t.Run("another helper runs", func(t *testing.T) {
		r := newRig(t)
		r.c.Engage("s1")
		r.muted(spk, true)
		other := r.c
		r.c = r.controller()
		r.helper(1002) // a second app beside the first (dev builds): pid 1001 runs
		r.muted(spk, true)
		if len(r.journalFiles()) != 1 {
			t.Errorf("a running helper's journal was taken: %v", r.journalFiles())
		}
		_ = other
	})
}

func TestSleep(t *testing.T) {
	r := newRig(t)
	r.c.Engage("s1")
	r.muted(spk, true)
	gone := r.out.ReleaseAll() // willSleep: the helper unmutes by itself
	r.c.Released(proto.OutputReleasedParams{Devices: gone, Reason: proto.OutputReasonSleep})
	r.c.Wait()
	if st := r.c.Status(); st.Session != "" || st.Claim != nil {
		t.Fatalf("status %+v", st)
	}
	r.sys.SetDefault(helpertest.AirPods) // after the wake: not followed
	r.c.Engage("s1")
	r.c.Release("s1", nil)
	r.writes(spk+":true", spk+":false")
	r.muted(pods, false)
}

func TestOlderHelper(t *testing.T) {
	r := newRig(t)
	r.c.HelperReady(false)
	r.c.Engage("s1")
	r.c.Release("s1", nil)
	r.writes()
	calls := r.out.Calls()
	if slices.Contains(calls[1:], proto.MethodOutputWatch) {
		t.Errorf("calls to an older helper: %v", calls)
	}
}

func TestClose(t *testing.T) {
	r := newRig(t)
	r.c.Engage("s1")
	r.muted(spk, true)
	r.c.Close(time.Second)
	r.muted(spk, false)
	r.c.Engage("s2") // quitting: nothing more
	r.writes(spk+":true", spk+":false")
}

// TestManySessions: engages and releases from many goroutines (the main
// thread's and the cue's), output changes and user unmutes meanwhile; at
// the end nothing is muted and the helper holds no claim (-race).
func TestManySessions(t *testing.T) {
	r := newRig(t)
	var wg sync.WaitGroup
	for i := range 40 {
		id := "s" + string(rune('a'+i%26)) + string(rune('a'+i/26))
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.c.Engage(id)
			switch i % 4 {
			case 1:
				r.sys.SetDefault(helpertest.AirPods)
			case 2:
				r.sys.SetDefault(helpertest.Speakers)
			case 3:
				r.sys.UserMute(spk, false)
			}
			r.c.Release(id, nil)
		}()
	}
	wg.Wait()
	r.c.Wait()
	if st := r.c.Status(); st.Session != "" || st.Claim != nil {
		t.Fatalf("status %+v", st)
	}
	if c := r.out.Claims(); len(c) != 0 {
		t.Fatalf("claims left %v", c)
	}
	r.muted(spk, false)
	r.muted(pods, false)
}

func TestErrorsClassified(t *testing.T) {
	r := newRig(t)
	r.out.setFail(proto.MethodOutputMute, &proto.Error{Code: proto.CodeDeviceUnavailable, Message: "gone"})
	r.c.Engage("s1")
	r.c.Release("s1", nil)
	r.c.Wait()
	r.out.setFail(proto.MethodOutputMute, &proto.Error{Code: proto.CodeUnsupported, Message: "no"})
	r.c.Engage("s2")
	r.c.Release("s2", nil)
	r.writes()
	if n := r.Notices(); !slices.Equal(n, []string{"s2"}) {
		t.Fatalf("notices %v", n)
	}
	var pe *proto.Error
	if !errors.As(error(&proto.Error{Code: proto.CodeUnsupported}), &pe) {
		t.Fatal("proto.Error")
	}
}

// TestNoJournal: without a journal nothing is recovered, and the helper
// never reads the working directory as one (it once took mygo.json for
// a journal and removed it).
func TestNoJournal(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "mygo.json"), []byte(`{"name":"x"}`), 0o644)
	os.WriteFile(filepath.Join(dir, "12.json"), []byte(`{"pid":12,"devices":[]}`), 0o644)
	t.Chdir(dir)
	h := &helpertest.OutputHelper{Sys: helpertest.NewOutputSystem(), PID: 1, Alive: func(int) bool { return false }}
	if _, err := h.Recover(context.Background(), proto.OutputRecoverParams{}); err != nil {
		t.Fatal(err)
	}
	h.Recover(context.Background(), proto.OutputRecoverParams{Journal: dir})
	if _, err := os.Stat(filepath.Join(dir, "mygo.json")); err != nil {
		t.Fatal("a file that is not a journal was removed")
	}
	if _, err := os.Stat(filepath.Join(dir, "12.json")); err == nil {
		t.Error("a dead helper's journal was kept")
	}
	o := &testOutput{OutputHelper: h, fail: map[string]error{}, lost: map[string]bool{}}
	c := New(Config{Output: o})
	c.HelperReady(true)
	c.Wait()
	if slices.Contains(o.Calls(), proto.MethodOutputRecover) {
		t.Error("recovered without a journal")
	}
}
