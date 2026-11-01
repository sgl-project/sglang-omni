package outmute

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// The test binary is also the fake helper (helpertest).
func TestMain(m *testing.M) {
	if helpertest.Enabled() {
		helpertest.Main()
	}
	os.Exit(m.Run())
}

// app is a Controller wired as the app wires it (mute_link.go): the real
// supervisor over the fake helper process, its notifications and its
// status changes, with the output kept in a file that outlives the
// helper's processes.
type app struct {
	t       *testing.T
	sup     helper.Supervisor
	c       *Controller
	state   string // the output's file (the fake's out=)
	journal string
	mu      sync.Mutex
	ready   int // helpers that became ready
	notices []string
}

func startApp(t *testing.T, dir, opts string) *app {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	a := &app{t: t, state: filepath.Join(dir, "output.json"), journal: filepath.Join(dir, "output-mute")}
	bus := core.NewBus()
	a.sup = helper.New(helper.Config{
		Bin: self, AppVersion: "test", Bus: bus, Env: helpertest.Env(helpertest.Opt("out", a.state) + "," + opts),
		OnNotify: func(m proto.Message) {
			switch m.Method {
			case proto.MethodOutputChanged:
				var s proto.OutputState
				if m.DecodeParams(&s) == nil {
					a.c.Changed(s)
				}
			case proto.MethodOutputReleased:
				var p proto.OutputReleasedParams
				if m.DecodeParams(&p) == nil {
					a.c.Released(p)
				}
			}
		},
		Logf: t.Logf, HelloTimeout: 2 * time.Second, CancelGrace: 500 * time.Millisecond,
	})
	a.c = New(Config{
		Output: HelperOutput{H: a.sup}, Journal: a.journal, Log: func(l, m string) { t.Log(l + ": " + m) },
		OnUnmutable: func(id string) { a.mu.Lock(); a.notices = append(a.notices, id); a.mu.Unlock() },
	})
	was := helper.Stopped
	core.On(bus, func(ev helper.StatusChanged) {
		st := ev.Status
		a.mu.Lock()
		defer a.mu.Unlock()
		switch {
		case st.State == helper.Ready && was != helper.Ready:
			a.c.HelperReady(st.Has(proto.MethodOutputMute))
			a.ready++
		case st.State != helper.Ready && was == helper.Ready:
			a.c.HelperGone()
		}
		was = st.State
	})
	a.sup.Start(context.Background())
	a.waitReady(1)
	return a
}

func (a *app) waitReady(n int) {
	a.t.Helper()
	waitFor(a.t, "a ready helper", func() bool {
		a.mu.Lock()
		defer a.mu.Unlock()
		return a.ready >= n
	})
	a.c.Wait()
}

func (a *app) system() *helpertest.OutputSystem { return helpertest.LoadOutputSystem(a.state) }

func (a *app) muted(uid string) bool { return a.system().Muted(uid) }

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestProcessMuteAndRestore(t *testing.T) {
	a := startApp(t, t.TempDir(), "")
	defer a.sup.Stop()
	a.c.Engage("s1")
	a.c.Wait()
	if !a.muted(spk) {
		t.Fatal("not muted")
	}
	a.c.Release("s1", nil)
	a.c.Wait()
	if a.muted(spk) {
		t.Fatal("not unmuted")
	}
	if got := a.system().WriteLog(); !slices.Equal(got, []string{spk + ":true", spk + ":false"}) {
		t.Fatalf("writes %v", got)
	}
}

// TestProcessAppGoesAway: the app quits or crashes while it records; the
// helper's stdin closes and it unmutes as it exits.
func TestProcessAppGoesAway(t *testing.T) {
	dir := t.TempDir()
	a := startApp(t, dir, "")
	a.c.Engage("s1")
	a.c.Wait()
	if !a.muted(spk) {
		t.Fatal("not muted")
	}
	a.sup.Stop() // helper.shutdown, then stdin closes: as when the app dies
	waitFor(t, "the helper to unmute", func() bool { return !a.muted(spk) })
	files, _ := filepath.Glob(filepath.Join(a.journal, "*.json"))
	if len(files) != 0 {
		t.Errorf("journal left: %v", files)
	}
	// Started again: nothing left to do, and a mute the user made since
	// is theirs.
	sys := a.system()
	sys.UserMute(spk, true)
	b := startApp(t, dir, "")
	defer b.sup.Stop()
	if !b.muted(spk) {
		t.Error("recovery unmuted the user's mute")
	}
}

// TestProcessHelperCrashes: the helper dies right after it muted; the
// supervisor starts another, which unmutes what the first one claimed.
func TestProcessHelperCrashes(t *testing.T) {
	a := startApp(t, t.TempDir(), "crash-mute")
	defer a.sup.Stop()
	a.c.Engage("s1")
	a.waitReady(2) // restarted after the crash
	waitFor(t, "the next helper to unmute", func() bool { return !a.muted(spk) })
	files, _ := filepath.Glob(filepath.Join(a.journal, "*.json"))
	if len(files) != 0 {
		t.Errorf("journal left: %v", files)
	}
	a.c.Release("s1", nil)
	a.c.Wait()
	if got := a.system().WriteLog(); !slices.Equal(got, []string{spk + ":true", spk + ":false"}) {
		t.Fatalf("writes %v", got)
	}
}

// TestProcessBothDie: the app and the helper are killed together (no
// stdin to close, no shutdown); the next start unmutes.
func TestProcessBothDie(t *testing.T) {
	dir := t.TempDir()
	a := startApp(t, dir, "crash-mute")
	a.c.Engage("s1")
	waitFor(t, "the crash", func() bool { return a.sup.Status().State != helper.Ready })
	a.sup.Stop() // before the supervisor restarts it
	if !a.muted(spk) {
		t.Fatal("nothing left muted: the test proves nothing")
	}
	b := startApp(t, dir, "")
	defer b.sup.Stop()
	if b.muted(spk) {
		t.Fatal("the next start did not unmute")
	}
}

func TestProcessChanges(t *testing.T) {
	a := startApp(t, t.TempDir(), "out-switch=30")
	defer a.sup.Stop()
	a.c.Engage("s1")
	waitFor(t, "AirPods muted", func() bool { return a.muted(pods) })
	if a.muted(spk) {
		t.Error("the speakers stay muted after the switch")
	}
	a.c.Release("s1", nil)
	a.c.Wait()
	if a.muted(pods) {
		t.Error("AirPods stay muted")
	}

	b := startApp(t, t.TempDir(), "out-unmute=30")
	defer b.sup.Stop()
	b.c.Engage("s1")
	waitFor(t, "the user's unmute to be seen", func() bool { return b.c.Status().Yielded })
	b.system().UserMute(spk, true) // the user mutes it again
	b.c.Release("s1", nil)
	b.c.Wait()
	if !b.muted(spk) {
		t.Error("the user's mute was undone")
	}
}

func TestProcessUnmutable(t *testing.T) {
	for _, def := range []string{"fixed", "none"} {
		a := startApp(t, t.TempDir(), "out-default="+def)
		a.c.Engage("s1")
		a.c.Release("s1", nil)
		a.c.Wait()
		a.mu.Lock()
		n := slices.Clone(a.notices)
		a.mu.Unlock()
		if !slices.Equal(n, []string{"s1"}) {
			t.Errorf("%s: notices %v", def, n)
		}
		if w := a.system().WriteLog(); len(w) != 0 {
			t.Errorf("%s: writes %v", def, w)
		}
		a.sup.Stop()
	}
}

func TestProcessOlderHelper(t *testing.T) {
	a := startApp(t, t.TempDir(), "no-output")
	defer a.sup.Stop()
	a.c.Engage("s1")
	a.c.Release("s1", nil)
	if st := a.c.Status(); st.Supported {
		t.Fatalf("status %+v", st)
	}
	if w := a.system().WriteLog(); len(w) != 0 {
		t.Errorf("writes %v", w)
	}
}
