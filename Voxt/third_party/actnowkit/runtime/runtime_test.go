package runtime

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/family"
)

func TestRuntimeFiles(t *testing.T) {
	d := family.At(t.TempDir())
	now := time.Now().Add(-2 * time.Minute)
	if err := Write(d, File{BundleID: "dev.hehehai.dicta", BudgetBytes: 10, Loaded: []Loaded{{Repo: "a/b", Revision: "r", Bytes: 300, IdleSince: &now}}}); err != nil {
		t.Fatal(err)
	}
	if err := Write(d, File{BundleID: "dev.hehehai.audiosmith", Loaded: []Loaded{{Repo: "a/c", Revision: "r", Bytes: 200}}}); err != nil {
		t.Fatal(err)
	}
	// A crashed app's file: its pid is gone.
	os.WriteFile(d.RuntimeFile("dev.hehehai.auralink"), []byte(`{"schema":"actnow.runtime/1","bundleId":"dev.hehehai.auralink","pid":999999999,"loaded":[{"repo":"x/y","bytes":5000}]}`), 0o600)
	all, errs := ReadAll(d)
	if len(all) != 2 || len(errs) != 0 {
		t.Fatalf("%d files, %v", len(all), errs)
	}
	if n := OthersLoadedBytes(d, "dev.hehehai.dicta"); n != 200 {
		t.Errorf("others %d", n)
	}
	if got := LoadedBy(d, "dev.hehehai.audiosmith", "a/b", "r"); len(got) != 1 || got[0].BundleID != "dev.hehehai.dicta" {
		t.Errorf("loaded by %+v", got)
	}
	f, _ := Read(d, "dev.hehehai.dicta")
	if !Idle(f.Loaded[0], time.Now()) || f.PID != os.Getpid() || f.Schema != "actnow.runtime/1" {
		t.Errorf("%+v", f)
	}
	if err := Remove(d, "dev.hehehai.dicta"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(d.RuntimeDir(), "dev.hehehai.dicta.json")); !os.IsNotExist(err) {
		t.Error("not removed")
	}
}

func TestBudget(t *testing.T) {
	gb := int64(1 << 30)
	cases := []struct{ own, cap, others, want int64 }{
		{0, 14 * gb, 0, 14 * gb},
		{8 * gb, 14 * gb, 0, 8 * gb},
		{8 * gb, 14 * gb, 10 * gb, 4 * gb},
		{8 * gb, 14 * gb, 20 * gb, 0},
	}
	for _, c := range cases {
		if got := Budget(c.own, c.cap, c.others); got != c.want {
			t.Errorf("%+v: %d", c, got)
		}
	}
	if FamilyCap(24*uint64(gb)) != int64(float64(24*gb)*0.6) {
		t.Error("cap")
	}
	if PhysicalMemory() == 0 {
		t.Error("physical memory unknown")
	}
}

func TestWaitPasting(t *testing.T) {
	d := family.At(t.TempDir())
	if !WaitPasting(context.Background(), d, "me") {
		t.Error("nobody pastes")
	}
	Write(d, File{BundleID: "other", Pasting: true})
	start := time.Now()
	if WaitPasting(context.Background(), d, "me") {
		t.Error("clear while pasting")
	}
	if el := time.Since(start); el < PasteWait || el > 3*PasteWait {
		t.Errorf("waited %v", el)
	}
	go func() { time.Sleep(60 * time.Millisecond); Write(d, File{BundleID: "other"}) }()
	if !WaitPasting(context.Background(), d, "me") {
		t.Error("paste ended")
	}
}

// C10: a reused pid or a file from before the boot is not a live app.
func TestLiveStale(t *testing.T) {
	self, _ := os.Executable()
	now := time.Now()
	if !Live(File{PID: os.Getpid(), Exe: self, Updated: now}) {
		t.Fatal("self not live")
	}
	if !Live(File{PID: os.Getpid(), Updated: now}) {
		t.Fatal("older writer (no exe) not live")
	}
	if Live(File{PID: os.Getpid(), Exe: "/Applications/Other.app/Contents/MacOS/other", Updated: now}) {
		t.Fatal("reused pid taken as live")
	}
	if bt := bootTime(); !bt.IsZero() && Live(File{PID: os.Getpid(), Updated: bt.Add(-time.Hour)}) {
		t.Fatal("file from before the boot taken as live")
	}
	d := family.At(t.TempDir())
	if err := Write(d, File{BundleID: "dev.hehehai.dicta"}); err != nil {
		t.Fatal(err)
	}
	f, _ := Read(d, "dev.hehehai.dicta")
	if f.Exe != self {
		t.Fatalf("exe %q", f.Exe)
	}
}
