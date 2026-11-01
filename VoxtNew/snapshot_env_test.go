package main

// Reproducible snapshots (refactor R0): two runs of the snapshot tests on
// one commit write the same bytes, so that a refactor is checked pixel for
// pixel against the commit before it (scripts/snapshots.sh, which compares
// with go run ./internal/tools/snapcmp BEFORE AFTER). What used to differ
// from run to run is pinned here, for the snapshot tests only (from
// AudioSmith's refactor B0):
//
//   - The clock. Each snapshot test runs in a testing/synctest bubble:
//     time.Now is snapEpoch when the test starts and moves on only when
//     every goroutine of the test waits for a timer. That pins what the
//     views print (log times, "3 min ago", dates) and the phase of spinners
//     (ui.Tester's clock starts at time.Now). time.Local is UTC, so that
//     machines in other zones draw the same.
//   - One core (GOMAXPROCS 1) while a test runs: MyGo's raster helpers are
//     goroutines kept for the life of the program, which a bubble does not
//     allow; on one core the renderer starts none. (It also suits the
//     shared Linux host.)
//   - Data directories: testDir gives <snapRoot>/001, 002, … in the order
//     asked for, instead of t.TempDir's random names, which Settings →
//     Storage and About show. snapRoot is emptied when a test starts and
//     locked against another snapshot run of the same user.
//   - Free disk space: diskFree reports snapDiskFree.
//   - The sizes Settings → Storage measures: the database's files grow by a
//     few kilobytes more or less from run to run (SQLite's pages and
//     write-ahead log), so pinStorage gives them fixed ones.
//
// The overlay's flows (VOXTNEW_FLOW_SHOTS, overlay_flows_test.go) are not
// pinned: they run the fake helper's process and loopback servers, and how
// many of a streamed draft's pieces one frame shows depends on scheduling.
// Their tests check the steps instead.
//
// A new snapshot test calls runSnapshots and makes its directories with
// testDir.

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"testing/synctest"
	"time"
)

var (
	// snapEpoch is the time at which every snapshot test starts.
	snapEpoch = time.Date(2026, 10, 16, 9, 41, 0, 0, time.UTC)
	// snapDiskFree is the free space of every disk in snapshots.
	snapDiskFree int64 = 412_300_000_000
	// snapDBSize and snapOtherSize are the sizes of the database and of the
	// data folder's other files in Settings → Storage.
	snapDBSize    int64 = 471_000
	snapOtherSize int64 = 12_300
)

// snapRun is the state of the snapshot test running, nil outside one.
var snapRun *snapState

type snapState struct {
	root string
	dirs int
}

// runSnapshots runs body, a snapshot test writing its pictures to the
// directory the environment variable env names (skipped when unset),
// reproducibly: see the top of this file.
func runSnapshots(t *testing.T, env string, body func(t *testing.T, out string)) {
	out := os.Getenv(env)
	if out == "" {
		t.Skip("set " + env + " to a directory")
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	root := lockSnapRoot(t)
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	local := time.Local
	time.Local = time.UTC
	defer func() { time.Local = local }()
	savedFree := diskFree
	diskFree = func(string) int64 { return snapDiskFree }
	defer func() { diskFree = savedFree }()
	snapRun = &snapState{root: root}
	defer func() { snapRun = nil }()
	synctest.Test(t, func(t *testing.T) {
		time.Sleep(time.Until(snapEpoch)) // a bubble's clock starts in 2000
		body(t, out)
	})
}

// lockSnapRoot empties and returns the directory under which the snapshot
// test makes its data directories, waiting for another snapshot run of the
// same user to finish with it.
func lockSnapRoot(t *testing.T) string {
	t.Helper()
	user := os.Getenv("USER")
	if user == "" {
		user = fmt.Sprint(os.Getuid())
	}
	root := filepath.Join(os.TempDir(), "voxtnew-snapshots-"+user)
	lock, err := os.OpenFile(root+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) != nil {
		t.Logf("waiting for another snapshot run to release %s.lock", root)
		if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
			t.Fatal(err)
		}
	}
	empty := func() {
		if err := os.RemoveAll(root); err != nil {
			t.Fatal(err)
		}
	}
	empty()
	t.Cleanup(func() {
		empty()
		lock.Close() // unlocks
	})
	return root
}

// testDir is t.TempDir, but in a snapshot test a directory of a fixed
// name: the next of <snapRoot>/001, 002, ….
func testDir(t *testing.T) string {
	t.Helper()
	if snapRun == nil {
		return t.TempDir()
	}
	snapRun.dirs++
	dir := filepath.Join(snapRun.root, fmt.Sprintf("%03d", snapRun.dirs))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	return dir
}

// pinStorage pins the sizes of Settings → Storage's report in a snapshot:
// the database's and the data folder's other files' (SQLite's write-ahead
// log may be written between the two measurements), and so the folder's.
func pinStorage(a *app) {
	r := a.stReport
	if snapRun == nil || r == nil {
		return
	}
	r.data = 0
	for i, c := range r.cats {
		switch c.id {
		case "database":
			r.cats[i].bytes = snapDBSize
		case "other":
			r.cats[i].bytes = snapOtherSize
		}
		if c.id != "logs" {
			r.data += r.cats[i].bytes
		}
	}
}
