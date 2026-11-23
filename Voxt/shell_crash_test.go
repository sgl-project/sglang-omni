package main

import (
	"bytes"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"
)

// A run that panics leaves its stack in crash.log; the next run moves it
// to crash.log.1 and says so in the log (issue 3814: a crash left nothing
// to read).
func TestKeepCrashesKeepsAPanicForTheNextRun(t *testing.T) {
	dir, tmp := t.TempDir(), t.TempDir()
	cmd := exec.Command(os.Args[0], "-test.run=^$") // no tests, should TestMain not crash first
	cmd.Env = append(os.Environ(), crashDirEnv+"="+dir, "TMPDIR="+tmp)
	if err := cmd.Run(); err == nil {
		t.Fatal("the crashing run exited 0")
	}
	// A crash runs no cleanup: the crashing run must make no folders.
	if left, _ := os.ReadDir(tmp); len(left) != 0 {
		t.Errorf("the crashing run left %q in its temporary folder", left[0].Name())
	}
	crash, err := os.ReadFile(filepath.Join(dir, "crash.log"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(crash, []byte("panic: runtime error: index out of range")) || !bytes.Contains(crash, []byte("shell_crash_test.go:")) {
		t.Fatalf("crash.log holds no stack of the panic:\n%s", crash)
	}

	var logged bytes.Buffer
	log.SetOutput(&logged)
	defer log.SetOutput(os.Stderr)
	keepCrashes(dir)
	defer debug.SetCrashOutput(nil, debug.CrashOptions{})
	if moved, _ := os.ReadFile(filepath.Join(dir, "crash.log.1")); !bytes.Equal(moved, crash) {
		t.Fatalf("crash.log.1 is not the last run's crash:\n%s", moved)
	}
	if fresh, _ := os.ReadFile(filepath.Join(dir, "crash.log")); len(fresh) != 0 {
		t.Fatalf("crash.log of this run is not empty: %q", fresh)
	}
	if !strings.Contains(logged.String(), "warning: Voxt crashed last time") {
		t.Fatalf("the log does not name the crash: %q", logged.String())
	}

	// A run that ended well leaves an empty crash.log: no warning.
	logged.Reset()
	keepCrashes(dir)
	if logged.Len() != 0 {
		t.Fatalf("a clean run before logged %q", logged.String())
	}
}

// crashDirEnv names the folder the test binary, run again by
// TestKeepCrashesKeepsAPanicForTheNextRun, keeps its crash in.
const crashDirEnv = "VOXT_TEST_CRASH_DIR"

// crashInTest panics with crash.log kept in dir. TestMain calls it before
// it makes the tests' folders, which a crash would leave behind.
func crashInTest(dir string) {
	keepCrashes(dir)
	var s []int
	_ = s[len(os.Args)+2] // index out of range
}
