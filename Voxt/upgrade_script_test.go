package main

// scripts/upgrade-test.sh (the Voxt Upgrade Test workflow) only runs on a
// GitHub macOS runner; here a step runs with some commands replaced by
// fakes that fail, at the front of PATH. The run's Summary is the script's
// results table: a command that stops a step must leave a ❌ row naming the
// step and the command, and a failed $(...) inside a step that goes on must
// not leave one that says the step stopped.

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestUpgradeScriptRows(t *testing.T) {
	for _, c := range []struct {
		step  string
		fakes []string
		want  string // in results.md
		stops bool   // a "stopped at line" row
	}{
		// prepare starts with open -g "$APP": a plain command, not a check.
		{"prepare", []string{"open"}, `open -g "$APP"`, true},
		// check-app records every check and goes on; the team row's
		// $(grep ...) finds nothing in an empty codesign.txt.
		{"check-app", []string{"codesign", "spctl", "xcrun"}, "| team | ❌ | (want RY2TQTYSKQ) |", false},
	} {
		bin, out := t.TempDir(), t.TempDir()
		for _, fake := range c.fakes {
			if err := os.WriteFile(filepath.Join(bin, fake), []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command("bash", "scripts/upgrade-test.sh", c.step)
		cmd.Env = append(os.Environ(), "PATH="+bin+string(os.PathListSeparator)+os.Getenv("PATH"),
			"HOME="+t.TempDir(), "OUT="+out, "FROM_TAG=voxt-v1.16.3", "TO_VERSION=2.0.2")
		printed, err := cmd.CombinedOutput()
		results, _ := os.ReadFile(filepath.Join(out, "results.md"))
		stopped := strings.Contains(string(results), "| "+c.step+" | ❌ | stopped at line ")
		if err == nil || !strings.Contains(string(results), c.want) || stopped != c.stops {
			t.Errorf("%s with %v failing: %v\nresults.md:\n%s\nprinted:\n%s", c.step, c.fakes, err, results, printed)
		}
	}
}
