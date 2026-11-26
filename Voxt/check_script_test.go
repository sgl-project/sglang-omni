package main

// Voxt CI runs the fast checks through scripts/check.sh, so a developer
// running the script locally runs what CI runs. This fails if the workflow
// stops calling the script or the script drops one of its checks.

import (
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func TestCICallsCheckScript(t *testing.T) {
	ci, err := os.ReadFile("../.github/workflows/voxtnew-ci.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(ci), "run: scripts/check.sh\n") {
		t.Error("Voxt CI does not run scripts/check.sh")
	}
	script, err := os.ReadFile("scripts/check.sh")
	if err != nil {
		t.Fatal(err)
	}
	for _, check := range []string{"xargs -0 gofmt -l", "go tool mygo vet .", "go test -count=1 ./internal/archtest"} {
		if !strings.Contains(string(script), check) {
			t.Errorf("scripts/check.sh no longer runs %q", check)
		}
	}
}

func TestCIChecksBrowserAndFixtureDemo(t *testing.T) {
	ci, err := os.ReadFile("../.github/workflows/voxtnew-ci.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, check := range []string{
		"CGO_ENABLED=0 go test -count=1 -v -run '^TestPageSanitizesTranscriptInBrowser$' ./internal/transcriptpreview",
		"grep -q -- '--- PASS: TestPageSanitizesTranscriptInBrowser'",
		"CGO_ENABLED=0 go build -tags meetingdemo -o /dev/null .",
		"CGO_ENABLED=0 go vet -tags meetingdemo .",
		"GOFLAGS=-tags=meetingdemo go tool mygo vet .",
	} {
		if !strings.Contains(string(ci), check) {
			t.Errorf("Voxt CI no longer runs %q", check)
		}
	}
}

// TestReleaseTestsHaveRoom: the release workflow's Go tests set their own
// time limit. The app's package runs for up to ten minutes on the macOS
// runner, which is go test's default limit for a package.
func TestReleaseTestsHaveRoom(t *testing.T) {
	data, err := os.ReadFile("../.github/workflows/voxt-release.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "go test -count=1 -timeout 30m ./...") {
		t.Error("voxt-release.yaml runs go test without -timeout 30m")
	}
}

// TestCIShardsRunEveryTest: Voxt CI runs this package's tests as shards
// beside each other, each with the names scripts/test-shard.sh deals it. A
// name left out of every shard would be a test CI never runs.
func TestCIShardsRunEveryTest(t *testing.T) {
	ci, err := os.ReadFile("../.github/workflows/voxtnew-ci.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(ci), "go test -list . . | scripts/test-shard.sh ${{ matrix.shard }}") {
		t.Error("Voxt CI does not take its shards from scripts/test-shard.sh")
	}
	names := []string{"TestA", "TestAB", "ExampleC", "FuzzD", "TestE", "TestF", "TestG"}
	list := strings.Join(names, "\n") + "\nok  \tpkg\t0.01s\n"
	const shards = 3
	ran := map[string]int{}
	for shard := range shards {
		cmd := exec.Command("scripts/test-shard.sh", strconv.Itoa(shard), strconv.Itoa(shards))
		cmd.Stdin = strings.NewReader(list)
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("shard %d: %v", shard, err)
		}
		pattern := regexp.MustCompile(strings.TrimSpace(string(out)))
		for _, name := range names {
			if pattern.MatchString(name) {
				ran[name]++
			}
		}
	}
	for _, name := range names {
		if ran[name] != 1 {
			t.Errorf("%s runs in %d shards, want 1", name, ran[name])
		}
	}
}
