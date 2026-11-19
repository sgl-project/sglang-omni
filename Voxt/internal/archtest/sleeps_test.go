package archtest

import (
	"bytes"
	"os"
	"strconv"
	"strings"
	"testing"
)

// TestSleepsInTests keeps time.Sleep out of new tests: a test waits on an
// event or a fake clock, so the suite does not grow slower or flaky with
// the machine's load. The calls there today are in sleeps.txt, a count per
// file, which may go down but not up.
func TestSleepsInTests(t *testing.T) {
	data, err := os.ReadFile("sleeps.txt")
	if err != nil {
		t.Fatal(err)
	}
	allowed := map[string]int{}
	for n, line := range strings.Split(string(data), "\n") {
		text, _, _ := strings.Cut(line, "#")
		fields := strings.Fields(text)
		if len(fields) == 0 {
			continue
		}
		count := 0
		if len(fields) == 2 {
			count, err = strconv.Atoi(fields[1])
		}
		if len(fields) != 2 || err != nil || count <= 0 {
			t.Fatalf("sleeps.txt:%d: want \"path count\", got %q", n+1, line)
		}
		allowed[fields[0]] = count
	}
	for _, f := range goFiles(t) {
		if !f.test() {
			continue
		}
		got, want := bytes.Count(f.data, []byte("time."+"Sleep(")), allowed[f.path]
		delete(allowed, f.path)
		switch {
		case got > want:
			t.Errorf("%s calls time.Sleep %d times, sleeps.txt allows %d: wait on an event or a fake clock", f.path, got, want)
		case got < want:
			t.Errorf("%s calls time.Sleep %d times: lower its count in sleeps.txt from %d (remove the line at 0)", f.path, got, want)
		}
	}
	for path := range allowed {
		t.Errorf("sleeps.txt lists %s, which is not a test file of the module: remove the line", path)
	}
}
