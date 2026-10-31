package logs

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
)

func TestRing(t *testing.T) {
	bus := core.NewBus()
	n := 0
	core.On(bus, func(core.LogAppended) { n++ })
	r := New(Options{Size: 4, Bus: bus})
	defer r.Close()
	for i := range 6 {
		src := "app"
		if i%2 == 1 {
			src = "helper:capture"
		}
		r.Append(core.LogEntry{Source: src, Message: fmt.Sprint(i)})
	}
	if n != 6 {
		t.Errorf("published %d", n)
	}
	msgs := func(es []core.LogEntry) string {
		var s []string
		for _, e := range es {
			s = append(s, e.Message)
		}
		return strings.Join(s, ",")
	}
	if got := msgs(r.Tail(10, "")); got != "2,3,4,5" {
		t.Errorf("all %s", got)
	}
	if got := msgs(r.Tail(1, "")); got != "5" {
		t.Errorf("newest %s", got)
	}
	if got := msgs(r.Tail(10, "helper:")); got != "3,5" {
		t.Errorf("helper %s", got)
	}
	if got := msgs(r.Tail(10, "app")); got != "2,4" {
		t.Errorf("app %s", got)
	}
	if e := r.Tail(1, "")[0]; e.Level != core.SeverityInfo || e.Time.IsZero() {
		t.Errorf("defaults %+v", e)
	}
	// Seq numbers the lines from 1, whatever the ring dropped.
	var seqs []uint64
	for _, e := range r.Tail(10, "") {
		seqs = append(seqs, e.Seq)
	}
	if fmt.Sprint(seqs) != "[3 4 5 6]" {
		t.Errorf("seqs %v", seqs)
	}
}

func TestFilesRotate(t *testing.T) {
	dir := t.TempDir()
	r := New(Options{Dir: dir, MaxFileBytes: 300, Keep: 2})
	l := log.New(r.Writer("app"), "", 0)
	for i := range 20 {
		l.Printf("line %02d with some padding to fill the file", i)
	}
	l.Print("an error happened")
	r.Close()
	if r.Tail(1, "")[0].Level != core.SeverityError {
		t.Error("error level not detected")
	}
	cur, err := os.ReadFile(filepath.Join(dir, FileName))
	if err != nil || !strings.Contains(string(cur), "an error happened") {
		t.Fatalf("current %q %v", cur, err)
	}
	if _, err := os.Stat(filepath.Join(dir, FileName+".2")); err != nil {
		t.Error("no second rotation:", err)
	}
	if _, err := os.Stat(filepath.Join(dir, FileName+".3")); err == nil {
		t.Error("kept more than 2 rotated files")
	}
	for _, name := range []string{FileName, FileName + ".1"} {
		st, _ := os.Stat(filepath.Join(dir, name))
		if st.Size() > 300 {
			t.Errorf("%s is %d bytes", name, st.Size())
		}
	}
	r.Close() // twice is fine
}
