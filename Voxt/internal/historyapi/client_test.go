package historyapi

import (
	"bufio"
	"encoding/json"
	"errors"
	"net"
	"path/filepath"
	"testing"
)

func TestUnavailable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.sock")
	if _, err := FetchStatus(path); !errors.Is(err, ErrUnavailable) {
		t.Errorf("FetchStatus: %v", err)
	}
	if _, err := Export(path, Request{}, func(Row) error { return nil }); !errors.Is(err, ErrUnavailable) {
		t.Errorf("Export: %v", err)
	}
}

// TestNoEndLine: a reply that stops short is interrupted, not success.
func TestNoEndLine(t *testing.T) {
	path := socketPath(t, "s.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		bufio.NewReader(c).ReadBytes('\n')
		data, _ := json.Marshal(Row{ID: "1"})
		c.Write(append(data, '\n'))
	}()
	got := 0
	_, err = Export(path, Request{}, func(Row) error { got++; return nil })
	if !errors.Is(err, ErrInterrupted) || got != 1 {
		t.Errorf("err %v, %d rows", err, got)
	}
}
