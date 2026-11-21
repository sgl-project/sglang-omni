package historyapi

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"
)

// Errors of a client call: the CLI's exit codes follow them.
var (
	// ErrUnavailable: no connection to reach (Voxt not running, or the
	// setting off).
	ErrUnavailable = errors.New("historyapi: unavailable")
	// ErrInterrupted: the reply stopped before its end line.
	ErrInterrupted = errors.New("historyapi: interrupted")
)

// clientIdle is how long a client waits for the next line: longer than a
// stuck client ahead of it in the queue can hold the server.
const clientIdle = 3 * WriteTimeout

// Export asks for rows and gives each to each, and returns how many the
// server sent.
func Export(path string, req Request, each func(Row) error) (int, error) {
	req.Op = OpExport
	return exchange(path, req, func(line []byte) error {
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			return fmt.Errorf("%w: %v", ErrInterrupted, err)
		}
		return each(r)
	})
}

// FetchStatus asks for the status.
func FetchStatus(path string) (Status, error) {
	var st Status
	_, err := exchange(path, Request{Op: OpStatus}, func(line []byte) error {
		var s statusLine
		if err := json.Unmarshal(line, &s); err != nil {
			return fmt.Errorf("%w: %v", ErrInterrupted, err)
		}
		st = s.Status
		return nil
	})
	return st, err
}

// exchange sends req and passes the reply's lines to each until the end
// line. A reply without one fails with ErrInterrupted.
func exchange(path string, req Request, each func([]byte) error) (int, error) {
	conn, err := dial(path)
	if err != nil {
		return 0, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer conn.Close()
	data, err := json.Marshal(req)
	if err != nil {
		return 0, err
	}
	if _, err := conn.Write(append(data, '\n')); err != nil {
		return 0, fmt.Errorf("%w: %v", ErrInterrupted, err)
	}
	br := bufio.NewReader(conn)
	for {
		conn.SetReadDeadline(time.Now().Add(clientIdle))
		line, err := br.ReadBytes('\n')
		if err != nil {
			if err == io.EOF || len(line) > 0 {
				return 0, fmt.Errorf("%w: reply ended early", ErrInterrupted)
			}
			return 0, fmt.Errorf("%w: %v", ErrInterrupted, err)
		}
		var f struct {
			Done  bool   `json:"done"`
			Count int    `json:"count"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal(line, &f); err != nil {
			return 0, fmt.Errorf("%w: %v", ErrInterrupted, err)
		}
		if f.Error != "" {
			return 0, fmt.Errorf("%w: %s", ErrInterrupted, f.Error)
		}
		if f.Done {
			return f.Count, nil
		}
		if err := each(line); err != nil {
			return 0, err
		}
	}
}
