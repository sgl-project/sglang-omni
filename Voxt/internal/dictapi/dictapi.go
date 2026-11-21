// Package dictapi is the socket of `Voxt dictionary import`
// (docs/dictionary-import.md): the command sends a dictionary file's
// name and bytes to the running app, which merges its terms as
// Dictionary → Import… does and answers with the counts. The command
// reads the file itself, so the app needs no access to the user's
// folders; this package only carries the request and the answer.
package dictapi

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"time"
)

// SocketName is the socket's name in the default data folder.
const SocketName = "dictionary.sock"

// OpImport is the one op of a request.
const OpImport = "import"

// MaxBytes bounds a file: dict.MaxImportRows terms of a few hundred
// bytes each fit well within it.
const MaxBytes = 8 << 20

// lineTimeout bounds reading the request line and the file, and writing
// the answer; clientWait how long the command waits for the answer (an
// import of the largest file takes well under it).
const (
	lineTimeout = 30 * time.Second
	clientWait  = 2 * time.Minute
	maxRequest  = 4 << 10
)

// ErrUnavailable: no socket to reach (Voxt is not running).
var ErrUnavailable = errors.New("dictapi: unavailable")

// Request is the command's one line; Bytes of the file follow it.
type Request struct {
	Op string `json:"op"`
	// Name is the file's base name: its extension picks the format, as
	// in Dictionary → Import… (.json a Voxt 1.x export, .txt text, else CSV).
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	DryRun bool   `json:"dry_run,omitempty"` // count, save nothing
}

// Result is what an import did, or would do with DryRun.
type Result struct {
	Added   int  `json:"added"`
	Updated int  `json:"updated"`
	Skipped int  `json:"skipped"` // rows without a usable term
	DryRun  bool `json:"dry_run,omitempty"`
}

type reply struct {
	Result *Result `json:"result,omitempty"`
	Error  string  `json:"error,omitempty"`
}

// Server answers one connection at a time, in the app's process: two
// imports never merge at once.
type Server struct {
	// Import merges the file's terms (or counts them, with dryRun).
	Import func(ctx context.Context, name string, data []byte, dryRun bool) (Result, error)
}

// Serve answers connections until ctx ends; ln closes then.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	stop := context.AfterFunc(ctx, func() { ln.Close() })
	defer stop()
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		s.handle(ctx, conn)
	}
}

func (s *Server) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()
	conn.SetDeadline(time.Now().Add(lineTimeout))
	send := func(r reply) {
		conn.SetWriteDeadline(time.Now().Add(lineTimeout))
		enc := json.NewEncoder(conn)
		enc.SetEscapeHTML(false)
		enc.Encode(r)
	}
	br := bufio.NewReaderSize(conn, maxRequest)
	line, err := br.ReadSlice('\n')
	if err != nil {
		return
	}
	var req Request
	if json.Unmarshal(line, &req) != nil || req.Op != OpImport || req.Bytes < 0 {
		send(reply{Error: "bad request"})
		return
	}
	if req.Bytes > MaxBytes {
		send(reply{Error: fmt.Sprintf("the file is larger than %d MB", MaxBytes>>20)})
		return
	}
	data := make([]byte, req.Bytes)
	if _, err := io.ReadFull(br, data); err != nil {
		send(reply{Error: "the file ended early"})
		return
	}
	res, err := s.Import(ctx, req.Name, data, req.DryRun)
	if err != nil {
		send(reply{Error: err.Error()})
		return
	}
	res.DryRun = req.DryRun
	send(reply{Result: &res})
}

// Import sends the file name and data to the app at path and returns
// what the import did.
func Import(path, name string, data []byte, dryRun bool) (Result, error) {
	if len(data) > MaxBytes {
		return Result{}, fmt.Errorf("the file is larger than %d MB", MaxBytes>>20)
	}
	conn, err := net.Dial("unix", path)
	if err != nil {
		return Result{}, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer conn.Close()
	head, err := json.Marshal(Request{Op: OpImport, Name: name, Bytes: int64(len(data)), DryRun: dryRun})
	if err != nil {
		return Result{}, err
	}
	conn.SetDeadline(time.Now().Add(clientWait))
	if _, err := conn.Write(append(head, '\n')); err != nil {
		return Result{}, fmt.Errorf("the connection to Voxt broke: %v", err)
	}
	if _, err := conn.Write(data); err != nil {
		return Result{}, fmt.Errorf("the connection to Voxt broke: %v", err)
	}
	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		return Result{}, fmt.Errorf("the connection to Voxt broke: %v", err)
	}
	var r reply
	if err := json.Unmarshal(line, &r); err != nil {
		return Result{}, fmt.Errorf("unexpected answer from Voxt: %v", err)
	}
	switch {
	case r.Error != "":
		return Result{}, errors.New(r.Error)
	case r.Result == nil:
		return Result{}, errors.New("unexpected answer from Voxt: no result")
	}
	return *r.Result, nil
}
