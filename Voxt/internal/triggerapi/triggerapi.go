// Package triggerapi is the socket of `Voxt toggle` and `Voxt cancel`
// (docs/trigger.md): another program on this Mac (a launcher, a key
// remapper, a stream deck) asks the running app to do what a feature's
// shortcut does, and the app answers once it has. It carries no text and
// no audio, only the action.
package triggerapi

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"slices"
	"time"
)

// SocketName is the socket's name in the default data folder.
const SocketName = "trigger.sock"

// The actions a request can ask for. The features' names are those of
// their shortcuts: one tap of the shortcut, so a second toggle stops what
// the first one started.
const (
	ActionTranscribe = "transcribe"
	ActionTranslate  = "translate"
	ActionRewrite    = "rewrite"
	ActionNote       = "note"
	ActionCancel     = "cancel"
)

// ToggleActions are the actions `Voxt toggle` takes, in the order its usage
// names them.
var ToggleActions = []string{ActionTranscribe, ActionTranslate, ActionRewrite, ActionNote}

// lineTimeout bounds reading the request and writing the answer;
// clientWait how long the command waits for the app's main thread.
const (
	lineTimeout = 5 * time.Second
	clientWait  = 10 * time.Second
	maxRequest  = 1 << 10
)

// ErrUnavailable: no socket to reach (Voxt is not running).
var ErrUnavailable = errors.New("triggerapi: unavailable")

// Request is the command's one line.
type Request struct {
	Action string `json:"action"`
}

type reply struct {
	OK    bool   `json:"ok,omitempty"`
	Error string `json:"error,omitempty"`
}

// Server answers one connection at a time, in the app's process.
type Server struct {
	// Fire does what the action's shortcut does; its error is the
	// reason the app did nothing, for the command to print.
	Fire func(ctx context.Context, action string) error
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
	ctx, cancel := context.WithTimeout(ctx, clientWait)
	defer cancel()
	conn.SetDeadline(time.Now().Add(lineTimeout))
	send := func(r reply) {
		conn.SetWriteDeadline(time.Now().Add(lineTimeout))
		_ = json.NewEncoder(conn).Encode(r) // the client reports a missing answer as a broken connection
	}
	line, err := bufio.NewReaderSize(conn, maxRequest).ReadSlice('\n')
	if err != nil {
		return
	}
	var request Request
	if json.Unmarshal(line, &request) != nil || !isAction(request.Action) {
		send(reply{Error: "bad request"})
		return
	}
	if err := s.Fire(ctx, request.Action); err != nil {
		send(reply{Error: err.Error()})
		return
	}
	send(reply{OK: true})
}

func isAction(action string) bool {
	return action == ActionCancel || slices.Contains(ToggleActions, action)
}

// Send asks the app at path for action and returns once the app has done it.
func Send(path, action string) error {
	if !isAction(action) {
		return fmt.Errorf("unknown action %q", action)
	}
	conn, err := net.Dial("unix", path)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer conn.Close()
	head, err := json.Marshal(Request{Action: action})
	if err != nil {
		return err
	}
	conn.SetDeadline(time.Now().Add(clientWait + lineTimeout))
	if _, err := conn.Write(append(head, '\n')); err != nil {
		return fmt.Errorf("the connection to Voxt broke: %v", err)
	}
	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("the connection to Voxt broke: %v", err)
	}
	var answer reply
	if err := json.Unmarshal(line, &answer); err != nil {
		return fmt.Errorf("unexpected answer from Voxt: %v", err)
	}
	switch {
	case answer.Error != "":
		return errors.New(answer.Error)
	case !answer.OK:
		return errors.New("unexpected answer from Voxt: no result")
	}
	return nil
}
