// Package proto is the wire format between VoxtNew and voxtnew-helper,
// the Swift helper that records the microphone, watches the keyboard,
// reads and writes text in the app in front and reads browser URLs:
// JSON-RPC-shaped messages, one per line, over the helper's stdin (app →
// helper) and stdout (helper → app). The spec is docs/helper-protocol.md;
// helper/Fixtures/*.jsonl are golden transcripts that this package's tests
// and the Swift helper's (helper/Tests) both decode, so the two sides
// cannot drift apart silently. The Swift mirror of these types is
// helper/Sources/VoxtNewHelperKit/Wire*.swift.
//
// It is a subset of AudioSmith's engine protocol 2, with the same
// envelope: no "jsonrpc" member, ids are strings, error codes are strings
// (see the Code* constants), and the helper never sends requests (only
// responses and notifications). Unlike the engine's, some notifications
// belong to no request: hotkey.event comes whenever the user presses a
// bound key, app.activated whenever another app comes to the front.
//
// This package has no dependencies and no behaviour beyond encoding: the
// supervisor (internal/helper) owns process handling.
package proto

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Version is the protocol this app speaks, sent in helper.hello.
const Version = 1

// MaxLine bounds one message on the wire, in bytes (chunks stay far below).
const MaxLine = 4 << 20

// Message is one line on the wire. Exactly one shape applies:
//
//	request       id + method (+ params)        app → helper
//	notification  method (+ params), no id      helper → app
//	response      id + result | id + error      helper → app
type Message struct {
	ID     string          `json:"id,omitempty"`
	Method string          `json:"method,omitempty"`
	Params json.RawMessage `json:"params,omitempty"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *Error          `json:"error,omitempty"`
}

// Kind is a message's shape.
type Kind int

const (
	KindInvalid Kind = iota
	KindRequest
	KindNotification
	KindResponse
)

func (k Kind) String() string {
	return [...]string{"invalid", "request", "notification", "response"}[k]
}

// Kind classifies m.
func (m *Message) Kind() Kind {
	switch {
	case m.Method != "" && m.Result == nil && m.Error == nil:
		if m.ID != "" {
			return KindRequest
		}
		return KindNotification
	case m.Method == "" && m.ID != "" && (m.Result == nil) != (m.Error == nil):
		return KindResponse
	}
	return KindInvalid
}

// Decode parses one line. It fails on malformed JSON and on messages that
// are none of the three shapes.
func Decode(line []byte) (Message, error) {
	var m Message
	if err := json.Unmarshal(line, &m); err != nil {
		return Message{}, fmt.Errorf("proto: %w", err)
	}
	if m.Kind() == KindInvalid {
		return Message{}, fmt.Errorf("proto: not a request, notification or response: %.200s", line)
	}
	return m, nil
}

// Encode returns m as one line, with its trailing newline.
func Encode(m Message) ([]byte, error) {
	data, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

// NewRequest builds a request; params may be nil.
func NewRequest(id, method string, params any) (Message, error) {
	raw, err := marshalParams(params)
	return Message{ID: id, Method: method, Params: raw}, err
}

// NewNotification builds a notification; params may be nil.
func NewNotification(method string, params any) (Message, error) {
	raw, err := marshalParams(params)
	return Message{Method: method, Params: raw}, err
}

// NewResult builds a successful response; a nil result encodes as {}.
func NewResult(id string, result any) (Message, error) {
	if result == nil {
		return Message{ID: id, Result: json.RawMessage(`{}`)}, nil
	}
	raw, err := json.Marshal(result)
	return Message{ID: id, Result: raw}, err
}

// NewError builds an error response.
func NewError(id string, e *Error) Message { return Message{ID: id, Error: e} }

func marshalParams(params any) (json.RawMessage, error) {
	if params == nil {
		return nil, nil
	}
	return json.Marshal(params)
}

// DecodeParams decodes m.Params into v. Unknown fields are ignored, so a
// newer peer may add optional fields within protocol 1 (the fixture tests
// decode strictly). Missing params decode as the zero value.
func (m *Message) DecodeParams(v any) error { return decodeRaw(m.Params, v) }

// DecodeResult decodes m.Result into v.
func (m *Message) DecodeResult(v any) error { return decodeRaw(m.Result, v) }

func decodeRaw(raw json.RawMessage, v any) error {
	if len(raw) == 0 {
		return nil
	}
	return json.Unmarshal(raw, v)
}

// Error is a response's failure.
type Error struct {
	Code    string          `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

// Is lets errors.Is(err, &proto.Error{Code: proto.CodeBusy}) match by code.
func (e *Error) Is(target error) bool {
	var t *Error
	return errors.As(target, &t) && t.Code == e.Code && t.Message == ""
}

// Errorf builds an Error with a formatted message.
func Errorf(code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// Error codes. The helper may only send these; the app treats an unknown
// code as CodeInternal.
const (
	CodeBadRequest        = "bad_request"        // malformed params
	CodeMethodNotFound    = "method_not_found"   // unknown method
	CodeNotInitialized    = "not_initialized"    // request before helper.hello
	CodeProtocolMismatch  = "protocol_mismatch"  // hello with a protocol the helper does not speak; data: {"supported":[1]}
	CodeBusy              = "busy"               // a recording is running already
	CodeCancelled         = "cancelled"          // the job was cancelled (job.cancel or shutdown)
	CodePermissionDenied  = "permission_denied"  // macOS refused a permission; data: {"permission":"microphone"|"accessibility"}
	CodeDeviceUnavailable = "device_unavailable" // no input device works (the chosen one and the default failed); data: {"device":"…"}
	CodeUnsupported       = "unsupported"        // the helper cannot do it on this Mac; data: {"capability":"…"}
	CodeTimeout           = "timeout"            // the helper gave up waiting on macOS (an app that did not answer Accessibility)
	CodeInternal          = "internal"           // anything else; message carries the detail
)

// Codes lists every error code the helper may send.
var Codes = []string{
	CodeBadRequest, CodeMethodNotFound, CodeNotInitialized, CodeProtocolMismatch, CodeBusy, CodeCancelled,
	CodePermissionDenied, CodeDeviceUnavailable, CodeUnsupported, CodeTimeout, CodeInternal,
}
