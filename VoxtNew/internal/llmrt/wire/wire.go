// Package wire is the wire format between VoxtNew and voxtnew-llm, the
// Swift process that runs the language models: the helper protocol's
// envelope (JSON objects, one per line, over the process's stdin and
// stdout) with methods of its own. The spec is docs/llm-protocol.md;
// llm/Fixtures/*.jsonl are golden transcripts that this package's tests
// and the Swift package's (llm/Tests) both decode. The Swift mirror of
// these types is llm/Sources/VoxtNewLLMKit/Wire*.swift.
//
// This package has no dependencies and no behaviour beyond encoding:
// internal/llmrt owns the process.
package wire

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Version is the protocol this app speaks, sent in llm.hello.
const Version = 1

// MaxLine bounds one message on the wire, in bytes.
const MaxLine = 4 << 20

// Message is one line on the wire. Exactly one shape applies:
//
//	request       id + method (+ params)        app → process
//	notification  method (+ params), no id      process → app
//	response      id + result | id + error      process → app
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
		return Message{}, fmt.Errorf("wire: %w", err)
	}
	if m.Kind() == KindInvalid {
		return Message{}, fmt.Errorf("wire: not a request, notification or response: %.200s", line)
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
	raw, err := marshal(params)
	return Message{ID: id, Method: method, Params: raw}, err
}

// NewNotification builds a notification; params may be nil.
func NewNotification(method string, params any) (Message, error) {
	raw, err := marshal(params)
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

func marshal(v any) (json.RawMessage, error) {
	if v == nil {
		return nil, nil
	}
	return json.Marshal(v)
}

// DecodeParams decodes m.Params into v (unknown fields are ignored;
// missing params decode as the zero value).
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

// Is lets errors.Is(err, &wire.Error{Code: wire.CodeCancelled}) match by code.
func (e *Error) Is(target error) bool {
	var t *Error
	return errors.As(target, &t) && t.Code == e.Code && t.Message == ""
}

// Error codes. The process may only send these; the app treats an unknown
// code as CodeInternal.
const (
	CodeBadRequest       = "bad_request"
	CodeMethodNotFound   = "method_not_found"
	CodeNotInitialized   = "not_initialized"
	CodeProtocolMismatch = "protocol_mismatch" // data: {"supported":[1]}
	CodeCancelled        = "cancelled"
	CodeLoadFailed       = "load_failed"
	CodeContextLength    = "context_length"
	CodeInternal         = "internal"
)

// Codes lists every error code the process may send.
var Codes = []string{
	CodeBadRequest, CodeMethodNotFound, CodeNotInitialized, CodeProtocolMismatch,
	CodeCancelled, CodeLoadFailed, CodeContextLength, CodeInternal,
}
