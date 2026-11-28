package voxtcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

// Methods Call answers.
const (
	MethodLanguages    = "languages"
	MethodPresets      = "presets"
	MethodProcess      = "process"
	MethodTestProvider = "provider.test"
	MethodTranscribe   = "speech.transcribe"
)

// Error is a failed call's error: a message for the user's log and a
// code the app tells failures apart by.
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	// Field is what a setup lacks, for a CodeConfig the app words itself
	// (Field*): the message is English, for the log.
	Field string `json:"field,omitempty"`
}

// What a language model's setup lacks (Error.Field).
const (
	FieldPreset  = "preset" // a service this core does not have
	FieldSignIn  = "sign_in"
	FieldAddress = "address"
	FieldModel   = "model"
	FieldKey     = "key"
)

// Error codes.
const (
	CodeCancelled     = "cancelled"
	CodeConfig        = "config"
	CodeNoText        = "no_text"
	CodeEmpty         = "empty"
	CodeUntranslated  = "untranslated"
	CodeMethod        = "method"
	CodeParams        = "params"
	CodeLanguageModel = "language_model"
	CodeSpeech        = "speech"
	CodeInternal      = "internal" // none of the above: a bug to report
)

// reply is every call's answer: Result, or Error.
type reply struct {
	Result any    `json:"result,omitempty"`
	Error  *Error `json:"error,omitempty"`
}

// Calls runs calls by id, so that one can be cancelled while it runs.
// The zero value is ready; HTTP is the client every request goes through
// (nil: http.DefaultClient).
type Calls struct {
	HTTP *http.Client

	mu      sync.Mutex
	running map[string]context.CancelFunc
}

// Call runs method with its JSON params and returns the JSON reply. id
// names the call for Cancel ("" for a call that is not cancelled).
func (c *Calls) Call(id, method string, params []byte) []byte {
	ctx, done := c.start(id)
	defer done()
	result, err := c.dispatch(ctx, method, params)
	var r reply
	if err != nil {
		r.Error = errorOf(ctx, err)
	} else {
		r.Result = result
	}
	out, err := json.Marshal(r)
	if err != nil {
		out, _ = json.Marshal(reply{Error: &Error{Code: CodeParams, Message: err.Error()}}) // a reply of strings always marshals
	}
	return out
}

// Cancel ends the running call id; it reports whether there was one.
func (c *Calls) Cancel(id string) bool {
	c.mu.Lock()
	cancel, ok := c.running[id]
	c.mu.Unlock()
	if ok {
		cancel()
	}
	return ok
}

func (c *Calls) start(id string) (context.Context, func()) {
	ctx, cancel := context.WithCancel(context.Background())
	if id == "" {
		return ctx, cancel
	}
	c.mu.Lock()
	if c.running == nil {
		c.running = map[string]context.CancelFunc{}
	}
	c.running[id] = cancel
	c.mu.Unlock()
	return ctx, func() {
		c.mu.Lock()
		delete(c.running, id)
		c.mu.Unlock()
		cancel()
	}
}

func (c *Calls) dispatch(ctx context.Context, method string, params []byte) (any, error) {
	switch method {
	case MethodLanguages:
		return Languages(), nil
	case MethodPresets:
		return Presets(), nil
	case MethodProcess:
		var req ProcessRequest
		if err := decode(params, &req); err != nil {
			return nil, err
		}
		return Process(ctx, req, c.HTTP)
	case MethodTestProvider:
		var p Provider
		if err := decode(params, &p); err != nil {
			return nil, err
		}
		return TestProvider(ctx, p, c.HTTP)
	case MethodTranscribe:
		var req SpeechRequest
		if err := decode(params, &req); err != nil {
			return nil, err
		}
		res, err := Transcribe(ctx, req, c.HTTP)
		if err != nil {
			err = speechError{err}
		}
		return res, err
	default:
		return nil, &Error{Code: CodeMethod, Message: fmt.Sprintf("unknown method %q", method)}
	}
}

func decode(params []byte, v any) error {
	if err := json.Unmarshal(params, v); err != nil {
		return &Error{Code: CodeParams, Message: err.Error()}
	}
	return nil
}

func (e *Error) Error() string { return e.Message }

// speechError marks a speech service's failure.
type speechError struct{ error }

func (e speechError) Unwrap() error { return e.error }

// errorOf is err as the app reads it.
func errorOf(ctx context.Context, err error) *Error {
	var e *Error
	var speech speechError
	var model *llm.Error
	switch {
	case errors.As(err, &e):
		return e
	case ctx.Err() != nil || errors.Is(err, context.Canceled):
		return &Error{Code: CodeCancelled, Message: "cancelled"}
	case errors.Is(err, ErrNoText):
		return &Error{Code: CodeNoText, Message: err.Error()}
	case errors.Is(err, ErrEmptyAnswer):
		return &Error{Code: CodeEmpty, Message: err.Error()}
	case errors.Is(err, ErrNotTranslated):
		return &Error{Code: CodeUntranslated, Message: err.Error()}
	case errors.As(err, &speech):
		var a *asr.Error
		if errors.As(err, &a) && (a.Kind == asr.KindConfig || a.Kind == asr.KindAuth) {
			return &Error{Code: CodeConfig, Message: err.Error()}
		}
		return &Error{Code: CodeSpeech, Message: err.Error()}
	case errors.As(err, &model):
		switch model.Code {
		case llm.CodeConfig, llm.CodeUnknown, llm.CodeNoKey, llm.CodeAuth: // the settings to fix
			return &Error{Code: CodeConfig, Message: err.Error()}
		default:
			return &Error{Code: CodeLanguageModel, Message: err.Error()}
		}
	case errors.Is(err, ErrTask), errors.Is(err, compose.ErrInput), errors.Is(err, compose.ErrTask):
		return &Error{Code: CodeParams, Message: err.Error()}
	default:
		return &Error{Code: CodeInternal, Message: err.Error()}
	}
}
